import os
from typing import List
from libinfinicore_infer import (
    MixtralMetaCStruct,
    MixtralWeightsCStruct,
    KVCacheCStruct,
    DataType,
    DeviceType,
    create_mixtral_model,
    destroy_mixtral_model,
    create_mixtral_kv_cache,
    drop_mixtral_kv_cache,
    infer_batch_mixtral,
)
from infer_task import InferTask, KVCache

from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref
from pathlib import Path
import safetensors
import sys
import time
import json
import math
import torch
import transformers

torch.set_default_device("cpu")


class MixtralWeightsNaming:
    def input_embd(self):
        return "model.embed_tokens.weight"

    def output_norm(self):
        return "model.norm.weight"

    def output_embd(self):
        return "lm_head.weight"

    def attn_norm(self, i):
        return f"model.layers.{i}.input_layernorm.weight"

    def attn_q(self, i):
        return f"model.layers.{i}.self_attn.q_proj.weight"

    def attn_k(self, i):
        return f"model.layers.{i}.self_attn.k_proj.weight"

    def attn_v(self, i):
        return f"model.layers.{i}.self_attn.v_proj.weight"

    def attn_o(self, i):
        return f"model.layers.{i}.self_attn.o_proj.weight"

    def ffn_norm(self, i):
        return f"model.layers.{i}.post_attention_layernorm.weight"

    def ffn_gate(self, i):
        return f"model.layers.{i}.block_sparse_moe.gate.weight"

    def ffn_gate_up(self, i, expert_idx):
        return f"model.layers.{i}.block_sparse_moe.experts.{expert_idx}.w1.weight"

    def ffn_down(self, i, expert_idx):
        return f"model.layers.{i}.block_sparse_moe.experts.{expert_idx}.w2.weight"
    
    def ffn_up(self, i, expert_idx):
        return f"model.layers.{i}.block_sparse_moe.experts.{expert_idx}.w3.weight"

    def match(state_dict):
        return "model.layers.0.block_sparse_moe.gate.weight" in state_dict


class MixtralMetaFromConfig(MixtralMetaCStruct):
    def __init__(self, config, dtype=torch.float16, max_tokens=None):
        dt_ = DataType.INFINI_DTYPE_F16
        if dtype == torch.float32:
            dt_ = DataType.INFINI_DTYPE_F32
        elif dtype == torch.bfloat16:
            dt_ = DataType.INFINI_DTYPE_BF16

        super().__init__(
            dt_logits=dt_,
            nlayer=config["num_hidden_layers"],
            d=config["hidden_size"],
            nh=config["num_attention_heads"],
            nkvh=config.get("num_key_value_heads", config["num_attention_heads"]),
            dh=config["hidden_size"] // config["num_attention_heads"],
            di=config["intermediate_size"],
            dctx=config.get("max_position_embeddings", max_tokens),
            dvoc=config["vocab_size"],
            nexpert=config.get("num_local_experts", 1),
            topk=config.get("num_experts_per_tok", 1),
            epsilon=config["rms_norm_eps"],
            theta=config.get("rope_theta", 10000.0),
            sliding_window=config.get("sliding_window", -1),
            end_token=config["eos_token_id"],
        )
        self.torch_dtype_logits = dtype


class MixtralWeightsImpl(MixtralWeightsCStruct):
    def __init__(
        self,
        meta,
        naming,
        state_dict,
        torch_dt_mat=torch.float16,
        torch_dt_norm=torch.float32,
        ndev=1,
        transpose_weight=True,
    ):
        nlayer = meta.nlayer
        nh = meta.nh
        nkvh = meta.nkvh
        dh = meta.dh
        d = meta.d
        di = meta.di
        
        torch_dt_logits = meta.torch_dtype_logits
        if torch_dt_mat == torch.float16:
            self.dt_mat = DataType.INFINI_DTYPE_F16
        elif torch_dt_mat == torch.float32:
            self.dt_mat = DataType.INFINI_DTYPE_F32
        elif torch_dt_mat == torch.bfloat16:
            self.dt_mat = DataType.INFINI_DTYPE_BF16
        else:
            raise ValueError("Unsupported proj weight data type")

        if torch_dt_norm == torch.float16:
            self.dt_norm = DataType.INFINI_DTYPE_F16
        elif torch_dt_norm == torch.float32:
            self.dt_norm = DataType.INFINI_DTYPE_F32
        elif torch_dt_norm == torch.bfloat16:
            self.dt_norm = DataType.INFINI_DTYPE_BF16
        else:
            raise ValueError("Unsupported norm weight data type")

        self.transpose_linear_weights = 1 if transpose_weight else 0
        self.nlayer = nlayer
        self.input_embd_tensor = (
            state_dict[naming.input_embd()].to(torch_dt_logits)
        )
        self.input_embd = self.input_embd_tensor.data_ptr()
        self.output_norm_tensor = (
            state_dict[naming.output_norm()].to(torch_dt_norm)
        )
        self.output_norm = self.output_norm_tensor.data_ptr()
        self.output_embd_tensor = state_dict[naming.output_embd()].to(torch_dt_mat)
        if not transpose_weight:
            self.output_embd_tensor = self.output_embd_tensor.transpose(
                0, 1
            ).contiguous()
        self.output_embd = self.output_embd_tensor.data_ptr()

        self.attn_norm_tensors = [
            state_dict[naming.attn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.attn_norm_ptrs = [
            self.attn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.attn_norm = (c_void_p * nlayer)(*self.attn_norm_ptrs)

        def qkv_slices(_i):
            _Q = (
                state_dict[naming.attn_q(_i)]
                .reshape([nh, 2, dh // 2, d])
                .transpose(1, 2)
            )
            _K = (
                state_dict[naming.attn_k(_i)]
                .reshape([nkvh, 2, dh // 2, d])
                .transpose(1, 2)
            )
            _V = state_dict[naming.attn_v(_i)].reshape([nkvh, dh // 2, 2, d])
            _result = []
            _nh = nh // ndev
            _nkvh = nkvh // ndev
            for _idev in range(ndev):
                _result.append(_Q[_idev * _nh : (_idev + 1) * _nh, :, :, :])
                _result.append(_K[_idev * _nkvh : (_idev + 1) * _nkvh, :, :, :])
                _result.append(_V[_idev * _nkvh : (_idev + 1) * _nkvh, :, :])
            return _result

        self.qkv_tensor = [
            torch.concat(qkv_slices(i)).to(torch_dt_mat) for i in range(nlayer)
        ]
        if not transpose_weight:
            for i in range(nlayer):
                self.qkv_tensor[i] = (
                    self.qkv_tensor[i]
                    .reshape(ndev, (nh + 2 * nkvh) // ndev * dh, d)
                    .transpose(1, 2)
                    .contiguous()
                )
        self.qkv_tensor_ptrs = [self.qkv_tensor[i].data_ptr() for i in range(nlayer)]
        self.attn_qkv = (c_void_p * nlayer)(*self.qkv_tensor_ptrs)

        self.attn_qkv_b = None

        self.attn_o_tensor = [
            (
                state_dict[naming.attn_o(i)]
                .to(torch_dt_mat)
                .reshape([d, ndev, nh // ndev * dh])
                .transpose(0, 1)
                .contiguous()
                if transpose_weight
                else state_dict[naming.attn_o(i)]
                .transpose(0, 1)
                .to(torch_dt_mat)
                .contiguous()
            )
            for i in range(nlayer)
        ]
        self.attn_o_ptrs = [self.attn_o_tensor[i].data_ptr() for i in range(nlayer)]
        self.attn_o = (c_void_p * nlayer)(*self.attn_o_ptrs)

        self.ffn_norm_tensors = [
            state_dict[naming.ffn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.ffn_norm_ptrs = [
            self.ffn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.ffn_norm = (c_void_p * nlayer)(*self.ffn_norm_ptrs)

        if meta.nexpert > 0:
            self.ffn_gate_tensors = [
                state_dict[naming.ffn_gate(i)].to(torch_dt_mat)
                for i in range(meta.nlayer)
            ]
            self.ffn_gate_ptrs = [t.data_ptr() for t in self.ffn_gate_tensors]
            self.ffn_gate = (c_void_p * meta.nlayer)(*self.ffn_gate_ptrs)

            self.ffn_gate_up_tensors = []
            self.ffn_down_tensors = []
            
            for i in range(meta.nlayer):
                gate_up_experts = []
                down_experts = []
                for j in range(meta.nexpert):
                    gate = state_dict[naming.ffn_gate_up(i, j)].to(torch_dt_mat)
                    up = state_dict[naming.ffn_up(i, j)].to(torch_dt_mat)
                    gate_up_experts.append(torch.cat([gate, up], dim=0))
                    down_experts.append(state_dict[naming.ffn_down(i, j)].to(torch_dt_mat))

                self.ffn_gate_up_tensors.append(gate_up_experts)
                self.ffn_down_tensors.append(down_experts)

            # Create nested pointers for C API
            self.ffn_gate_up_expert_ptrs = [
                (c_void_p * meta.nexpert)(*[t.data_ptr() for t in layer_tensors])
                for layer_tensors in self.ffn_gate_up_tensors
            ]
            self.ffn_gate_up = (POINTER(c_void_p) * meta.nlayer)(*self.ffn_gate_up_expert_ptrs)

            self.ffn_down_expert_ptrs = [
                (c_void_p * meta.nexpert)(*[t.data_ptr() for t in layer_tensors])
                for layer_tensors in self.ffn_down_tensors
            ]
            self.ffn_down = (POINTER(c_void_p) * meta.nlayer)(*self.ffn_down_expert_ptrs)
        else: # Non-MoE
            self.ffn_gate_up_tensors = []
            self.ffn_down_tensors = []
            
            for i in range(meta.nlayer):
                gate = state_dict[f"model.layers.{i}.mlp.gate_proj.weight"].to(torch_dt_mat)
                up = state_dict[f"model.layers.{i}.mlp.up_proj.weight"].to(torch_dt_mat)
                self.ffn_gate_up_tensors.append([torch.cat([gate, up], dim=0)])
                self.ffn_down_tensors.append([state_dict[f"model.layers.{i}.mlp.down_proj.weight"].to(torch_dt_mat)])

            self.ffn_gate_up_expert_ptrs = [
                (c_void_p * 1)(*[t.data_ptr() for t in layer_tensors])
                for layer_tensors in self.ffn_gate_up_tensors
            ]
            self.ffn_gate_up = (POINTER(c_void_p) * meta.nlayer)(*self.ffn_gate_up_expert_ptrs)
            self.ffn_down_expert_ptrs = [
                (c_void_p * 1)(*[t.data_ptr() for t in layer_tensors])
                for layer_tensors in self.ffn_down_tensors
            ]
            self.ffn_down = (POINTER(c_void_p) * meta.nlayer)(*self.ffn_down_expert_ptrs)


class MixtralBatchedTask:
    def __init__(self, tasks: List[InferTask]):
        self.tasks = tasks
        self.nreq = len(tasks)

        # Precompute fields
        token_lists = [t.tokens for t in tasks]
        self.req_lens_list = [len(toks) for toks in token_lists]
        self.req_pos_list = [t.pos for t in tasks]
        self.kv_cache_ptrs = [t.kvcache().data() for t in tasks]
        self.temperaturas_list = [t.temperature for t in tasks]
        self.topks_list = [t.topk for t in tasks]
        self.topps_list = [t.topp for t in tasks]

        # Flatten token lists
        flat_tokens = [tok for toks in token_lists for tok in toks]
        self.ntok = len(flat_tokens)

        # Convert to ctypes arrays in one pass
        self.tokens = (c_uint * self.ntok)(*flat_tokens)
        self.req_lens = (c_uint * self.nreq)(*self.req_lens_list)
        self.req_pos = (c_uint * self.nreq)(*self.req_pos_list)
        self.kv_caches = (POINTER(KVCacheCStruct) * self.nreq)(*self.kv_cache_ptrs)
        self.temperaturas = (c_float * self.nreq)(*self.temperaturas_list)
        self.topks = (c_uint * self.nreq)(*self.topks_list)
        self.topps = (c_float * self.nreq)(*self.topps_list)

    def input_args(self):
        return (
            self.tokens,
            self.ntok,
            self.req_lens,
            self.nreq,
            self.req_pos,
            self.kv_caches,
            self.temperaturas,
            self.topks,
            self.topps,
        )

class MixtralForCauslLM:
    def __init__(
        self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None
    ):
        print("Loading model weights to host...")
        load_start_time = time.time()
        
        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
        self.config = config
        eos_token_id = self.config["eos_token_id"]
        self.eos_token_id = (
            [eos_token_id] if type(eos_token_id) == int else eos_token_id
        )
        
        state_dict = {}
        with open(os.path.join(model_dir_path, "pytorch_model.bin.index.json"), "r") as f:
            index = json.load(f)
            files = sorted(list(set(index["weight_map"].values())))
        for file in files:
            p = os.path.join(model_dir_path, file)
            state_dict.update(torch.load(p, map_location="cpu"))

        naming = MixtralWeightsNaming()
        self.meta = MixtralMetaFromConfig(config, max_tokens=max_tokens)
        self.weights = MixtralWeightsImpl(self.meta, naming, state_dict, ndev=ndev)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir_path)

        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")
        
        print(f"Creating model on {ndev} devices...")
        create_start_time = time.time()
        dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
        self.model_instance = create_mixtral_model(
            byref(self.meta),
            byref(self.weights),
            device,
            ndev,
            dev_ids,
        )
        create_end_time = time.time()
        print(f"Time used: {create_end_time - create_start_time:.3f}s")

    def max_context_len(self):
        return self.meta.dctx

    def create_kv_cache(self):
        return create_mixtral_kv_cache(self.model_instance)

    def drop_kv_cache(self, kv_cache):
        return drop_mixtral_kv_cache(self.model_instance, kv_cache)

    def batch_infer_one_round(self, tasks: List[InferTask]):
        output = (c_uint * len(tasks))()
        batch_inputs = MixtralBatchedTask(tasks) 
        infer_batch_mixtral(
            self.model_instance,
            *(batch_inputs.input_args()),
            output,
        )
        return list(output)

    def generate(self, input_content, max_steps, topp_=1.0, topk_=1, temperature_=1.0):
        print(input_content, end="", flush=True)
        tokens = self.tokenizer.encode(input_content)
        infer_task = InferTask(
            0,
            tokens,
            self.max_context_len(),
            temperature_,
            topk_,
            topp_,
            self.eos_token_id,
        )
        infer_task.bind_kvcache(KVCache(self))

        steps = 0
        total_time = 0
        output_content = ""

        for step_i in range(max_steps):
            start_time = time.time()
            output_tokens = self.batch_infer_one_round([infer_task])
            end_time = time.time()
            steps += 1
            output_str = (
                self.tokenizer._tokenizer.id_to_token(output_tokens[0])
                .replace(" ", " ")
                .replace("<0x0A>", "\n")
            )
            output_content += output_str
            print(output_str, end="", flush=True)
            if output_tokens[0] in self.eos_token_id:
                break
            infer_task.next(output_tokens[0])

            if step_i > 0:
                total_time += end_time - start_time

        print("\n")
        avg_time = total_time * 1000 / (steps - 1) if steps > 1 else 0
        print(f"Time per step: {avg_time:.3f}ms")

        infer_task._kv_cache.drop(self)
        return output_content, avg_time

    def destroy_model_instance(self):
        destroy_mixtral_model(self.model_instance)
        print("Model destroyed")


def test():
    if len(sys.argv) < 3:
        print(
            "Usage: python mixtral.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)
    model_path = sys.argv[2]
    device_type = DeviceType.DEVICE_TYPE_CPU
    if sys.argv[1] == "--cpu":
        device_type = DeviceType.DEVICE_TYPE_CPU
    elif sys.argv[1] == "--nvidia":
        device_type = DeviceType.DEVICE_TYPE_NVIDIA
    elif sys.argv[1] == "--cambricon":
        device_type = DeviceType.DEVICE_TYPE_CAMBRICON
    elif sys.argv[1] == "--ascend":
        device_type = DeviceType.DEVICE_TYPE_ASCEND
    elif sys.argv[1] == "--metax":
        device_type = DeviceType.DEVICE_TYPE_METAX
    elif sys.argv[1] == "--moore":
        device_type = DeviceType.DEVICE_TYPE_MOORE
    else:
        print(
            "Usage: python mixtral.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model = MixtralForCauslLM(model_path, device_type, ndev)
    model.generate("Once upon a time", 100)
    model.destroy_model_instance()


if __name__ == "__main__":
    test()