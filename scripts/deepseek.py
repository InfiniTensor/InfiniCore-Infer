import ctypes
from typing import List, Sequence

from tqdm import tqdm

from libinfinicore_infer import (
    DeepSeekV3MetaCStruct,
    DeepSeekV3CacheCStruct,
    DataType,
    DeviceType,
    create_deepseek_v3_model,
    create_deepseek_v3_weights,
    create_deepseek_v3_weight_loader,
    destroy_deepseek_v3_model,
    create_deepseek_v3_cache,
    drop_deepseek_v3_cache,
    infer_batch_deepseek_v3,
)
from infer_task import InferTask, KVCache

from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref
import os
from pathlib import Path
import safetensors
import sys
import time
import json
import math
import torch
import transformers

torch.set_default_device("cpu")


class DeepseekR1WeightsNaming:
    def __init__(self, dense_replace=3):
        self.dense_replace = dense_replace

    def input_embd(self):
        return "model.embed_tokens.weight"

    def output_norm(self):
        return "model.norm.weight"

    def output_embd(self):
        return "lm_head.weight"

    # MLA
    def attn_norm(self, i):
        return f"model.layers.{i}.input_layernorm.weight"

    def attn_kv_a_layernorm(self, i):
        return f"model.layers.{i}.self_attn.kv_a_layernorm.weight"

    def attn_kv_a_proj_with_mqa_weight(self, i):
        return f"model.layers.{i}.self_attn.kv_a_proj_with_mqa.qweight"

    def attn_kv_a_proj_with_mqa_scale(self, i):
        return f"model.layers.{i}.self_attn.kv_a_proj_with_mqa.scales"

    def attn_kv_a_proj_with_mqa_zero(self, i):
        return f"model.layers.{i}.self_attn.kv_a_proj_with_mqa.qzeros"

    def attn_kv_b_proj_weight(self, i):
        return f"model.layers.{i}.self_attn.kv_b_proj.qweight"

    def attn_kv_b_proj_scale(self, i):
        return f"model.layers.{i}.self_attn.kv_b_proj.scales"

    def attn_kv_b_proj_zero(self, i):
        return f"model.layers.{i}.self_attn.kv_b_proj.qzeros"

    def attn_o_proj_weight(self, i):
        return f"model.layers.{i}.self_attn.o_proj.qweight"

    def attn_o_proj_scale(self, i):
        return f"model.layers.{i}.self_attn.o_proj.scales"

    def attn_o_proj_zero(self, i):
        return f"model.layers.{i}.self_attn.o_proj.qzeros"

    def attn_q_a_layernorm(self, i):
        return f"model.layers.{i}.self_attn.q_a_layernorm.weight"

    def attn_q_a_proj_weight(self, i):
        return f"model.layers.{i}.self_attn.q_a_proj.qweight"

    def attn_q_a_proj_scale(self, i):
        return f"model.layers.{i}.self_attn.q_a_proj.scales"

    def attn_q_a_proj_zero(self, i):
        return f"model.layers.{i}.self_attn.q_a_proj.qzeros"

    def attn_q_b_proj_weight(self, i):
        return f"model.layers.{i}.self_attn.q_b_proj.qweight"

    def attn_q_b_proj_scale(self, i):
        return f"model.layers.{i}.self_attn.q_b_proj.scales"

    def attn_q_b_proj_zero(self, i):
        return f"model.layers.{i}.self_attn.q_b_proj.qzeros"

    # MLP

    def mlp_norm(self, i):
        return f"model.layers.{i}.post_attention_layernorm.weight"

    # First self.dense_replace layers are dense
    def mlp_down_proj_weight(self, i):
        assert i < self.dense_replace
        return f"model.layers.{i}.mlp.down_proj.qweight"

    def mlp_down_proj_scale(self, i):
        assert i < self.dense_replace
        return f"model.layers.{i}.mlp.down_proj.scales"

    def mlp_down_proj_zero(self, i):
        assert i < self.dense_replace
        return f"model.layers.{i}.mlp.down_proj.qzeros"

    def mlp_up_proj_weight(self, i):
        assert i < self.dense_replace
        return f"model.layers.{i}.mlp.up_proj.qweight"

    def mlp_up_proj_scale(self, i):
        assert i < self.dense_replace
        return f"model.layers.{i}.mlp.up_proj.scales"

    def mlp_up_proj_zero(self, i):
        assert i < self.dense_replace
        return f"model.layers.{i}.mlp.up_proj.qzeros"

    def mlp_gate_proj_weight(self, i):
        assert i < self.dense_replace
        return f"model.layers.{i}.mlp.gate_proj.qweight"

    def mlp_gate_proj_scale(self, i):
        assert i < self.dense_replace
        return f"model.layers.{i}.mlp.gate_proj.scales"

    def mlp_gate_proj_zero(self, i):
        assert i < self.dense_replace
        return f"model.layers.{i}.mlp.gate_proj.qzeros"

    # Latter layers are sparse
    # Gating
    def mlp_gate_weight(self, i):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.gate.weight"

    def mlp_gate_bias(self, i):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.gate.e_score_correction_bias"

    # Experts
    def mlp_shared_experts_down_proj_weight(self, i):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.shared_experts.down_proj.qweight"

    def mlp_shared_experts_down_proj_scale(self, i):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.shared_experts.down_proj.scales"

    def mlp_shared_experts_down_proj_zero(self, i):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.shared_experts.down_proj.qzeros"

    def mlp_shared_experts_gate_proj_weight(self, i):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.shared_experts.gate_proj.qweight"

    def mlp_shared_experts_gate_proj_scale(self, i):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.shared_experts.gate_proj.scales"

    def mlp_shared_experts_gate_proj_zero(self, i):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.shared_experts.gate_proj.qzeros"

    def mlp_shared_experts_up_proj_weight(self, i):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.shared_experts.up_proj.qweight"

    def mlp_shared_experts_up_proj_scale(self, i):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.shared_experts.up_proj.scales"

    def mlp_shared_experts_up_proj_zero(self, i):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.shared_experts.up_proj.qzeros"

    # Experts
    def mlp_experts_down_proj_weight(self, i, e):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.experts.{e}.down_proj.qweight"

    def mlp_experts_down_proj_scale(self, i, e):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.experts.{e}.down_proj.scales"

    def mlp_experts_down_proj_zero(self, i, e):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.experts.{e}.down_proj.qzeros"

    def mlp_experts_gate_proj_weight(self, i, e):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.experts.{e}.gate_proj.qweight"

    def mlp_experts_gate_proj_scale(self, i, e):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.experts.{e}.gate_proj.scales"

    def mlp_experts_gate_proj_zero(self, i, e):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.experts.{e}.gate_proj.qzeros"

    def mlp_experts_up_proj_weight(self, i, e):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.experts.{e}.up_proj.qweight"

    def mlp_experts_up_proj_scale(self, i, e):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.experts.{e}.up_proj.scales"

    def mlp_experts_up_proj_zero(self, i, e):
        assert i >= self.dense_replace
        return f"model.layers.{i}.mlp.experts.{e}.up_proj.qzeros"


class DeepSeekV3Meta(DeepSeekV3MetaCStruct):
    def __init__(self, config, dtype=torch.float16, max_tokens=None):
        if dtype == torch.float16:
            dt_ = DataType.INFINI_DTYPE_F16
        elif dtype == torch.bfloat16:
            dt_ = DataType.INFINI_DTYPE_BF16
        else:
            dt_ = DataType.INFINI_DTYPE_F16

        super().__init__(
            # dtypes
            dt_logits=DataType.INFINI_DTYPE_F16,
            dt_norm=DataType.INFINI_DTYPE_BF16,
            dt_quant_weight=DataType.INFINI_DTYPE_I32,
            dt_quant_scale=DataType.INFINI_DTYPE_F16,
            dt_quant_zero=DataType.INFINI_DTYPE_I32,
            dt_gate_weight=DataType.INFINI_DTYPE_BF16,
            dt_gate_bias=DataType.INFINI_DTYPE_BF16,
            # sizes
            n_sparse_layer=config["num_hidden_layers"],
            n_dense_layer=config.get("first_k_dense_replace", 0),
            d=config["hidden_size"],
            nh=config["num_attention_heads"],
            nkvh=config.get("num_key_value_heads", config["num_attention_heads"]),
            d_rope=config["qk_rope_head_dim"],
            d_nope=config["qk_nope_head_dim"],
            r_q=config["q_lora_rank"],
            r_kv=config["kv_lora_rank"],
            d_qk=config["qk_nope_head_dim"] + config["qk_rope_head_dim"],
            d_v=config["v_head_dim"],
            # routing / experts / vocab / ctx
            routed_scale=config.get("routed_scaling_factor", 1.0),
            nexperts=config["n_routed_experts"],
            kexperts=config["num_experts_per_tok"],
            di=config["intermediate_size"],
            di_moe=config["moe_intermediate_size"],
            dctx=(
                config["max_position_embeddings"] if max_tokens is None else max_tokens
            ),
            dvoc=config["vocab_size"],
            # misc
            epsilon=config.get("rms_norm_eps", 1e-6),
            rope_theta=config.get("rope_theta", 10000.0),
            end_token=config.get("eos_token_id", 2),
        )
        self.torch_dtype_logits = dtype


def load_specific_tensor(model_dir, tensor_name):
    """
    Load a specific tensor from a sharded safetensors model using its index JSON.
    """
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")

    with open(index_file, "r") as f:
        index = json.load(f)

    # Get mapping: tensor name -> file name
    weight_map = index["weight_map"]
    if tensor_name not in weight_map:
        raise KeyError(f"{tensor_name} not found in index")

    filename = weight_map[tensor_name]
    tensor_file = os.path.join(model_dir, filename)

    # Open only the relevant file and tensor
    with safetensors.safe_open(tensor_file, framework="pt", device="cpu") as f:
        tensor = f.get_tensor(tensor_name)
    return tensor


def load_deepseek_weights(
    meta: DeepSeekV3Meta,
    weights,
    model_path: str,
    ndev: int,
):
    weight_loader = create_deepseek_v3_weight_loader()
    names = DeepseekR1WeightsNaming()
    input_embd = load_specific_tensor(model_path, names.input_embd()).to(meta.torch_dtype_logits)
    weight_loader.contents.load_input_embd(weights, input_embd.data_ptr())
    del input_embd

    output_norm = load_specific_tensor(model_path, names.output_norm())
    weight_loader.contents.load_output_norm(weights, output_norm.data_ptr())
    del output_norm

    output_embd = load_specific_tensor(model_path, names.output_embd())
    weight_loader.contents.load_output_embd(weights, output_embd.data_ptr())
    del output_embd

    # -------------------------------
    # Per-layer weights
    # -------------------------------

    def load_quant(w_name, s_name, zero_name, split_dim=0):
        weight = load_specific_tensor(model_path, w_name)
        scale = load_specific_tensor(model_path, s_name)
        zero = load_specific_tensor(model_path, zero_name)
        if split_dim == 0 or ndev == 1:
            return weight, scale, zero
        elif split_dim == 1:
            weight = (
                weight.reshape(weight.shape[0], ndev, -1).permute(1, 0, 2).contiguous()
            )
            scale = (
                scale.reshape(scale.shape[0], ndev, -1).permute(1, 0, 2).contiguous()
            )
            zero = zero.reshape(zero.shape[0], ndev, -1).permute(1, 0, 2).contiguous()
            return weight, scale, zero
        else:
            raise ValueError("split_dim must be 0 or 1")

    for i in tqdm(
        range(meta.n_sparse_layer + meta.n_dense_layer), desc="Loading layers"
    ):

        # Attention norms + projections
        attn_norm = load_specific_tensor(model_path, names.attn_norm(i))
        weight_loader.contents.load_attn_norm(weights, attn_norm.data_ptr(), i)
        del attn_norm

        load_attn_q_a_layernorm = load_specific_tensor(
            model_path, names.attn_q_a_layernorm(i)
        )
        weight_loader.contents.load_attn_q_a_layernorm(
            weights, load_attn_q_a_layernorm.data_ptr(), i
        )
        del load_attn_q_a_layernorm

        attn_kv_a_layernorm = load_specific_tensor(
            model_path, names.attn_kv_a_layernorm(i)
        )
        weight_loader.contents.load_attn_kv_a_layernorm(
            weights, attn_kv_a_layernorm.data_ptr(), i
        )
        del attn_kv_a_layernorm

        w, s, z = load_quant(
            names.attn_q_a_proj_weight(i),
            names.attn_q_a_proj_scale(i),
            names.attn_q_a_proj_zero(i),
        )
        weight_loader.contents.load_attn_q_a_proj(
            weights, w.data_ptr(), s.data_ptr(), z.data_ptr(), i
        )

        w, s, z = load_quant(
            names.attn_q_b_proj_weight(i),
            names.attn_q_b_proj_scale(i),
            names.attn_q_b_proj_zero(i),
        )
        weight_loader.contents.load_attn_q_b_proj(
            weights, w.data_ptr(), s.data_ptr(), z.data_ptr(), i
        )

        w, s, z = load_quant(
            names.attn_kv_a_proj_with_mqa_weight(i),
            names.attn_kv_a_proj_with_mqa_scale(i),
            names.attn_kv_a_proj_with_mqa_zero(i),
        )
        weight_loader.contents.load_attn_kv_a_proj_with_mqa(
            weights, w.data_ptr(), s.data_ptr(), z.data_ptr(), i
        )

        w, s, z = load_quant(
            names.attn_kv_b_proj_weight(i),
            names.attn_kv_b_proj_scale(i),
            names.attn_kv_b_proj_zero(i),
        )

        weight_loader.contents.load_attn_kv_b_proj(
            weights, w.data_ptr(), s.data_ptr(), z.data_ptr(), i
        )

        w, s, z = load_quant(
            names.attn_o_proj_weight(i),
            names.attn_o_proj_scale(i),
            names.attn_o_proj_zero(i),
            1,
        )

        weight_loader.contents.load_attn_o_proj(
            weights, w.data_ptr(), s.data_ptr(), z.data_ptr(), i
        )

        # -------------------------------
        # MLP: dense or sparse
        # -------------------------------
        mlp_norm = load_specific_tensor(model_path, names.mlp_norm(i))
        weight_loader.contents.load_mlp_norm(weights, mlp_norm.data_ptr(), i)

        if i < meta.n_dense_layer:
            # Dense MLP is grouped into one call
            w_gate, s_gate, z_gate = load_quant(
                names.mlp_gate_proj_weight(i),
                names.mlp_gate_proj_scale(i),
                names.mlp_gate_proj_zero(i),
            )
            w_up, s_up, z_up = load_quant(
                names.mlp_up_proj_weight(i),
                names.mlp_up_proj_scale(i),
                names.mlp_up_proj_zero(i),
            )
            w_down, s_down, z_down = load_quant(
                names.mlp_down_proj_weight(i),
                names.mlp_down_proj_scale(i),
                names.mlp_down_proj_zero(i),
                1,
            )
            weight_loader.contents.load_mlp_dense(
                weights,
                w_gate.data_ptr(),
                s_gate.data_ptr(),
                z_gate.data_ptr(),
                w_up.data_ptr(),
                s_up.data_ptr(),
                z_up.data_ptr(),
                w_down.data_ptr(),
                s_down.data_ptr(),
                z_down.data_ptr(),
                i,
            )

        else:
            # Sparse MLP gating
            mlp_gate_weight = load_specific_tensor(model_path, names.mlp_gate_weight(i))
            weight_loader.contents.load_mlp_gate_weight(
                weights, mlp_gate_weight.data_ptr(), i
            )
            del mlp_gate_weight

            mlp_gate_bias = load_specific_tensor(model_path, names.mlp_gate_bias(i))
            weight_loader.contents.load_mlp_gate_bias(
                weights, mlp_gate_bias.data_ptr(), i
            )
            del mlp_gate_bias

            # Shared experts
            w_gate, s_gate, z_gate = load_quant(
                names.mlp_shared_experts_gate_proj_weight(i),
                names.mlp_shared_experts_gate_proj_scale(i),
                names.mlp_shared_experts_gate_proj_zero(i),
            )
            w_up, s_up, z_up = load_quant(
                names.mlp_shared_experts_up_proj_weight(i),
                names.mlp_shared_experts_up_proj_scale(i),
                names.mlp_shared_experts_up_proj_zero(i),
            )
            w_down, s_down, z_down = load_quant(
                names.mlp_shared_experts_down_proj_weight(i),
                names.mlp_shared_experts_down_proj_scale(i),
                names.mlp_shared_experts_down_proj_zero(i),
                1,
            )
            weight_loader.contents.load_mlp_shared_experts(
                weights,
                w_gate.data_ptr(),
                s_gate.data_ptr(),
                z_gate.data_ptr(),
                w_up.data_ptr(),
                s_up.data_ptr(),
                z_up.data_ptr(),
                w_down.data_ptr(),
                s_down.data_ptr(),
                z_down.data_ptr(),
                i,
            )

            # Per-expert MLP
            for e in range(meta.nexperts):
                w_gate, s_gate, z_gate = load_quant(
                    names.mlp_experts_gate_proj_weight(i, e),
                    names.mlp_experts_gate_proj_scale(i, e),
                    names.mlp_experts_gate_proj_zero(i, e),
                )
                w_up, s_up, z_up = load_quant(
                    names.mlp_experts_up_proj_weight(i, e),
                    names.mlp_experts_up_proj_scale(i, e),
                    names.mlp_experts_up_proj_zero(i, e),
                )
                w_down, s_down, z_down = load_quant(
                    names.mlp_experts_down_proj_weight(i, e),
                    names.mlp_experts_down_proj_scale(i, e),
                    names.mlp_experts_down_proj_zero(i, e),
                    1,
                )
                weight_loader.contents.load_mlp_experts(
                    weights,
                    w_gate.data_ptr(),
                    s_gate.data_ptr(),
                    z_gate.data_ptr(),
                    w_up.data_ptr(),
                    s_up.data_ptr(),
                    z_up.data_ptr(),
                    w_down.data_ptr(),
                    s_down.data_ptr(),
                    z_down.data_ptr(),
                    i,
                    e,
                )


class DeepSeekV3BatchedTask:
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
        self.kv_caches = (POINTER(DeepSeekV3CacheCStruct) * self.nreq)(
            *self.kv_cache_ptrs
        )
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


class DeepSeekV3ForCauslLM:
    def __init__(
        self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None
    ):
        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config
        eos_token_id = self.config["eos_token_id"]
        self.eos_token_id = (
            [eos_token_id] if type(eos_token_id) == int else eos_token_id
        )

        print(model_dir_path)

        if "deepseek_v3" == config["model_type"]:
            self.meta = DeepSeekV3Meta(config, max_tokens=max_tokens, dtype=torch.float16)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir_path)
        else:
            raise ValueError("Unsupported model architecture")

        print(f"Creating model on {ndev} devices...")
        load_start_time = time.time()
        dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
        weights = create_deepseek_v3_weights(
            self.meta,
            device,
            ndev,
            dev_ids,
        )
        # Load weights from host
        # load_deepseek_weights(self.meta, weights, model_dir_path, ndev)
        # Create model instance
        self.model_instance = create_deepseek_v3_model(
            byref(self.meta),
            weights,
        )
        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")

    def max_context_len(self):
        return self.meta.dctx

    def create_kv_cache(self):
        return create_deepseek_v3_cache(self.model_instance)

    def drop_kv_cache(self, kv_cache):
        drop_deepseek_v3_cache(self.model_instance, kv_cache)

    def batch_infer_one_round(self, tasks: List[InferTask]):
        output = (c_uint * len(tasks))()
        batch_inputs = DeepSeekV3BatchedTask(tasks)
        infer_batch_deepseek_v3(
            self.model_instance,
            *(batch_inputs.input_args()),
            output,
        )
        return list(output)

    def generate(self, input_content, max_steps, topp_=1.0, topk_=1, temperature_=1.0):
        input_content = self.tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": input_content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        
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
        print(input_content, end="", flush=True)
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
                .replace("▁", " ")
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
        avg_time = total_time * 1000 / (steps - 1)
        print(f"Time per step: {avg_time:.3f}ms")

        infer_task._kv_cache.drop(self)
        return output_content, avg_time

    # def perplexity(self, test_sequences: List[Sequence[int]], batch_size=10):
    #     tasks = [
    #         InferTask(i, [], self.max_context_len(), 1.0, 1, 1.0, self.eos_token_id)
    #         for i in range(batch_size)
    #     ]
    #     kv_caches = [KVCache(self) for _ in range(batch_size)]

    #     nll = 0.0
    #     total_len = 0

    #     for i in range(0, len(test_sequences), batch_size):
    #         batch_id = 0
    #         true_tokens = []
    #         while batch_id < batch_size and batch_id + i < len(test_sequences):
    #             input_tokens = test_sequences[i + batch_id][:-1]
    #             true_tokens.extend(test_sequences[i + batch_id][1:])
    #             tasks[batch_id].tokens = input_tokens
    #             tasks[batch_id].bind_kvcache(kv_caches[batch_id])
    #             batch_id += 1

    #         batch_inputs = DeepSeekV3BatchedTask(tasks[:batch_id])
    #         logits = torch.zeros(
    #             (batch_inputs.ntok, self.meta.dvoc), dtype=self.meta.torch_dtype_logits
    #         )
    #         forward_batch_deepseek_v3(
    #             self.model_instance,
    #             batch_inputs.tokens,
    #             batch_inputs.ntok,
    #             batch_inputs.req_lens,
    #             batch_inputs.nreq,
    #             batch_inputs.req_pos,
    #             batch_inputs.kv_caches,
    #             logits.data_ptr(),
    #         )

    #         logits = logits.float()
    #         token_ids = torch.tensor(true_tokens, dtype=torch.int64)  # [ntok,]
    #         log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (ntok, vocab)
    #         token_logprobs = log_probs[
    #             torch.arange(batch_inputs.ntok), token_ids
    #         ]  # (ntok,)

    #         start = 0
    #         for l in batch_inputs.req_lens_list:
    #             nll += -token_logprobs[start : start + l].sum().item()
    #             start += l
    #         total_len += token_logprobs.numel()

    #     for task in tasks:
    #         task.release_kvcache()

    #     return math.exp(nll / total_len)

    def destroy_model_instance(self):
        destroy_deepseek_v3_model(self.model_instance)
        print("Model destroyed")


def test():
    if len(sys.argv) < 3:
        print(
            "Usage: python deepseek.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
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
    elif sys.argv[1] == "--iluvatar":
        device_type = DeviceType.DEVICE_TYPE_ILUVATAR
    else:
        print(
            "Usage: python deepseek.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model = DeepSeekV3ForCauslLM(model_path, device_type, ndev, max_tokens=1024)
    model.generate("山东最高的山是？", 50)
    model.destroy_model_instance()


if __name__ == "__main__":
    test()
