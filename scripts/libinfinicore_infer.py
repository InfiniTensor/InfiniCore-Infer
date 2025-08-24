import ctypes
from ctypes import c_size_t, c_uint, c_int, c_float, c_void_p, POINTER
import os


class DataType(ctypes.c_int):
    INFINI_DTYPE_INVALID = 0
    INFINI_DTYPE_BYTE = 1
    INFINI_DTYPE_BOOL = 2
    INFINI_DTYPE_I8 = 3
    INFINI_DTYPE_I16 = 4
    INFINI_DTYPE_I32 = 5
    INFINI_DTYPE_I64 = 6
    INFINI_DTYPE_U8 = 7
    INFINI_DTYPE_U16 = 8
    INFINI_DTYPE_U32 = 9
    INFINI_DTYPE_U64 = 10
    INFINI_DTYPE_F8 = 11
    INFINI_DTYPE_F16 = 12
    INFINI_DTYPE_F32 = 13
    INFINI_DTYPE_F64 = 14
    INFINI_DTYPE_C16 = 15
    INFINI_DTYPE_C32 = 16
    INFINI_DTYPE_C64 = 17
    INFINI_DTYPE_C128 = 18
    INFINI_DTYPE_BF16 = 19


class DeviceType(ctypes.c_int):
    DEVICE_TYPE_CPU = 0
    DEVICE_TYPE_NVIDIA = 1
    DEVICE_TYPE_CAMBRICON = 2
    DEVICE_TYPE_ASCEND = 3
    DEVICE_TYPE_METAX = 4
    DEVICE_TYPE_MOORE = 5
    DEVICE_TYPE_ILUVATAR = 6


class JiugeMetaCStruct(ctypes.Structure):
    _fields_ = [
        ("dt_logits", DataType),
        ("nlayer", c_size_t),
        ("d", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("dctx", c_size_t),
        ("dvoc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_uint),
    ]


# Define the JiugeWeights struct
class JiugeWeightsCStruct(ctypes.Structure):
    _fields_ = [
        ("nlayer", c_size_t),
        ("dt_norm", DataType),
        ("dt_mat", DataType),
        ("transpose_linear_weights", c_int),
        ("input_embd", c_void_p),
        ("output_norm", c_void_p),
        ("output_embd", c_void_p),
        ("attn_norm", POINTER(c_void_p)),
        ("attn_qkv", POINTER(c_void_p)),
        ("attn_qkv_b", POINTER(c_void_p)),
        ("attn_o", POINTER(c_void_p)),
        ("ffn_norm", POINTER(c_void_p)),
        ("ffn_gate_up", POINTER(c_void_p)),
        ("ffn_down", POINTER(c_void_p)),
    ]


class JiugeModelCSruct(ctypes.Structure):
    pass


class DeepSeekV3MetaCStruct(ctypes.Structure):
    _fields_ = [
        # dtypes
        ("dt_logits", DataType),
        ("dt_norm", DataType),
        ("dt_quant_weight", DataType),
        ("dt_quant_scale", DataType),
        ("dt_quant_zero", DataType),
        ("dt_gate_weight", DataType),
        ("dt_gate_bias", DataType),
        # sizes
        ("n_sparse_layer", c_size_t),
        ("n_dense_layer", c_size_t),
        ("d", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("d_rope", c_size_t),
        ("d_nope", c_size_t),
        ("r_q", c_size_t),
        ("r_kv", c_size_t),
        ("d_qk", c_size_t),
        ("d_v", c_size_t),
        # routing / experts / vocab / ctx
        ("routed_scale", c_float),
        ("nexperts", c_size_t),
        ("kexperts", c_size_t),
        ("di", c_size_t),
        ("di_moe", c_size_t),
        ("dctx", c_size_t),
        ("dvoc", c_size_t),
        # misc
        ("epsilon", c_float),
        ("rope_theta", c_float),
        ("end_token", c_uint),
    ]


class DeepSeekV3WeightsCStruct(ctypes.Structure):
    pass


# void (*load_global_fn)(DeepSeekV3Weights*, void *cpu_ptr)
load_global_fn = ctypes.CFUNCTYPE(None, POINTER(DeepSeekV3WeightsCStruct), c_void_p)

# void (*load_layer_fn)(DeepSeekV3Weights*, void *cpu_ptr, size_t layer_id)
load_layer_fn = ctypes.CFUNCTYPE(
    None, POINTER(DeepSeekV3WeightsCStruct), c_void_p, c_size_t
)

# void (*load_layer_linear_fn)(DeepSeekV3Weights*, void *weight_ptr, void *scale_ptr, void *zero_ptr, size_t layer_id)
load_layer_linear_fn = ctypes.CFUNCTYPE(
    None, POINTER(DeepSeekV3WeightsCStruct), c_void_p, c_void_p, c_void_p, c_size_t
)

# void (*load_layer_mlp_fn)(DeepSeekV3Weights*, ... , size_t layer_id)
load_layer_mlp_fn = ctypes.CFUNCTYPE(
    None,
    POINTER(DeepSeekV3WeightsCStruct),
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_size_t,
)

# void (*load_layer_expert_mlp_fn)(DeepSeekV3Weights*, ..., size_t layer_id, size_t expert_id)
load_layer_expert_mlp_fn = ctypes.CFUNCTYPE(
    None,
    POINTER(DeepSeekV3WeightsCStruct),
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_size_t,
    c_size_t,
)
# -------------------------------------------------------------------
# Struct containing all weight loading functions
# -------------------------------------------------------------------


class DeepSeekV3WeightLoaderCStruct(ctypes.Structure):
    _fields_ = [
        # Global
        ("load_input_embd", load_global_fn),
        ("load_output_norm", load_global_fn),
        ("load_output_embd", load_global_fn),
        # Attention
        ("load_attn_norm", load_layer_fn),
        ("load_attn_q_a_proj", load_layer_linear_fn),
        ("load_attn_q_a_layernorm", load_layer_fn),
        ("load_attn_q_b_proj", load_layer_linear_fn),
        ("load_attn_kv_a_proj_with_mqa", load_layer_linear_fn),
        ("load_attn_kv_a_layernorm", load_layer_fn),
        ("load_attn_kv_b_proj", load_layer_linear_fn),
        ("load_attn_o_proj", load_layer_linear_fn),
        # MLP
        ("load_mlp_norm", load_layer_fn),
        # MLP dense part
        ("load_mlp_dense", load_layer_mlp_fn),
        # MLP sparse gating
        ("load_mlp_gate_weight", load_layer_fn),
        ("load_mlp_gate_bias", load_layer_fn),
        # Shared experts
        ("load_mlp_shared_experts", load_layer_mlp_fn),
        # Per-expert functions
        ("load_mlp_experts", load_layer_expert_mlp_fn),
    ]


class DeepSeekV3ModelCStruct(ctypes.Structure):
    pass


class KVCacheCStruct(ctypes.Structure):
    pass


class DeepSeekV3CacheCStruct(ctypes.Structure):
    pass


def __open_library__():
    lib_path = os.path.join(
        os.environ.get("INFINI_ROOT"), "lib", "libinfinicore_infer.so"
    )
    lib = ctypes.CDLL(lib_path)
    lib.createJiugeModel.restype = POINTER(JiugeModelCSruct)
    lib.createJiugeModel.argtypes = [
        POINTER(JiugeMetaCStruct),  # JiugeMeta const *
        POINTER(JiugeWeightsCStruct),  # JiugeWeights const *
        DeviceType,  # DeviceType
        c_int,  # int ndev
        POINTER(c_int),  # int const *dev_ids
    ]
    lib.destroyJiugeModel.argtypes = [POINTER(JiugeModelCSruct)]
    lib.createKVCache.argtypes = [POINTER(JiugeModelCSruct)]
    lib.createKVCache.restype = POINTER(KVCacheCStruct)
    lib.dropKVCache.argtypes = [POINTER(JiugeModelCSruct), POINTER(KVCacheCStruct)]
    lib.inferBatch.restype = None
    lib.inferBatch.argtypes = [
        POINTER(JiugeModelCSruct),  # struct JiugeModel const *
        POINTER(c_uint),  # unsigned int const *tokens
        c_uint,  # unsigned int ntok
        POINTER(c_uint),  # unsigned int const *req_lens
        c_uint,  # unsigned int nreq
        POINTER(c_uint),  # unsigned int const *req_pos
        POINTER(POINTER(KVCacheCStruct)),  # struct KVCache **kv_caches
        POINTER(c_float),  # float temperature
        POINTER(c_uint),  # unsigned int topk
        POINTER(c_float),  # float topp
        POINTER(c_uint),  # unsigned int *output
    ]
    lib.forwardBatch.restype = None
    lib.forwardBatch.argtypes = [
        POINTER(JiugeModelCSruct),  # struct JiugeModel const *
        POINTER(c_uint),  # unsigned int const *tokens
        c_uint,  # unsigned int ntok
        POINTER(c_uint),  # unsigned int const *req_lens
        c_uint,  # unsigned int nreq
        POINTER(c_uint),  # unsigned int const *req_pos
        POINTER(POINTER(KVCacheCStruct)),  # struct KVCache **kv_caches
        c_void_p,  # void *logits
    ]

    # createDeepSeekV3WeightLoader
    lib.createDeepSeekV3WeightLoader.argtypes = []
    lib.createDeepSeekV3WeightLoader.restype = POINTER(DeepSeekV3WeightLoaderCStruct)

    lib.createDeepSeekV3Weights.argtypes = [
        POINTER(DeepSeekV3MetaCStruct),
        DeviceType,
        c_int,
        POINTER(c_int),
    ]
    lib.createDeepSeekV3Weights.restype = POINTER(DeepSeekV3WeightsCStruct)

    lib.createDeepSeekV3Model.argtypes = [
        POINTER(DeepSeekV3MetaCStruct),
        POINTER(DeepSeekV3WeightsCStruct),
    ]
    lib.createDeepSeekV3Model.restype = POINTER(DeepSeekV3ModelCStruct)

    # destroyDeepSeekV3Model
    lib.destroyDeepSeekV3Model.argtypes = [POINTER(DeepSeekV3ModelCStruct)]
    lib.destroyDeepSeekV3Model.restype = None

    # createDeepSeekV3Cache
    lib.createDeepSeekV3Cache.argtypes = [POINTER(DeepSeekV3ModelCStruct)]
    lib.createDeepSeekV3Cache.restype = POINTER(DeepSeekV3CacheCStruct)

    # dropDeepSeekV3Cache
    lib.dropDeepSeekV3Cache.argtypes = [
        POINTER(DeepSeekV3ModelCStruct),
        POINTER(DeepSeekV3CacheCStruct),
    ]
    lib.dropDeepSeekV3Cache.restype = None

    # inferBatchDeepSeekV3
    lib.inferBatchDeepSeekV3.argtypes = [
        POINTER(DeepSeekV3ModelCStruct),
        POINTER(c_uint),
        c_uint,
        POINTER(c_uint),
        c_uint,
        POINTER(c_uint),
        POINTER(POINTER(DeepSeekV3CacheCStruct)),
        POINTER(c_float),
        POINTER(c_uint),
        POINTER(c_float),
        POINTER(c_uint),
    ]
    lib.inferBatchDeepSeekV3.restype = None

    # forwardBatchDeepSeekV3
    lib.forwardBatchDeepSeekV3.argtypes = [
        POINTER(DeepSeekV3ModelCStruct),
        POINTER(c_uint),
        c_uint,
        POINTER(c_uint),
        c_uint,
        POINTER(c_uint),
        POINTER(POINTER(DeepSeekV3CacheCStruct)),
        c_void_p,
    ]
    lib.forwardBatchDeepSeekV3.restype = None

    return lib


LIB = __open_library__()

create_jiuge_model = LIB.createJiugeModel
destroy_jiuge_model = LIB.destroyJiugeModel
create_kv_cache = LIB.createKVCache
drop_kv_cache = LIB.dropKVCache
infer_batch = LIB.inferBatch
forward_batch = LIB.forwardBatch

create_deepseek_v3_model = LIB.createDeepSeekV3Model
destroy_deepseek_v3_model = LIB.destroyDeepSeekV3Model
create_deepseek_v3_weight_loader = LIB.createDeepSeekV3WeightLoader
create_deepseek_v3_weights = LIB.createDeepSeekV3Weights
create_deepseek_v3_cache = LIB.createDeepSeekV3Cache
drop_deepseek_v3_cache = LIB.dropDeepSeekV3Cache
infer_batch_deepseek_v3 = LIB.inferBatchDeepSeekV3
