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


class KVCacheCStruct(ctypes.Structure):
    pass


class TinyMixMetaCStruct(ctypes.Structure):
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
        ("nexpert", c_size_t),
        ("n_expert_activate", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_uint),
    ]

class TinyMixWeightsCStruct(ctypes.Structure):
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
        ("ffn_gate_up", POINTER(POINTER(c_void_p))),
        ("ffn_down", POINTER(POINTER(c_void_p))),
        ("ffn_gate", POINTER(c_void_p)),
    ]

class TinyMixModelCSruct(ctypes.Structure):
    pass


class MixtralMetaCStruct(ctypes.Structure):
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
        ("nexpert", c_size_t),
        ("topk", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("sliding_window", c_int),
        ("end_token", c_uint),
    ]


class MixtralWeightsCStruct(ctypes.Structure):
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
        ("ffn_gate_up", POINTER(POINTER(c_void_p))),
        ("ffn_down", POINTER(POINTER(c_void_p))),
        ("ffn_gate", POINTER(c_void_p)),
    ]


class MixtralModelCSruct(ctypes.Structure):
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

    # TinyMix API
    lib.createTinyMixModel.restype = POINTER(TinyMixModelCSruct)
    lib.createTinyMixModel.argtypes = [
        POINTER(TinyMixMetaCStruct),
        POINTER(TinyMixWeightsCStruct),
        DeviceType,
        c_int,
        POINTER(c_int),
    ]
    lib.destroyTinyMixModel.argtypes = [POINTER(TinyMixModelCSruct)]
    lib.createTinyMixKVCache.argtypes = [POINTER(TinyMixModelCSruct)]
    lib.createTinyMixKVCache.restype = POINTER(KVCacheCStruct)
    lib.dropTinyMixKVCache.argtypes = [POINTER(TinyMixModelCSruct), POINTER(KVCacheCStruct)]
    lib.inferBatchTinyMix.restype = None
    lib.inferBatchTinyMix.argtypes = [
        POINTER(TinyMixModelCSruct),
        POINTER(c_uint),
        c_uint,
        POINTER(c_uint),
        c_uint,
        POINTER(c_uint),
        POINTER(POINTER(KVCacheCStruct)),
        POINTER(c_float),
        POINTER(c_uint),
        POINTER(c_float),
        POINTER(c_uint),
    ]

    # Mixtral API
    lib.createMixtralModel.restype = POINTER(MixtralModelCSruct)
    lib.createMixtralModel.argtypes = [
        POINTER(MixtralMetaCStruct),
        POINTER(MixtralWeightsCStruct),
        DeviceType,
        c_int,
        POINTER(c_int),
    ]
    lib.destroyMixtralModel.argtypes = [POINTER(MixtralModelCSruct)]
    lib.createMixtralKVCache.argtypes = [POINTER(MixtralModelCSruct)]
    lib.createMixtralKVCache.restype = POINTER(KVCacheCStruct)
    lib.dropMixtralKVCache.argtypes = [
        POINTER(MixtralModelCSruct),
        POINTER(KVCacheCStruct),
    ]
    lib.inferBatchMixtral.restype = None
    lib.inferBatchMixtral.argtypes = [
        POINTER(MixtralModelCSruct),
        POINTER(c_uint),
        c_uint,
        POINTER(c_uint),
        c_uint,
        POINTER(c_uint),
        POINTER(POINTER(KVCacheCStruct)),
        POINTER(c_float),
        POINTER(c_uint),
        POINTER(c_float),
        POINTER(c_uint),
    ]

    return lib


LIB = __open_library__()

create_jiuge_model = LIB.createJiugeModel
destroy_jiuge_model = LIB.destroyJiugeModel
create_kv_cache = LIB.createKVCache
drop_kv_cache = LIB.dropKVCache
infer_batch = LIB.inferBatch

create_tinymix_model = LIB.createTinyMixModel
destroy_tinymix_model = LIB.destroyTinyMixModel
create_tinymix_kv_cache = LIB.createTinyMixKVCache
drop_tinymix_kv_cache = LIB.dropTinyMixKVCache
infer_batch_tinymix = LIB.inferBatchTinyMix

create_mixtral_model = LIB.createMixtralModel
destroy_mixtral_model = LIB.destroyMixtralModel
create_mixtral_kv_cache = LIB.createMixtralKVCache
drop_mixtral_kv_cache = LIB.dropMixtralKVCache
infer_batch_mixtral = LIB.inferBatchMixtral
