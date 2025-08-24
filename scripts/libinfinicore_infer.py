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
        ("attn_q_norm", POINTER(c_void_p)),
        ("attn_k_norm", POINTER(c_void_p)),
        ("attn_o", POINTER(c_void_p)),
        ("ffn_norm", POINTER(c_void_p)),
        ("ffn_gate_up", POINTER(c_void_p)),
        ("ffn_down", POINTER(c_void_p)),
    ]


class JiugeModelCSruct(ctypes.Structure):
    pass


class KVCacheCStruct(ctypes.Structure):
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

    return lib


LIB = __open_library__()

create_jiuge_model = LIB.createJiugeModel
destroy_jiuge_model = LIB.destroyJiugeModel
create_kv_cache = LIB.createKVCache
drop_kv_cache = LIB.dropKVCache
infer_batch = LIB.inferBatch
forward_batch = LIB.forwardBatch
