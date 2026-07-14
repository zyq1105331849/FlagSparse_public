"""Sparse triangular solve (SpSV) CSR/COO."""

import ctypes

from . import _common as _common_mod
from ._common import *

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
import os
import time
import triton

hip = _common_mod.hip
hipsparse = _common_mod.hipsparse
HipPointer = _common_mod.HipPointer
_benchmark_prepared_cuda_op = _common_mod._benchmark_prepared_cuda_op
_hip_check_result = _common_mod._hip_check_result
_hipsparse_lookup = _common_mod._hipsparse_lookup
_hipsparse_unavailable_reason = _common_mod._hipsparse_unavailable_reason
_hipsparse_value_type = _common_mod._hipsparse_value_type
_hipsparse_scalar = _common_mod._hipsparse_scalar
_hipsparse_index_type = _common_mod._hipsparse_index_type
_hipsparse_spmv_operation = _common_mod._hipsparse_spmv_operation

SUPPORTED_SPSV_VALUE_DTYPES = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)
SUPPORTED_SPSV_INDEX_DTYPES = (torch.int32, torch.int64)
SPSV_NON_TRANS_SUPPORTED_COMBOS = (
    (torch.float32, torch.int32),
    (torch.float64, torch.int32),
    (torch.complex64, torch.int32),
    (torch.complex128, torch.int32),
    (torch.float32, torch.int64),
    (torch.float64, torch.int64),
    (torch.complex64, torch.int64),
    (torch.complex128, torch.int64),
)
SPSV_TRANS_SUPPORTED_COMBOS = (
    (torch.float32, torch.int32),
    (torch.float64, torch.int32),
    (torch.complex64, torch.int32),
    (torch.complex128, torch.int32),
    (torch.float32, torch.int64),
    (torch.float64, torch.int64),
    (torch.complex64, torch.int64),
    (torch.complex128, torch.int64),
)
def _spsv_env_flag(name, default="0"):
    return str(os.environ.get(name, default)).lower() in ("1", "true", "yes", "on")


def _spsv_env_warp_size(name, default):
    value = int(os.environ.get(name, default))
    if value not in (32, 64):
        raise ValueError(f"{name} must be 32 or 64, got {value}")
    return value


SPSV_PROMOTE_FP32_TO_FP64 = _spsv_env_flag("FLAGSPARSE_SPSV_PROMOTE_FP32_TO_FP64", "0")
SPSV_PROMOTE_TRANSPOSE_FP32_TO_FP64 = _spsv_env_flag(
    "FLAGSPARSE_SPSV_PROMOTE_TRANSPOSE_FP32_TO_FP64", "0"
)
SPSV_PROMOTE_TRANSPOSE_COMPLEX64_TO_COMPLEX128 = _spsv_env_flag(
    "FLAGSPARSE_SPSV_PROMOTE_TRANSPOSE_COMPLEX64_TO_COMPLEX128", "0"
)
SPSV_ROCM_ENABLE_ADVANCED_AUTO = _spsv_env_flag(
    "FLAGSPARSE_SPSV_ROCM_ENABLE_ADVANCED_AUTO", "0"
)
SPSV_ROCM_ALG3_WARP_SIZE = _spsv_env_warp_size(
    "FLAGSPARSE_SPSV_ROCM_ALG3_WARP_SIZE", "64"
)
_SPSV_CSR_PREPROCESS_CACHE = OrderedDict()
_SPSV_CSR_PREPROCESS_CACHE_SIZE = 8


def _torch_current_stream_ptr():
    stream = torch.cuda.current_stream()
    for attr_name in ("cuda_stream", "hip_stream"):
        stream_ptr = getattr(stream, attr_name, None)
        if callable(stream_ptr):
            stream_ptr = stream_ptr()
        if stream_ptr is not None:
            return int(stream_ptr)
    return None


def _try_set_hipsparse_current_stream(handle):
    set_stream = getattr(hipsparse, "hipsparseSetStream", None)
    if set_stream is None:
        return "hipSPARSE binding does not expose hipsparseSetStream"

    stream_ptr = _torch_current_stream_ptr()
    if stream_ptr is None:
        return "could not resolve torch current CUDA/HIP stream pointer"

    stream_args = []
    if HipPointer is not None:
        try:
            stream_args.append(HipPointer.fromObj(stream_ptr))
        except Exception:
            pass
    stream_args.extend([stream_ptr, ctypes.c_void_p(stream_ptr)])

    last_error = None
    for stream_arg in stream_args:
        try:
            _hip_check_result(set_stream(handle, stream_arg), "hipsparseSetStream")
            return None
        except TypeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
    return f"hipsparseSetStream failed for torch current stream: {last_error}"


def _hipsparse_spsv_op(op):
    return _hipsparse_spmv_operation(op, "hipSPARSE CSR SpSV")


def _hipsparse_spsv_alg():
    return _hipsparse_lookup(
        "hipsparseSpSVAlg_t",
        (
            "HIPSPARSE_SPSV_ALG_DEFAULT",
            "HIPSPARSE_SPSV_CSR_ALG1",
        ),
    )


def _hipsparse_spmat_attribute(name):
    mapping = {
        "fill_mode": ("HIPSPARSE_SPMAT_FILL_MODE",),
        "diag_type": ("HIPSPARSE_SPMAT_DIAG_TYPE",),
    }
    if name not in mapping:
        raise RuntimeError(f"Unsupported hipSPARSE SpMat attribute: {name}")
    return _hipsparse_lookup("hipsparseSpMatAttribute_t", mapping[name])


def _hipsparse_fill_mode_enum(lower):
    return _hipsparse_lookup(
        "hipsparseFillMode_t",
        ("HIPSPARSE_FILL_MODE_LOWER",)
        if lower
        else ("HIPSPARSE_FILL_MODE_UPPER",),
    )


def _hipsparse_diag_type_enum(unit_diagonal):
    return _hipsparse_lookup(
        "hipsparseDiagType_t",
        ("HIPSPARSE_DIAG_TYPE_UNIT",)
        if unit_diagonal
        else ("HIPSPARSE_DIAG_TYPE_NON_UNIT",),
    )


def _hipsparse_call(attr_names, context):
    for attr_name in attr_names:
        fn = getattr(hipsparse, attr_name, None) if hipsparse is not None else None
        if fn is not None:
            return fn
    names = ", ".join(attr_names)
    raise RuntimeError(f"{context} is unavailable: missing {names}")


def _hipsparse_create_descriptor(attr_names, context, handle=None):
    ptr_type = type(handle) if handle is not None else None
    last_error = None
    for attr_name in attr_names:
        create_fn = getattr(hipsparse, attr_name, None) if hipsparse is not None else None
        if create_fn is None:
            continue
        try:
            raw = create_fn()
            if ptr_type is not None and isinstance(raw, ptr_type):
                return raw
            if hasattr(raw, "createRef"):
                return raw
            payload = _hip_check_result(raw, context)
            if payload is not None:
                return payload
        except TypeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc

        if ptr_type is None:
            continue
        descr = ptr_type()
        attempts = []
        if hasattr(descr, "createRef"):
            attempts.append((descr.createRef(),))
        attempts.append((descr,))
        for args in attempts:
            try:
                payload = _hip_check_result(create_fn(*args), context)
                return payload if payload is not None else descr
            except TypeError as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
    raise RuntimeError(f"{context} failed: {last_error}") from last_error


def _hipsparse_enum_storage(enum_value):
    try:
        raw_value = int(enum_value)
    except Exception:
        raw_value = getattr(enum_value, "value", enum_value)
    return ctypes.c_int(int(raw_value))


def _hipsparse_set_spmat_attribute(spmat, attr_name, enum_value):
    setter = _hipsparse_call(
        ("hipsparseSpMatSetAttribute",),
        "hipsparseSpMatSetAttribute",
    )
    attr = _hipsparse_spmat_attribute(attr_name)
    payload = _hipsparse_enum_storage(enum_value)
    attempts = (
        (spmat, attr, payload, ctypes.sizeof(payload)),
        (spmat, attr, ctypes.byref(payload), ctypes.sizeof(payload)),
        (spmat, attr, enum_value, ctypes.sizeof(payload)),
    )
    last_error = None
    for args in attempts:
        try:
            _hip_check_result(setter(*args), "hipsparseSpMatSetAttribute")
            return
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"hipsparseSpMatSetAttribute({attr_name}) failed: {last_error}"
    ) from last_error


def _hipsparse_create_spsv_descr(handle):
    return _hipsparse_create_descriptor(
        ("hipsparseSpSV_createDescr", "hipsparseCreateSpSVDescr"),
        "hipsparseSpSV_createDescr",
        handle=handle,
    )


def _hipsparse_destroy_spsv_descr(descr):
    if descr is None:
        return
    destroy_fn = _hipsparse_call(
        ("hipsparseSpSV_destroyDescr", "hipsparseDestroySpSVDescr"),
        "hipsparseSpSV_destroyDescr",
    )
    _hip_check_result(destroy_fn(descr), "hipsparseSpSV_destroyDescr")


def _hipsparse_spsv_skip_reason(
    value_dtype,
    index_dtype,
    indptr_dtype=None,
    *,
    op="non",
):
    indptr_dtype = index_dtype if indptr_dtype is None else indptr_dtype
    if not _is_rocm_runtime():
        return "hipSPARSE CSR SpSV reference requires a ROCm runtime"
    unavailable_reason = _hipsparse_unavailable_reason()
    if unavailable_reason is not None:
        return unavailable_reason
    required_symbols = (
        "hipsparseCreate",
        "hipsparseDestroy",
        "hipsparseCreateCsr",
        "hipsparseCreateDnVec",
        "hipsparseDestroyDnVec",
        "hipsparseDestroySpMat",
        "hipsparseSpMatSetAttribute",
        "hipsparseSpSV_bufferSize",
        "hipsparseSpSV_analysis",
        "hipsparseSpSV_solve",
    )
    for symbol in required_symbols:
        if not hasattr(hipsparse, symbol):
            return f"hipSPARSE CSR SpSV direct API is unavailable: missing {symbol}"
    if not any(
        hasattr(hipsparse, name)
        for name in ("hipsparseSpSV_createDescr", "hipsparseCreateSpSVDescr")
    ):
        return "hipSPARSE CSR SpSV direct API is unavailable: missing descriptor create API"
    if not any(
        hasattr(hipsparse, name)
        for name in ("hipsparseSpSV_destroyDescr", "hipsparseDestroySpSVDescr")
    ):
        return "hipSPARSE CSR SpSV direct API is unavailable: missing descriptor destroy API"
    if value_dtype not in SUPPORTED_SPSV_VALUE_DTYPES:
        return f"hipSPARSE CSR SpSV has no supported value dtype mapping for {value_dtype}"
    if index_dtype not in SUPPORTED_SPSV_INDEX_DTYPES:
        return f"hipSPARSE CSR SpSV has no supported index dtype mapping for {index_dtype}"
    if indptr_dtype not in SUPPORTED_SPSV_INDEX_DTYPES:
        return f"hipSPARSE CSR SpSV has no supported row offset dtype mapping for {indptr_dtype}"
    try:
        _validate_spsv_non_trans_combo(value_dtype, index_dtype, "CSR")
        _validate_spsv_trans_combo(value_dtype, index_dtype, "CSR")
        _ = _hipsparse_value_type(value_dtype)
        _ = _hipsparse_scalar(value_dtype, 1.0, 0.0)
        _ = _hipsparse_index_type(index_dtype, "hipSPARSE CSR SpSV column indices")
        _ = _hipsparse_index_type(indptr_dtype, "hipSPARSE CSR SpSV row offsets")
        _ = _hipsparse_spsv_op(op)
        _ = _hipsparse_spsv_alg()
        _ = _hipsparse_fill_mode_enum(True)
        _ = _hipsparse_fill_mode_enum(False)
        _ = _hipsparse_diag_type_enum(False)
        _ = _hipsparse_diag_type_enum(True)
        _ = _hipsparse_spmat_attribute("fill_mode")
        _ = _hipsparse_spmat_attribute("diag_type")
    except Exception as exc:
        return str(exc)
    return None


def _spsv_csr_sparse_ref_backend(value_dtype, index_dtype, indptr_dtype=None, op="non"):
    indptr_dtype = index_dtype if indptr_dtype is None else indptr_dtype
    if _is_rocm_runtime():
        reason = _hipsparse_spsv_skip_reason(
            value_dtype,
            index_dtype,
            indptr_dtype,
            op=op,
        )
        if reason is None:
            return "hipsparse", None
        return None, reason
    return None, "direct hipSPARSE CSR SpSV reference requires a ROCm runtime"


@dataclass
class FlagSparseSpSVDescr:
    """Host-side analysis handle for Triton SpSV.

    This is the Triton/Python equivalent of the CUDA-side SpSV descriptor:
    it stores the analyzed matrix metadata, the selected solve route, and the
    workspace layout needed by the current implementation.
    """

    format: str
    canonical_format: str
    shape: tuple
    lower: bool
    unit_diagonal: bool
    fill_mode: str
    diag_type: str
    matrix_type: str
    index_base: int
    transpose_mode: str
    value_dtype: torch.dtype
    compute_dtype: torch.dtype
    index_dtype: torch.dtype
    solve_kind: str
    route_name: str
    storage_view: str
    buffer_size: int
    workspace_layout: tuple
    data: torch.Tensor = field(repr=False)
    indices: torch.Tensor = field(repr=False)
    indptr: torch.Tensor = field(repr=False)
    solve_plan: dict = field(repr=False)


@dataclass
class FlagSparseSpSVWorkspace:
    """Caller-owned workspace object for Triton SpSV host APIs."""

    buffer_size: int
    layout: tuple
    device: torch.device
    buffers: dict = field(default_factory=dict, repr=False)
    prepared_solve_kind: str = ""
    prepared_signature: tuple | None = None


@dataclass
class FlagSparseSpSVHandle:
    """Host-side execution handle for Triton SpSV."""

    device: torch.device
    stream: object = None


@dataclass
class FlagSparseSpMatDescr:
    """Sparse matrix descriptor mirroring the CUDA SpMat inputs."""

    format: str
    shape: tuple
    values: torch.Tensor = field(repr=False)
    indices: torch.Tensor = field(repr=False)
    indptr_or_col: torch.Tensor = field(repr=False)
    lower: bool = True
    unit_diagonal: bool = False
    diag_type: str = "non_unit"
    fill_mode: str = "lower"
    matrix_type: str = "triangular"
    index_base: int = 0


@dataclass
class FlagSparseDnVecDescr:
    """Dense vector descriptor mirroring the CUDA DnVec inputs."""

    values: torch.Tensor = field(repr=False)


def flagsparse_create_spsv_handle(device=None, stream=None):
    if device is None:
        device = torch.device("cuda")
    return FlagSparseSpSVHandle(device=torch.device(device), stream=stream)


def flagsparse_create_dnvec(values):
    if not torch.is_tensor(values):
        raise TypeError("values must be a torch.Tensor")
    if values.ndim != 1:
        raise ValueError("DnVec values must be 1D")
    return FlagSparseDnVecDescr(values=values)


def flagsparse_create_spmat_csr(
    values,
    indices,
    indptr,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    matrix_type="triangular",
    index_base=0,
):
    return FlagSparseSpMatDescr(
        format="csr",
        shape=(int(shape[0]), int(shape[1])),
        values=values,
        indices=indices,
        indptr_or_col=indptr,
        lower=bool(lower),
        unit_diagonal=bool(unit_diagonal),
        diag_type="unit" if unit_diagonal else "non_unit",
        fill_mode="lower" if lower else "upper",
        matrix_type=str(matrix_type),
        index_base=int(index_base),
    )


def flagsparse_create_spmat_coo(
    values,
    row,
    col,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    matrix_type="triangular",
    index_base=0,
):
    return FlagSparseSpMatDescr(
        format="coo",
        shape=(int(shape[0]), int(shape[1])),
        values=values,
        indices=row,
        indptr_or_col=col,
        lower=bool(lower),
        unit_diagonal=bool(unit_diagonal),
        diag_type="unit" if unit_diagonal else "non_unit",
        fill_mode="lower" if lower else "upper",
        matrix_type=str(matrix_type),
        index_base=int(index_base),
    )


def _clear_spsv_csr_preprocess_cache():
    _SPSV_CSR_PREPROCESS_CACHE.clear()


def _as_strided_contiguous(tensor):
    if tensor is None:
        return None
    if tensor.layout != torch.strided:
        out = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        out.copy_(tensor)
        return out
    return tensor.contiguous()


def _complex_interleaved_view(tensor):
    tensor_strided = _as_strided_contiguous(tensor)
    return torch.view_as_real(tensor_strided).reshape(-1).contiguous()


def _attach_spsv_complex_plan_views(plan):
    kernel_data = plan.get("kernel_data")
    if kernel_data is None or not torch.is_complex(kernel_data):
        return plan
    plan["kernel_data_ri"] = _complex_interleaved_view(kernel_data)
    return plan


def _validate_spsv_non_trans_combo(data_dtype, index_dtype, fmt_name):
    """Validate NON_TRANS support matrix and keep error messages explicit."""
    if (data_dtype, index_dtype) in SPSV_NON_TRANS_SUPPORTED_COMBOS:
        return
    raise TypeError(
        f"{fmt_name} SpSV currently supports NON_TRANS combinations: "
        "(float32, int32/int64), (float64, int32/int64), "
        "(complex64, int32/int64), (complex128, int32/int64)"
    )


def _validate_spsv_trans_combo(data_dtype, index_dtype, fmt_name):
    if (data_dtype, index_dtype) in SPSV_TRANS_SUPPORTED_COMBOS:
        return
    raise TypeError(
        f"{fmt_name} SpSV currently supports TRANS/CONJ combinations: "
        "(float32, int32/int64), (float64, int32/int64), "
        "(complex64, int32/int64), (complex128, int32/int64)"
    )


def _normalize_spsv_transpose_mode(transpose):
    if isinstance(transpose, bool):
        return "T" if transpose else "N"
    token = str(transpose).strip().upper()
    if token in ("N", "NON", "NON_TRANS"):
        return "N"
    if token in ("T", "TRANS"):
        return "T"
    if token in ("C", "H", "CONJ", "CONJ_TRANS", "CONJUGATE_TRANSPOSE"):
        return "C"
    raise ValueError(
        "transpose must be bool or one of: "
        "N/NON/NON_TRANS, T/TRANS, C/H/CONJ/CONJ_TRANS/CONJUGATE_TRANSPOSE"
    )


def _prepare_spsv_csr_ref_hipsparse(
    data,
    indices,
    indptr,
    rhs,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    op="non",
    out=None,
):
    op_name = _normalize_sparse_reference_op(op)
    skip_reason = _hipsparse_spsv_skip_reason(
        data.dtype,
        indices.dtype,
        indptr.dtype,
        op=op_name,
    )
    if skip_reason is not None:
        raise RuntimeError(skip_reason)
    if not all(torch.is_tensor(t) for t in (data, indices, indptr, rhs)):
        raise TypeError("data, indices, indptr, rhs must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, indices, indptr, rhs)):
        raise ValueError("data, indices, indptr, rhs must all be CUDA tensors")
    if not all(t.device == data.device for t in (indices, indptr, rhs)):
        raise ValueError("data, indices, indptr, rhs must be on the same CUDA device")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1 or rhs.ndim != 1:
        raise ValueError("data, indices, indptr, rhs must all be 1D tensors")
    if indices.numel() != data.numel():
        raise ValueError("data and indices must have the same length")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows != n_cols:
        raise ValueError(f"hipSPARSE CSR SpSV reference expects a square matrix, got {shape}")
    if indptr.numel() != n_rows + 1:
        raise ValueError(f"indptr length must be n_rows+1={n_rows + 1}")
    rhs_size = n_rows if op_name == "non" else n_cols
    if rhs.numel() != rhs_size:
        raise ValueError(f"rhs length must be {rhs_size} for op={op_name}")

    data = data.contiguous()
    indices = indices.contiguous()
    indptr = indptr.contiguous()
    rhs = rhs.contiguous()
    value_type = _hipsparse_value_type(data.dtype)
    alpha = _hipsparse_scalar(data.dtype, 1.0, 0.0)
    row_index_type = _hipsparse_index_type(
        indptr.dtype, "hipSPARSE CSR SpSV row offsets"
    )
    col_index_type = _hipsparse_index_type(
        indices.dtype, "hipSPARSE CSR SpSV column indices"
    )
    op_enum = _hipsparse_spsv_op(op_name)
    alg = _hipsparse_spsv_alg()
    fill_mode = _hipsparse_fill_mode_enum(lower)
    diag_type = _hipsparse_diag_type_enum(unit_diagonal)

    solution = out
    if solution is None:
        solution = torch.empty_like(rhs)
    else:
        if not torch.is_tensor(solution):
            raise TypeError("out must be a torch.Tensor")
        if not solution.is_cuda or solution.device != data.device:
            raise ValueError("out must be a CUDA tensor on the same device as data")
        if solution.dtype != data.dtype or solution.shape != rhs.shape:
            raise ValueError("out must match the solution shape and dtype")
        if not solution.is_contiguous():
            raise ValueError("out must be contiguous")

    if rhs.numel() == 0:
        return {
            "backend": "hipsparse",
            "buffer_size": 0,
            "format": "csr",
            "solution": solution,
            "empty": True,
        }

    handle = None
    spmat = None
    rhs_desc = None
    sol_desc = None
    spsv_descr = None
    workspace = 0
    workspace_allocated = False
    try:
        handle = _hip_check_result(hipsparse.hipsparseCreate(), "hipsparseCreate")
        stream_warning = _try_set_hipsparse_current_stream(handle)
        ptr_type = type(handle)

        spmat = ptr_type()
        rhs_desc = ptr_type()
        sol_desc = ptr_type()
        spmat_ref = spmat.createRef()
        rhs_desc_ref = rhs_desc.createRef()
        sol_desc_ref = sol_desc.createRef()

        row_ptr = HipPointer.fromObj(indptr.data_ptr())
        col_ptr = HipPointer.fromObj(indices.data_ptr())
        values_ptr = HipPointer.fromObj(data.data_ptr())
        rhs_ptr = HipPointer.fromObj(rhs.data_ptr())
        sol_ptr = HipPointer.fromObj(solution.data_ptr())

        index_base = _hipsparse_lookup(
            "hipsparseIndexBase_t",
            ("HIPSPARSE_INDEX_BASE_ZERO",),
        )
        _hip_check_result(
            hipsparse.hipsparseCreateCsr(
                spmat_ref,
                n_rows,
                n_cols,
                int(data.numel()),
                row_ptr,
                col_ptr,
                values_ptr,
                row_index_type,
                col_index_type,
                index_base,
                value_type,
            ),
            "hipsparseCreateCsr",
        )
        _hipsparse_set_spmat_attribute(spmat, "fill_mode", fill_mode)
        _hipsparse_set_spmat_attribute(spmat, "diag_type", diag_type)
        _hip_check_result(
            hipsparse.hipsparseCreateDnVec(rhs_desc_ref, rhs_size, rhs_ptr, value_type),
            "hipsparseCreateDnVec(rhs)",
        )
        _hip_check_result(
            hipsparse.hipsparseCreateDnVec(sol_desc_ref, rhs_size, sol_ptr, value_type),
            "hipsparseCreateDnVec(solution)",
        )
        spsv_descr = _hipsparse_create_spsv_descr(handle)

        buffer_size_fn = _hipsparse_call(
            ("hipsparseSpSV_bufferSize",),
            "hipsparseSpSV_bufferSize",
        )
        analysis_fn = _hipsparse_call(
            ("hipsparseSpSV_analysis",),
            "hipsparseSpSV_analysis",
        )
        size_out = ctypes.c_size_t()
        _hip_check_result(
            buffer_size_fn(
                handle,
                op_enum,
                alpha,
                spmat,
                rhs_desc,
                sol_desc,
                value_type,
                alg,
                spsv_descr,
                size_out,
            ),
            "hipsparseSpSV_bufferSize",
        )
        buffer_size = int(size_out.value)
        if buffer_size > 0:
            workspace = _hip_check_result(hip.hipMalloc(buffer_size), "hipMalloc")
            workspace_allocated = True
        else:
            workspace = 0
        _hip_check_result(
            analysis_fn(
                handle,
                op_enum,
                alpha,
                spmat,
                rhs_desc,
                sol_desc,
                value_type,
                alg,
                spsv_descr,
                workspace,
            ),
            "hipsparseSpSV_analysis",
        )
        return {
            "backend": "hipsparse",
            "buffer_size": buffer_size,
            "format": "csr",
            "handle": handle,
            "spmat": spmat,
            "rhs_desc": rhs_desc,
            "sol_desc": sol_desc,
            "spsv_descr": spsv_descr,
            "workspace": workspace,
            "workspace_allocated": workspace_allocated,
            "op_enum": op_enum,
            "alpha": alpha,
            "value_type": value_type,
            "alg": alg,
            "solution": solution,
            "stream_binding_warning": stream_warning,
            "empty": False,
        }
    finally:
        if handle is None and spsv_descr is not None:
            try:
                _hipsparse_destroy_spsv_descr(spsv_descr)
            except Exception:
                pass
        if handle is None and sol_desc is not None:
            try:
                _hip_check_result(
                    hipsparse.hipsparseDestroyDnVec(sol_desc),
                    "hipsparseDestroyDnVec(solution)",
                )
            except Exception:
                pass
        if handle is None and rhs_desc is not None:
            try:
                _hip_check_result(
                    hipsparse.hipsparseDestroyDnVec(rhs_desc),
                    "hipsparseDestroyDnVec(rhs)",
                )
            except Exception:
                pass
        if handle is None and spmat is not None:
            try:
                _hip_check_result(
                    hipsparse.hipsparseDestroySpMat(spmat),
                    "hipsparseDestroySpMat",
                )
            except Exception:
                pass
        if handle is None and workspace_allocated:
            try:
                _hip_check_result(hip.hipFree(workspace), "hipFree")
            except Exception:
                pass


def _run_spsv_csr_ref_hipsparse_prepared(state):
    if state.get("empty"):
        return state["solution"]
    solve_fn = _hipsparse_call(
        ("hipsparseSpSV_solve",),
        "hipsparseSpSV_solve",
    )
    solve_args = (
        state["handle"],
        state["op_enum"],
        state["alpha"],
        state["spmat"],
        state["rhs_desc"],
        state["sol_desc"],
        state["value_type"],
        state["alg"],
        state["spsv_descr"],
    )
    # Some hip-python/hipSPARSE builds expose SpSV_solve without externalBuffer.
    try:
        result = solve_fn(*solve_args, state["workspace"])
    except TypeError as exc:
        if "positional" not in str(exc) and "argument" not in str(exc):
            raise
        result = solve_fn(*solve_args)
    _hip_check_result(
        result,
        "hipsparseSpSV_solve",
    )
    return state["solution"]


def _reanalyze_spsv_csr_ref_hipsparse_prepared(state):
    if state.get("empty"):
        return
    analysis_fn = _hipsparse_call(
        ("hipsparseSpSV_analysis",),
        "hipsparseSpSV_analysis",
    )
    _hip_check_result(
        analysis_fn(
            state["handle"],
            state["op_enum"],
            state["alpha"],
            state["spmat"],
            state["rhs_desc"],
            state["sol_desc"],
            state["value_type"],
            state["alg"],
            state["spsv_descr"],
            state["workspace"],
        ),
        "hipsparseSpSV_analysis",
    )


def _destroy_spsv_csr_ref_hipsparse_prepared(state):
    spsv_descr = state.get("spsv_descr")
    sol_desc = state.get("sol_desc")
    rhs_desc = state.get("rhs_desc")
    spmat = state.get("spmat")
    workspace_allocated = bool(state.get("workspace_allocated"))
    workspace = state.get("workspace", 0)
    handle = state.get("handle")
    if spsv_descr is not None:
        try:
            _hipsparse_destroy_spsv_descr(spsv_descr)
        except Exception:
            pass
    if sol_desc is not None:
        try:
            _hip_check_result(
                hipsparse.hipsparseDestroyDnVec(sol_desc),
                "hipsparseDestroyDnVec(solution)",
            )
        except Exception:
            pass
    if rhs_desc is not None:
        try:
            _hip_check_result(
                hipsparse.hipsparseDestroyDnVec(rhs_desc),
                "hipsparseDestroyDnVec(rhs)",
            )
        except Exception:
            pass
    if spmat is not None:
        try:
            _hip_check_result(
                hipsparse.hipsparseDestroySpMat(spmat),
                "hipsparseDestroySpMat",
            )
        except Exception:
            pass
    if workspace_allocated:
        try:
            _hip_check_result(hip.hipFree(workspace), "hipFree")
        except Exception:
            pass
    if handle is not None:
        try:
            _hip_check_result(hipsparse.hipsparseDestroy(handle), "hipsparseDestroy")
        except Exception:
            pass


def _spsv_csr_ref_hipsparse(
    data,
    indices,
    indptr,
    rhs,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    op="non",
    out=None,
    return_metadata=False,
):
    state = _prepare_spsv_csr_ref_hipsparse(
        data,
        indices,
        indptr,
        rhs,
        shape,
        lower=lower,
        unit_diagonal=unit_diagonal,
        op=op,
        out=out,
    )
    try:
        solution = _run_spsv_csr_ref_hipsparse_prepared(state)
        metadata = {
            "backend": "hipsparse",
            "buffer_size": int(state.get("buffer_size", 0)),
            "format": "csr",
        }
        if return_metadata:
            return solution, metadata
        return solution
    finally:
        _destroy_spsv_csr_ref_hipsparse_prepared(state)


def _benchmark_spsv_csr_sparse_ref(
    data,
    indices,
    indptr,
    rhs,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    op="non",
    warmup=0,
    iters=1,
    fresh_each_iter=False,
):
    backend, reason = _spsv_csr_sparse_ref_backend(
        data.dtype,
        indices.dtype,
        indptr.dtype,
        op=op,
    )
    result = {
        "backend": backend,
        "values": None,
        "ms": None,
        "reason": reason,
    }
    if backend is None:
        return result
    try:
        if fresh_each_iter:
            warmup = max(0, int(warmup))
            iters = max(1, int(iters))
            values = None
            state = _prepare_spsv_csr_ref_hipsparse(
                data,
                indices,
                indptr,
                rhs,
                shape,
                lower=lower,
                unit_diagonal=unit_diagonal,
                op=op,
            )
            try:
                for _ in range(warmup):
                    _reanalyze_spsv_csr_ref_hipsparse_prepared(state)
                    values = _run_spsv_csr_ref_hipsparse_prepared(state)
                    torch.cuda.synchronize()

                times = []
                for _ in range(iters):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _reanalyze_spsv_csr_ref_hipsparse_prepared(state)
                    values = _run_spsv_csr_ref_hipsparse_prepared(state)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - t0) * 1000.0)
            finally:
                _destroy_spsv_csr_ref_hipsparse_prepared(state)
            if times:
                ordered = sorted(times)
                median = ordered[len(ordered) // 2]
                lo = median * 0.9
                hi = median * 1.1
                kept = [t for t in ordered if lo <= t <= hi]
                ms = sum(kept) / len(kept) if kept else median
            else:
                ms = None
        else:
            values, ms = _benchmark_prepared_cuda_op(
                lambda: _prepare_spsv_csr_ref_hipsparse(
                    data,
                    indices,
                    indptr,
                    rhs,
                    shape,
                    lower=lower,
                    unit_diagonal=unit_diagonal,
                    op=op,
                ),
                _run_spsv_csr_ref_hipsparse_prepared,
                _destroy_spsv_csr_ref_hipsparse_prepared,
                warmup=warmup,
                iters=iters,
            )
        result["values"] = values
        result["ms"] = ms
        result["reason"] = None
    except Exception as exc:
        result["values"] = None
        result["ms"] = None
        result["reason"] = str(exc)
    return result


def _prepare_spsv_inputs(data, indices, indptr, b, shape):
    """Validate and normalize inputs for sparse solve A x = b with CSR A."""
    if not all(torch.is_tensor(t) for t in (data, indices, indptr, b)):
        raise TypeError("data, indices, indptr, b must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, indices, indptr, b)):
        raise ValueError("data, indices, indptr, b must all be CUDA tensors")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, indptr must be 1D")
    if b.ndim != 1:
        raise ValueError("b must be a 1D dense vector (DnVec)")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if indptr.numel() != n_rows + 1:
        raise ValueError(f"indptr length must be n_rows+1={n_rows + 1}")
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")
    if b.numel() != n_rows:
        raise ValueError(f"b length must equal n_rows={n_rows}")

    if data.dtype not in SUPPORTED_SPSV_VALUE_DTYPES:
        raise TypeError(
            "data dtype must be one of: float32, float64, complex64, complex128"
        )
    if indices.dtype not in SUPPORTED_SPSV_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")
    if indptr.dtype not in SUPPORTED_SPSV_INDEX_DTYPES:
        raise TypeError("indptr dtype must be torch.int32 or torch.int64")
    if b.dtype != data.dtype:
        raise TypeError("b dtype must match data dtype")

    indices64 = indices.to(torch.int64).contiguous()
    indptr64 = indptr.to(torch.int64).contiguous()
    if indices64.numel() > 0 and int(indices64.max().item()) > _INDEX_LIMIT_INT32:
        raise ValueError(
            f"int64 index value {int(indices64.max().item())} exceeds Triton int32 kernel range"
        )
    if indptr64.numel() > 0:
        if int(indptr64[0].item()) != 0:
            raise ValueError("indptr[0] must be 0")
        if int(indptr64[-1].item()) != data.numel():
            raise ValueError("indptr[-1] must equal nnz")
        if bool(torch.any(indptr64[1:] < indptr64[:-1]).item()):
            raise ValueError("indptr must be non-decreasing")
    if indices64.numel() > 0:
        if bool(torch.any(indices64 < 0).item()):
            raise IndexError("indices must be non-negative")
        max_idx = int(indices64.max().item())
        if max_idx >= n_cols:
            raise IndexError(f"indices out of range for n_cols={n_cols}")

    return (
        data.contiguous(),
        indices.dtype,
        indices64,
        indptr64,
        b.contiguous(),
        n_rows,
        n_cols,
    )

def _spsv_diag_eps_for_dtype(value_dtype):
    return 1e-12 if value_dtype in (torch.float64, torch.complex128) else 1e-6


def _tensor_cache_token(tensor):
    try:
        storage_ptr = int(tensor.untyped_storage().data_ptr())
    except Exception:
        storage_ptr = 0
    return (
        str(tensor.device),
        str(tensor.dtype),
        tuple(int(v) for v in tensor.shape),
        int(tensor.numel()),
        storage_ptr,
        int(getattr(tensor, "_version", 0)),
    )


def _spsv_cache_get(cache, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _spsv_cache_put(cache, key, value, max_entries):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_entries:
        cache.popitem(last=False)


def _normalize_spsv_format(fmt):
    token = str(fmt).strip().lower()
    if token not in ("csr", "coo"):
        raise ValueError("format must be 'csr' or 'coo'")
    return token


def _normalize_spsv_storage_view(storage_view):
    if storage_view is None:
        return "csr_as_csc"
    token = str(storage_view).strip().lower()
    aliases = {
        "csr_as_csc": "csr_as_csc",
        "csc_view": "csr_as_csc",
        "reuse_csr_storage": "csr_as_csc",
    }
    if token not in aliases:
        raise ValueError(
            "storage_view must be one of: csr_as_csc, csc_view, reuse_csr_storage"
        )
    return aliases[token]


def _resolve_spsv_stream(handle, stream, device):
    resolved = stream
    if handle is not None:
        if not isinstance(handle, FlagSparseSpSVHandle):
            raise TypeError("handle must be a FlagSparseSpSVHandle or None")
        if torch.device(handle.device) != torch.device(device):
            raise ValueError("handle device must match the solve device")
        if resolved is None:
            resolved = handle.stream
    return resolved


def _coerce_spsv_alpha(alpha, dtype, device):
    if torch.is_tensor(alpha):
        alpha_tensor = alpha.to(device=device, dtype=dtype).reshape(-1)
        if alpha_tensor.numel() != 1:
            raise ValueError("alpha must be a scalar tensor")
        return alpha_tensor.reshape(())
    return torch.tensor(alpha, device=device, dtype=dtype)


def _spsv_alpha_is_identity(alpha):
    if torch.is_tensor(alpha):
        alpha_flat = alpha.reshape(-1)
        if alpha_flat.numel() != 1:
            raise ValueError("alpha must be a scalar tensor")
        return bool((alpha_flat == 1).all().item())
    return alpha == 1


def _workspace_entry(name, numel, dtype):
    return {
        "name": str(name),
        "numel": int(numel),
        "dtype": dtype,
        "bytes": int(numel) * int(torch.empty((), dtype=dtype).element_size()),
    }


def _workspace_size_bytes(layout):
    return int(sum(int(entry["bytes"]) for entry in layout))


def _spsv_effective_compute_dtype(value_dtype, trans_mode, compute_dtype=None):
    if compute_dtype is not None:
        if compute_dtype not in SUPPORTED_SPSV_VALUE_DTYPES:
            raise TypeError(
                "compute_dtype must be one of: float32, float64, complex64, complex128"
            )
        return compute_dtype
    if (
        value_dtype == torch.complex64
        and trans_mode in ("T", "C")
        and SPSV_PROMOTE_TRANSPOSE_COMPLEX64_TO_COMPLEX128
    ):
        return torch.complex128
    if value_dtype == torch.float32 and SPSV_PROMOTE_FP32_TO_FP64:
        return torch.float64
    if (
        value_dtype == torch.float32
        and trans_mode in ("T", "C")
        and SPSV_PROMOTE_TRANSPOSE_FP32_TO_FP64
    ):
        return torch.float64
    return value_dtype


def _build_spsv_workspace_layout(n_rows, solve_kind, value_dtype=None):
    n_rows = int(n_rows)
    if solve_kind == "csr_cw":
        return (
            _workspace_entry("ready", n_rows, torch.int32),
            _workspace_entry("row_counter", 1, torch.int32),
        )
    if solve_kind == "csr_roc":
        return (
            _workspace_entry("ready", n_rows, torch.int32),
        )
    if solve_kind == "csr_smblk":
        return (
            _workspace_entry("ready", n_rows, torch.int32),
        )
    if solve_kind == "csr_cw_levelschd":
        return (
            _workspace_entry("ready", n_rows, torch.int32),
        )
    if solve_kind == "csr_nnz_balance":
        if value_dtype is None:
            raise ValueError("value_dtype is required for csr_nnz_balance workspace sizing")
        return (
            _workspace_entry("tmp_sum", n_rows, value_dtype),
            _workspace_entry("ready", n_rows, torch.int32),
            _workspace_entry("indegree", n_rows, torch.int32),
        )
    if solve_kind == "transpose_cw":
        if value_dtype is None:
            raise ValueError("value_dtype is required for transpose_cw workspace sizing")
        return (
            _workspace_entry("residual", n_rows, value_dtype),
            _workspace_entry("indegree", n_rows, torch.int32),
            _workspace_entry("row_counter", 1, torch.int32),
        )
    raise ValueError(f"Unsupported SpSV solve kind for workspace sizing: {solve_kind}")


def _clone_spsv_plan(plan):
    cloned = dict(plan)
    matrix_stats = plan.get("matrix_stats")
    if matrix_stats is not None:
        cloned["matrix_stats"] = dict(matrix_stats)
    return cloned


def _alloc_spsv_workspace_buffers(layout, device):
    buffers = {}
    for entry in layout:
        buffers[entry["name"]] = torch.empty(
            int(entry["numel"]), dtype=entry["dtype"], device=device
        )
    return buffers


def _resolve_spsv_workspace(workspace, layout, device):
    if workspace is None:
        return _alloc_spsv_workspace_buffers(layout, device)
    if not isinstance(workspace, FlagSparseSpSVWorkspace):
        raise TypeError("workspace must be a FlagSparseSpSVWorkspace or None")
    if torch.device(workspace.device) != torch.device(device):
        raise ValueError("workspace device must match the solve device")
    if int(workspace.buffer_size) < _workspace_size_bytes(layout):
        raise ValueError("workspace buffer is smaller than the required SpSV size")
    required = {entry["name"]: entry for entry in layout}
    for name, entry in required.items():
        buf = workspace.buffers.get(name)
        if buf is None:
            workspace.buffers[name] = torch.empty(
                int(entry["numel"]), dtype=entry["dtype"], device=device
            )
            continue
        if buf.device != torch.device(device):
            raise ValueError(f"workspace buffer {name!r} is on the wrong device")
        if buf.dtype != entry["dtype"] or int(buf.numel()) < int(entry["numel"]):
            workspace.buffers[name] = torch.empty(
                int(entry["numel"]), dtype=entry["dtype"], device=device
            )
    return workspace.buffers


def _transpose_cw_preprocess_signature(
    solve_plan, n_rows, unit_diagonal, block_nnz_use, max_segments_use
):
    kernel_indices32 = solve_plan["kernel_indices32"]
    kernel_indptr64 = solve_plan["kernel_indptr64"]
    return (
        "transpose_cw",
        int(n_rows),
        bool(solve_plan["lower_eff"]),
        bool(unit_diagonal),
        int(block_nnz_use),
        int(max_segments_use),
        _tensor_cache_token(kernel_indices32),
        _tensor_cache_token(kernel_indptr64),
    )


def flagsparse_spsv_buffer_size(
    shape,
    value_dtype,
    *,
    format="csr",
    transpose=False,
    solve_kind=None,
    compute_dtype=None,
    alpha=None,
    handle=None,
    vecX=None,
    vecY=None,
    storage_view="csr_as_csc",
):
    """Return the caller-managed workspace size for the current Triton SpSV route.

    This is the Triton host-side equivalent of the CUDA bufferSize query.
    The returned byte count matches the scratch buffers used by the current
    Triton implementation, rather than the raw CUDA ABI layout.
    """

    _normalize_spsv_format(format)
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows != n_cols:
        raise ValueError(f"SpSV expects a square matrix, got shape={shape}")
    if value_dtype not in SUPPORTED_SPSV_VALUE_DTYPES:
        raise TypeError(
            "value_dtype must be one of: float32, float64, complex64, complex128"
        )
    trans_mode = _normalize_spsv_transpose_mode(transpose)
    storage_view = _normalize_spsv_storage_view(storage_view)
    compute_dtype = _spsv_effective_compute_dtype(
        value_dtype, trans_mode, compute_dtype=compute_dtype
    )
    route = _normalize_requested_spsv_route(solve_kind, trans_mode)
    if route is None:
        route = "transpose_cw" if trans_mode in ("T", "C") else "csr_cw"
    if trans_mode in ("T", "C") and storage_view != "csr_as_csc":
        raise ValueError("TRANS/CONJ SpSV only supports storage_view='csr_as_csc'")
    layout = _build_spsv_workspace_layout(n_rows, route, value_dtype=compute_dtype)
    return _workspace_size_bytes(layout)


def flagsparse_spsv_create_workspace(descr, device=None):
    """Allocate a caller-owned SpSV workspace object from an analysis descriptor."""

    if not isinstance(descr, FlagSparseSpSVDescr):
        raise TypeError("descr must be a FlagSparseSpSVDescr")
    if device is None:
        device = descr.data.device
    device = torch.device(device)
    buffers = _alloc_spsv_workspace_buffers(descr.workspace_layout, device)
    return FlagSparseSpSVWorkspace(
        buffer_size=int(descr.buffer_size),
        layout=tuple(descr.workspace_layout),
        device=device,
        buffers=buffers,
    )


def _csr_preprocess_cache_key(
    data, indices, indptr, shape, lower, trans_mode, unit_diagonal, requested_route=None, storage_view="csr_as_csc"
):
    return (
        "csr_preprocess",
        trans_mode,
        bool(lower),
        bool(unit_diagonal),
        str(requested_route),
        str(storage_view),
        int(shape[0]),
        int(shape[1]),
        _tensor_cache_token(data),
        _tensor_cache_token(indices),
        _tensor_cache_token(indptr),
    )


def _normalize_requested_spsv_route(solve_kind, trans_mode):
    if solve_kind is None:
        return None
    token = str(solve_kind).strip().lower()
    aliases = {
        "alg1": "csr_cw",
        "csr_cw": "csr_cw",
        "csr_roc": "csr_roc",
        "roc": "csr_roc",
        "alg3": "csr_roc",
        "csr_smblk": "csr_smblk",
        "smblk": "csr_smblk",
        "alg4": "csr_smblk",
        "csr_levelschd": "csr_cw_levelschd",
        "csr_cw_levelschd": "csr_cw_levelschd",
        "levelschd": "csr_cw_levelschd",
        "level_sched": "csr_cw_levelschd",
        "alg2": "csr_cw_levelschd",
        "csr_nnz_balance": "csr_nnz_balance",
        "nnz_balance": "csr_nnz_balance",
        "alg8": "csr_nnz_balance",
        "cw": "csr_cw" if trans_mode == "N" else "transpose_cw",
        "transpose_cw": "transpose_cw",
        "csc_cw": "transpose_cw",
    }
    route = aliases.get(token)
    if route is None:
        raise ValueError(
            "solve_kind must be one of: csr_cw, csr_roc, csr_smblk, csr_cw_levelschd, csr_nnz_balance, transpose_cw"
        )
    if trans_mode in ("T", "C") and route != "transpose_cw":
        raise ValueError("TRANS/CONJ SpSV only supports solve_kind='transpose_cw'")
    if trans_mode == "N" and route == "transpose_cw":
        raise ValueError("NON_TRANS SpSV cannot use solve_kind='transpose_cw'")
    return route


def _csr_transpose(data, indices, indptr, n_rows_or_shape, n_cols=None, conjugate=False):
    if n_cols is None:
        n_rows, n_cols = int(n_rows_or_shape[0]), int(n_rows_or_shape[1])
    else:
        n_rows, n_cols = int(n_rows_or_shape), int(n_cols)
    device = data.device
    if data.numel() == 0:
        return (
            data.conj() if conjugate and torch.is_complex(data) else data,
            torch.empty(0, dtype=torch.int64, device=device),
            torch.zeros(n_cols + 1, dtype=torch.int64, device=device),
        )

    indptr64 = indptr.to(torch.int64)
    row = torch.repeat_interleave(
        torch.arange(n_rows, device=device, dtype=torch.int64),
        indptr64[1:] - indptr64[:-1],
    )
    col = indices.to(torch.int64)
    row_t = col
    col_t = row
    key = row_t * max(1, n_rows) + col_t
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)

    row_t = row_t[order]
    col_t = col_t[order]
    data_eff = data.conj() if conjugate and torch.is_complex(data) else data
    data_t = data_eff[order]
    nnz_per_row = torch.bincount(row_t, minlength=n_cols)
    indptr_t = torch.zeros(n_cols + 1, dtype=torch.int64, device=device)
    indptr_t[1:] = torch.cumsum(nnz_per_row, dim=0)
    return data_t, col_t.to(torch.int64), indptr_t


from .spsv_csr_n_cw import (
    _spsv_csr_cw_kernel as _spsv_csr_n_cw_kernel,
    _spsv_csr_cw_kernel_complex as _spsv_csr_n_cw_kernel_complex,
)
from .spsv_csr_u_cw import (
    _spsv_csr_cw_kernel as _spsv_csr_u_cw_kernel,
    _spsv_csr_cw_kernel_complex as _spsv_csr_u_cw_kernel_complex,
)
from .spsv_csr_n_cw_levelschd import (
    _spsv_csr_cw_levelschd_kernel,
    _spsv_csr_cw_levelschd_kernel_complex,
    _spsv_levelschd_analysis_kernel,
)
from .spsv_csr_n_nnz_balance import (
    _spsv_csr_nnz_balance_kernel,
    _spsv_csr_nnz_balance_kernel_complex,
    _spsv_nnz_balance_preprocess_kernel,
)
from .spsv_csr_n_roc import (
    _spsv_csr_roc_kernel,
    _spsv_csr_roc_kernel_complex,
)
from .spsv_csr_n_smblk import (
    _spsv_csr_smblk_kernel,
    _spsv_csr_smblk_kernel_complex,
)
from .spsv_csc_n_cw import (
    _spsv_csc_preprocess_kernel as _spsv_csc_n_preprocess_kernel,
    _spsv_csr_transpose_cw_kernel as _spsv_csc_n_cw_kernel,
    _spsv_csr_transpose_cw_kernel_complex as _spsv_csc_n_cw_kernel_complex,
)
from .spsv_csc_u_cw import (
    _spsv_csc_preprocess_kernel as _spsv_csc_u_preprocess_kernel,
    _spsv_csr_transpose_cw_kernel as _spsv_csc_u_cw_kernel,
    _spsv_csr_transpose_cw_kernel_complex as _spsv_csc_u_cw_kernel_complex,
)

def _sort_csr_rows(data, indices64, indptr64, n_rows, n_cols, lower=True):
    if data.numel() == 0:
        return data, indices64, indptr64
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, device=data.device, dtype=torch.int64),
        indptr64[1:] - indptr64[:-1],
    )
    key = row_ids * max(1, n_cols)
    if lower:
        key = key + indices64
    else:
        key = key + (n_cols - 1 - indices64)
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)
    return data[order], indices64[order], indptr64


def _spsv_csr_row_length_summary(indptr64, n_rows):
    if indptr64.numel() <= 1 or int(n_rows) <= 0:
        return torch.empty(0, dtype=torch.int64, device=indptr64.device), 0.0, 0
    row_lengths = (indptr64[1:] - indptr64[:-1]).to(torch.int64)
    avg_nnz_per_row = float(row_lengths.to(torch.float32).mean().item())
    max_nnz_per_row = int(row_lengths.max().item()) if row_lengths.numel() > 0 else 0
    return row_lengths, avg_nnz_per_row, max_nnz_per_row


def _csr_rows_are_sorted(indices64, indptr64, n_rows, lower=True):
    if indices64.numel() <= 1 or int(n_rows) <= 0:
        return True
    row_lengths, _, max_nnz_per_row = _spsv_csr_row_length_summary(indptr64, n_rows)
    if row_lengths.numel() == 0 or max_nnz_per_row <= 1:
        return True
    same_row = torch.ones(
        indices64.numel() - 1,
        dtype=torch.bool,
        device=indices64.device,
    )
    row_ends = indptr64[1:-1].to(torch.int64) - 1
    row_ends = row_ends[(row_ends >= 0) & (row_ends < same_row.numel())]
    if row_ends.numel() > 0:
        same_row[row_ends] = False
    if not bool(torch.any(same_row).item()):
        return True
    if lower:
        ordered = indices64[1:] >= indices64[:-1]
    else:
        ordered = indices64[1:] <= indices64[:-1]
    return bool(torch.all(ordered | (~same_row)).item())


def _maybe_sort_csr_rows(data, indices64, indptr64, n_rows, n_cols, lower=True):
    if _csr_rows_are_sorted(indices64, indptr64, n_rows, lower=lower):
        return data, indices64, indptr64
    return _sort_csr_rows(data, indices64, indptr64, n_rows, n_cols, lower=lower)


def _cw_rhs_bucket(n_rhs):
    if n_rhs <= 1:
        return 1
    if n_rhs <= 2:
        return 2
    if n_rhs <= 4:
        return 4
    if n_rhs <= 8:
        return 8
    if n_rhs <= 16:
        return 16
    return 32


def _snap_cw_worker_count(target, n_rows):
    if n_rows <= 0:
        return 1
    target = max(1, min(int(target), int(n_rows)))
    snapped = 1
    tier = 1
    while tier < target and tier < 4096:
        tier *= 2
        if tier <= target:
            snapped = tier
    return int(max(1, min(snapped, int(n_rows))))


def _cw_worker_count(n_rows, max_frontier, avg_nnz_per_row, n_rhs):
    if n_rows <= 0:
        return 1
    rhs_bucket = _cw_rhs_bucket(n_rhs)
    if rhs_bucket == 1:
        target = min(n_rows, 32)
    else:
        target = max(32, min(n_rows, 512))
    if max_frontier > 0:
        target = min(target, max(4, min(n_rows, max_frontier * 2)))
    if avg_nnz_per_row > 8192:
        target = max(4, target // 8)
    elif avg_nnz_per_row > 4096:
        target = max(4, target // 4)
    elif avg_nnz_per_row > 2048:
        target = max(4, target // 2)
    elif avg_nnz_per_row > 1024:
        target = max(8, (target * 2) // 3)
    elif avg_nnz_per_row > 512:
        target = max(8, (target * 3) // 5)
    if rhs_bucket >= 16:
        target = max(4, target // 4)
    elif rhs_bucket >= 8:
        target = max(4, target // 2)
    elif rhs_bucket >= 4:
        target = max(4, (target * 3) // 4)
    return _snap_cw_worker_count(target, n_rows)

def _resolve_cw_worker_count(n_rows, matrix_stats, n_rhs, cached_worker_count=None):
    rhs_bucket = _cw_rhs_bucket(n_rhs)
    max_frontier = int(matrix_stats.get("max_frontier", n_rows))
    avg_frontier = float(matrix_stats.get("avg_frontier", float(max_frontier)))
    frontier_ratio = float(matrix_stats.get("frontier_ratio", 1.0 if n_rows > 0 else 0.0))
    num_levels = int(matrix_stats.get("num_levels", 0))
    avg_nnz_per_row = float(matrix_stats.get("avg_nnz_per_row", 0.0))
    if cached_worker_count is not None and rhs_bucket == 1:
        target = int(max(1, min(int(cached_worker_count), int(max(n_rows, 1)))))
    else:
        target = _cw_worker_count(
            n_rows,
            max_frontier,
            avg_nnz_per_row,
            rhs_bucket,
        )
    if frontier_ratio < 0.01 or avg_frontier < 4.0:
        target = min(target, max(1, min(n_rows, 4)))
    elif frontier_ratio < 0.02 or avg_frontier < 8.0:
        target = min(target, max(1, min(n_rows, 8)))
    elif frontier_ratio < 0.05 or avg_frontier < 16.0:
        target = min(target, max(2, min(n_rows, 16)))
    if num_levels > max(1024, n_rows // 2):
        target = max(1, target // 2)
    if avg_nnz_per_row > 2048:
        target = max(1, target // 2)
    return _snap_cw_worker_count(
        target,
        n_rows,
    )


def _build_spsv_cw_matrix_stats(
    indptr64,
    n_rows,
    *,
    avg_nnz_per_row=None,
    max_nnz_per_row=None,
):
    if avg_nnz_per_row is None or max_nnz_per_row is None:
        _, avg_nnz_per_row, max_nnz_per_row = _spsv_csr_row_length_summary(indptr64, n_rows)
    return {
        "num_levels": 0,
        "max_frontier": int(n_rows),
        "avg_frontier": float(n_rows),
        "frontier_ratio": 1.0 if n_rows > 0 else 0.0,
        "avg_nnz_per_row": avg_nnz_per_row,
        "max_nnz_per_row": max_nnz_per_row,
        "n_rows": int(n_rows),
    }


def _build_spsv_nnz_balance_launch_order(indptr64, n_rows, *, lower):
    n_rows = int(n_rows)
    device = indptr64.device
    total_nnz = int(indptr64[-1].item()) if indptr64.numel() > 0 else 0
    if total_nnz <= 0 or n_rows <= 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    if lower:
        return torch.arange(total_nnz, dtype=torch.int32, device=device)

    indptr_cpu = indptr64.to("cpu", non_blocking=False).tolist()
    launch_order = []
    for row in range(n_rows - 1, -1, -1):
        start = int(indptr_cpu[row])
        end = int(indptr_cpu[row + 1])
        launch_order.extend(range(start, end))
    return torch.tensor(launch_order, dtype=torch.int32, device=device)


def _supports_spsv_advanced_nontrans_routes(trans_mode, lower, unit_diagonal, value_dtype):
    return (
        trans_mode == "N"
        and (not bool(unit_diagonal))
        and value_dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128)
    )


def _choose_spsv_nontrans_auto_route(
    n_rows,
    matrix_stats,
    *,
    lower,
    unit_diagonal,
    value_dtype,
):
    """Heuristic route picker for NON_TRANS triangular solves.

    This is an analysis-time auto selector, not a runtime autotuner.  The goal
    is to keep default routing predictable while still steering obviously
    serialized systems and wide-frontier systems onto more suitable kernels.
    """
    if _is_rocm_runtime() and not SPSV_ROCM_ENABLE_ADVANCED_AUTO:
        return "csr_cw"
    if bool(unit_diagonal):
        return "csr_cw"
    n_rows = int(n_rows)
    if n_rows <= 0:
        return "csr_cw"

    # Upper NON sweeps consistently favor ALG4 (csr_smblk), so keep that as the
    # unconditional AUTO route on the upper side.
    if not lower:
        if value_dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128):
            return "csr_smblk"
        return "csr_cw"

    # Keep lower NON AUTO predictable as well: default to ALG4 (csr_smblk) for
    # supported dtypes instead of sending many SuiteSparse-derived cases through
    # heuristic detours such as level scheduling.
    if value_dtype in (torch.float32, torch.float64):
        return "csr_smblk"

    if value_dtype in (torch.complex64, torch.complex128):
        return "csr_smblk"
    return "csr_cw"


def _build_spsv_level_schedule_metadata_lower_gpu(
    indices64, indptr64, n_rows, *, unit_diagonal, minimal=False
):
    n_rows = int(n_rows)
    device = indices64.device
    base_stats = _build_spsv_cw_matrix_stats(indptr64, n_rows)
    empty_meta = {
        "row_map32": torch.empty(0, dtype=torch.int32, device=device),
        "level_ptr32": torch.zeros(1, dtype=torch.int32, device=device),
        "indegree_init32": torch.empty(0, dtype=torch.int32, device=device),
        "csr_row_idx32": torch.empty(0, dtype=torch.int32, device=device),
        "matrix_stats": {
            **base_stats,
            "num_levels": 0,
            "max_frontier": 0,
            "avg_frontier": 0.0,
            "frontier_ratio": 0.0,
        },
    }
    if n_rows == 0:
        return empty_meta

    indices32 = indices64.to(torch.int32).contiguous()
    levels32 = torch.zeros(n_rows, dtype=torch.int32, device=device)
    ready32 = torch.zeros(n_rows, dtype=torch.int32, device=device)
    indegree32 = torch.empty(n_rows, dtype=torch.int32, device=device)
    _spsv_levelschd_analysis_kernel[(triton.cdiv(n_rows, 8),)](
        indices32,
        indptr64,
        levels32,
        ready32,
        indegree32,
        n_rows,
        BLOCK_ROWS=8,
        UNIT_DIAGONAL=bool(unit_diagonal),
        num_warps=1,
    )

    # Stable GPU sort reproduces the row_map stage after roc-style level analysis.

    try:
        row_map64 = torch.argsort(levels32.to(torch.int64), stable=True)
    except TypeError:
        row_map64 = torch.argsort(levels32.to(torch.int64))
    row_map32 = row_map64.to(torch.int32).contiguous()
    sorted_levels32 = levels32.index_select(0, row_map64)
    if sorted_levels32.numel() > 0:
        _, frontier_counts64 = torch.unique_consecutive(sorted_levels32, return_counts=True)
        level_ptr32 = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(frontier_counts64.to(torch.int32), dim=0),
            ]
        )
        num_levels = int(frontier_counts64.numel())
        max_frontier = int(frontier_counts64.max().item())
        avg_frontier = float(frontier_counts64.to(torch.float32).mean().item())
    else:
        level_ptr32 = torch.zeros(1, dtype=torch.int32, device=device)
        num_levels = 0
        max_frontier = 0
        avg_frontier = 0.0

    row_lengths64 = indptr64[1:] - indptr64[:-1]
    csr_row_idx32 = torch.repeat_interleave(
        torch.arange(n_rows, device=device, dtype=torch.int32),
        row_lengths64.to(torch.int64),
    ).contiguous()
    matrix_stats = {
        **base_stats,
        "num_levels": int(num_levels),
        "max_frontier": int(max_frontier),
        "avg_frontier": float(avg_frontier),
        "frontier_ratio": (float(max_frontier) / float(n_rows)) if n_rows > 0 else 0.0,
    }
    if minimal:
        return {
            "row_map32": row_map32,
            "level_ptr32": torch.zeros(1, dtype=torch.int32, device=device),
            "indegree_init32": torch.empty(0, dtype=torch.int32, device=device),
            "csr_row_idx32": torch.empty(0, dtype=torch.int32, device=device),
            "matrix_stats": matrix_stats,
        }
    return {
        "row_map32": row_map32,
        "level_ptr32": level_ptr32,
        "indegree_init32": indegree32,
        "csr_row_idx32": csr_row_idx32,
        "matrix_stats": matrix_stats,
    }


def _build_spsv_nnz_balance_metadata(indices64, indptr64, n_rows, *, lower, unit_diagonal):
    n_rows = int(n_rows)
    device = indices64.device
    base_stats = _build_spsv_cw_matrix_stats(indptr64, n_rows)
    empty_meta = {
        "indegree_init32": torch.empty(0, dtype=torch.int32, device=device),
        "csr_row_idx32": torch.empty(0, dtype=torch.int32, device=device),
        "launch_order32": torch.empty(0, dtype=torch.int32, device=device),
        "matrix_stats": base_stats,
    }
    if n_rows == 0:
        return empty_meta
    if indices64.is_cuda:
        if not lower:
            # Upper-triangular preprocessing reuses the generic host path for now.
            indices_cpu = indices64.to("cpu", non_blocking=False).tolist()
            indptr_cpu = indptr64.to("cpu", non_blocking=False).tolist()
            indegree_init = [0] * n_rows
            row_idx = [0] * int(indices64.numel())
            for row in range(n_rows):
                start = int(indptr_cpu[row])
                end = int(indptr_cpu[row + 1])
                degree = 0
                for ptr in range(start, end):
                    col = int(indices_cpu[ptr])
                    if col > row:
                        row_idx[ptr] = row
                        degree += 1
                        continue
                    if (not unit_diagonal) and col == row:
                        row_idx[ptr] = row
                        degree += 1
                    break
                indegree_init[row] = degree
            return {
                "indegree_init32": torch.tensor(indegree_init, dtype=torch.int32, device=device),
                "csr_row_idx32": torch.tensor(row_idx, dtype=torch.int32, device=device),
                "launch_order32": _build_spsv_nnz_balance_launch_order(
                    indptr64, n_rows, lower=lower
                ),
                "matrix_stats": base_stats,
            }
        indices32 = indices64.to(torch.int32).contiguous()
        indegree32 = torch.zeros(n_rows, dtype=torch.int32, device=device)
        row_idx32 = torch.zeros(indices32.numel(), dtype=torch.int32, device=device)
        _spsv_nnz_balance_preprocess_kernel[(n_rows,)](
            indices32,
            indptr64,
            indegree32,
            row_idx32,
            n_rows,
            WARP_SIZE=32,
            UNIT_DIAGONAL=bool(unit_diagonal),
            num_warps=1,
        )
        return {
            "indegree_init32": indegree32,
            "csr_row_idx32": row_idx32,
            "launch_order32": _build_spsv_nnz_balance_launch_order(
                indptr64, n_rows, lower=lower
            ),
            "matrix_stats": base_stats,
        }

    indptr_cpu = indptr64.to("cpu", non_blocking=False).tolist()
    indices_cpu = indices64.to("cpu", non_blocking=False).tolist()
    indegree_init = [0] * n_rows
    row_idx = [0] * int(indices64.numel())
    for row in range(n_rows):
        start = int(indptr_cpu[row])
        end = int(indptr_cpu[row + 1])
        degree = 0
        for ptr in range(start, end):
            col = int(indices_cpu[ptr])
            if lower:
                if col < row:
                    row_idx[ptr] = row
                    degree += 1
                    continue
                if (not unit_diagonal) and col == row:
                    row_idx[ptr] = row
                    degree += 1
                break
            else:
                if col > row:
                    row_idx[ptr] = row
                    degree += 1
                    continue
                if (not unit_diagonal) and col == row:
                    row_idx[ptr] = row
                    degree += 1
                break
        indegree_init[row] = degree
    return {
        "indegree_init32": torch.tensor(indegree_init, dtype=torch.int32, device=device),
        "csr_row_idx32": torch.tensor(row_idx, dtype=torch.int32, device=device),
        "launch_order32": _build_spsv_nnz_balance_launch_order(
            indptr64, n_rows, lower=lower
        ),
        "matrix_stats": base_stats,
    }


def _build_spsv_level_schedule_metadata(
    indices64, indptr64, n_rows, *, lower, unit_diagonal, minimal=False
):
    n_rows = int(n_rows)
    device = indices64.device
    base_stats = _build_spsv_cw_matrix_stats(indptr64, n_rows)
    empty_meta = {
        "row_map32": torch.empty(0, dtype=torch.int32, device=device),
        "level_ptr32": torch.zeros(1, dtype=torch.int32, device=device),
        "indegree_init32": torch.empty(0, dtype=torch.int32, device=device),
        "csr_row_idx32": torch.empty(0, dtype=torch.int32, device=device),
        "matrix_stats": {
            **base_stats,
            "num_levels": 0,
            "max_frontier": 0,
            "avg_frontier": 0.0,
            "frontier_ratio": 0.0,
        },
    }
    if n_rows == 0:
        return empty_meta

    if lower and indices64.is_cuda and not _is_rocm_runtime():
        return _build_spsv_level_schedule_metadata_lower_gpu(
            indices64,
            indptr64,
            n_rows,
            unit_diagonal=unit_diagonal,
            minimal=minimal,
        )

    indptr_cpu = indptr64.to("cpu", non_blocking=False).tolist()
    indices_cpu = indices64.to("cpu", non_blocking=False).tolist()
    levels = [0] * n_rows
    indegree_init = [0] * n_rows
    level_buckets = {}

    row_iter = range(n_rows) if lower else range(n_rows - 1, -1, -1)
    for row in row_iter:
        start = int(indptr_cpu[row])
        end = int(indptr_cpu[row + 1])
        deps = []
        degree = 0
        for ptr in range(start, end):
            col = int(indices_cpu[ptr])
            if lower:
                if unit_diagonal:
                    if col < row:
                        deps.append(col)
                        degree += 1
                    else:
                        break
                else:
                    if col < row:
                        deps.append(col)
                        degree += 1
                        continue
                    if col == row:
                        degree += 1
                    break
            else:
                if unit_diagonal:
                    if col > row:
                        deps.append(col)
                        degree += 1
                        continue
                    break
                else:
                    if col > row:
                        deps.append(col)
                        degree += 1
                        continue
                    if col == row:
                        degree += 1
                    break
        indegree_init[row] = degree
        row_level = 1
        if deps:
            row_level = max(levels[col] for col in deps) + 1
        levels[row] = row_level
        level_buckets.setdefault(row_level, []).append(row)

    row_map = []
    level_ptr = [0]
    frontier_sizes = []
    for level_id in sorted(level_buckets):
        rows = level_buckets[level_id]
        frontier_sizes.append(len(rows))
        row_map.extend(rows)
        level_ptr.append(len(row_map))

    row_lengths64 = indptr64[1:] - indptr64[:-1]
    csr_row_idx32 = torch.repeat_interleave(
        torch.arange(n_rows, device=device, dtype=torch.int32),
        row_lengths64.to(torch.int64),
    ).contiguous()
    num_levels = len(frontier_sizes)
    max_frontier = max(frontier_sizes) if frontier_sizes else 0
    avg_frontier = (float(sum(frontier_sizes)) / float(num_levels)) if frontier_sizes else 0.0
    matrix_stats = {
        **base_stats,
        "num_levels": int(num_levels),
        "max_frontier": int(max_frontier),
        "avg_frontier": float(avg_frontier),
        "frontier_ratio": (float(max_frontier) / float(n_rows)) if n_rows > 0 else 0.0,
    }
    if minimal:
        return {
            "row_map32": torch.tensor(row_map, dtype=torch.int32, device=device),
            "level_ptr32": torch.zeros(1, dtype=torch.int32, device=device),
            "indegree_init32": torch.empty(0, dtype=torch.int32, device=device),
            "csr_row_idx32": torch.empty(0, dtype=torch.int32, device=device),
            "matrix_stats": matrix_stats,
        }
    return {
        "row_map32": torch.tensor(row_map, dtype=torch.int32, device=device),
        "level_ptr32": torch.tensor(level_ptr, dtype=torch.int32, device=device),
        "indegree_init32": torch.tensor(indegree_init, dtype=torch.int32, device=device),
        "csr_row_idx32": csr_row_idx32,
        "matrix_stats": matrix_stats,
    }


def _prepare_spsv_csr_system(
    data,
    indices64,
    indptr64,
    n_rows,
    n_cols,
    lower,
    trans_mode,
    unit_diagonal,
    requested_solve_kind=None,
    storage_view="csr_as_csc",
):
    if trans_mode == "N":
        data, indices64, indptr64 = _maybe_sort_csr_rows(
            data, indices64, indptr64, n_rows, n_cols, lower=lower
        )
        requested_route = _normalize_requested_spsv_route(requested_solve_kind, trans_mode)
        _, avg_nnz_per_row, max_nnz_per_row = _spsv_csr_row_length_summary(indptr64, n_rows)
        base_stats = _build_spsv_cw_matrix_stats(
            indptr64,
            n_rows,
            avg_nnz_per_row=avg_nnz_per_row,
            max_nnz_per_row=max_nnz_per_row,
        )
        default_block_nnz, default_max_segments = _auto_spsv_launch_config(
            indptr64,
            max_nnz_per_row=max_nnz_per_row,
        )
        if lower:
            nontrans_variant = "csr_u_lo_cw" if unit_diagonal else "csr_n_lo_cw"
        else:
            nontrans_variant = "csr_u_up_cw" if unit_diagonal else "csr_n_up_cw"
        level_meta = None
        nnz_meta = None
        auto_route = None
        auto_matrix_stats = base_stats
        if requested_route is None and _is_rocm_runtime() and not SPSV_ROCM_ENABLE_ADVANCED_AUTO:
            auto_route = "csr_cw"
        if requested_route is None and _supports_spsv_advanced_nontrans_routes(
            "N", lower, unit_diagonal, data.dtype
        ) and auto_route is None:
            auto_route = _choose_spsv_nontrans_auto_route(
                n_rows,
                base_stats,
                lower=lower,
                unit_diagonal=unit_diagonal,
                value_dtype=data.dtype,
            )
            if auto_route in ("csr_cw_levelschd", "csr_nnz_balance"):
                level_meta = _build_spsv_level_schedule_metadata(
                    indices64,
                    indptr64,
                    n_rows,
                    lower=lower,
                    unit_diagonal=unit_diagonal,
                )
                auto_matrix_stats = level_meta["matrix_stats"]
            if auto_route == "csr_nnz_balance":
                nnz_meta = _build_spsv_nnz_balance_metadata(
                    indices64,
                    indptr64,
                    n_rows,
                    lower=lower,
                    unit_diagonal=unit_diagonal,
                )

        effective_route = requested_route if requested_route is not None else auto_route

        if effective_route == "csr_cw":
            default_solve_kind = "csr_cw"
            matrix_stats = base_stats
            supported_solve_kinds = ("csr_cw",)
        elif effective_route == "csr_roc":
            if bool(unit_diagonal):
                raise ValueError("solve_kind='csr_roc' currently supports non-unit diagonal only")
            if level_meta is None:
                level_meta = _build_spsv_level_schedule_metadata(
                    indices64,
                    indptr64,
                    n_rows,
                    lower=lower,
                    unit_diagonal=unit_diagonal,
                )
            matrix_stats = level_meta["matrix_stats"]
            default_solve_kind = "csr_roc"
            supported_solve_kinds = ("csr_roc",)
        elif effective_route == "csr_smblk":
            if bool(unit_diagonal):
                raise ValueError("solve_kind='csr_smblk' currently supports non-unit diagonal only")
            matrix_stats = auto_matrix_stats if requested_route is None else base_stats
            default_solve_kind = "csr_smblk"
            supported_solve_kinds = ("csr_smblk",)
        elif effective_route == "csr_cw_levelschd":
            if level_meta is None:
                level_meta = _build_spsv_level_schedule_metadata(
                    indices64,
                    indptr64,
                    n_rows,
                    lower=lower,
                    unit_diagonal=unit_diagonal,
                )
            matrix_stats = level_meta["matrix_stats"]
            default_solve_kind = "csr_cw_levelschd"
            supported_solve_kinds = ("csr_cw_levelschd",)
        elif effective_route == "csr_nnz_balance":
            if nnz_meta is None:
                nnz_meta = _build_spsv_nnz_balance_metadata(
                    indices64,
                    indptr64,
                    n_rows,
                    lower=lower,
                    unit_diagonal=unit_diagonal,
                )
            matrix_stats = auto_matrix_stats if requested_route is None else nnz_meta["matrix_stats"]
            default_solve_kind = "csr_nnz_balance"
            supported_solve_kinds = ("csr_nnz_balance",)
        else:
            if not _supports_spsv_advanced_nontrans_routes(
                "N", lower, unit_diagonal, data.dtype
            ):
                matrix_stats = base_stats
                default_solve_kind = "csr_cw"
                supported_solve_kinds = ("csr_cw",)
            else:
                matrix_stats = auto_matrix_stats
                default_solve_kind = auto_route if auto_route is not None else "csr_smblk"
                supported_solve_kinds = (default_solve_kind,)
        route_name = nontrans_variant
        if default_solve_kind == "csr_roc":
            route_name = "csr_n_lo_roc" if lower else "csr_n_up_roc"
        elif default_solve_kind == "csr_smblk":
            route_name = "csr_n_lo_smblk" if lower else "csr_n_up_smblk"
        elif default_solve_kind == "csr_cw_levelschd":
            route_name = "csr_n_lo_cw_levelschd" if lower else "csr_n_up_cw_levelschd"
        elif default_solve_kind == "csr_nnz_balance":
            route_name = "csr_n_lo_nnz_balance" if lower else "csr_n_up_nnz_balance"
        cw_plan = {
            "solve_kind": default_solve_kind,
            "default_solve_kind": default_solve_kind,
            "supported_solve_kinds": tuple(supported_solve_kinds),
            "nontrans_variant": nontrans_variant,
            "kernel_data": data,
            "kernel_indices32": indices64.to(torch.int32),
            "kernel_indptr64": indptr64,
            "lower_eff": lower,
            "unit_diagonal": unit_diagonal,
            "default_block_nnz": default_block_nnz,
            "default_max_segments": default_max_segments,
            "storage_view": "csr",
            "cw_worker_count": _cw_worker_count(
                n_rows, matrix_stats["max_frontier"], matrix_stats["avg_nnz_per_row"], 1
            ),
            "matrix_stats": matrix_stats,
            "route_name": route_name,
            "level_row_map32": (
                level_meta["row_map32"]
                if level_meta is not None
                else torch.empty(0, dtype=torch.int32, device=data.device)
            ),
            "level_ptr32": (
                level_meta["level_ptr32"]
                if level_meta is not None
                else torch.zeros(1, dtype=torch.int32, device=data.device)
            ),
            "nnz_balance_indegree32": (
                nnz_meta["indegree_init32"]
                if nnz_meta is not None
                else torch.empty(0, dtype=torch.int32, device=data.device)
            ),
            "nnz_balance_row_idx32": (
                nnz_meta["csr_row_idx32"]
                if nnz_meta is not None
                else torch.empty(0, dtype=torch.int32, device=data.device)
            ),
            "nnz_balance_launch_order32": (
                nnz_meta["launch_order32"]
                if nnz_meta is not None
                else torch.empty(0, dtype=torch.int32, device=data.device)
            ),
        }
        _attach_spsv_complex_plan_views(cw_plan)
        return cw_plan

    lower_eff = not lower
    storage_view = _normalize_spsv_storage_view(storage_view)
    if storage_view != "csr_as_csc":
        raise ValueError("TRANS/CONJ SpSV only supports storage_view='csr_as_csc'")
    matrix_stats = _build_spsv_cw_matrix_stats(indptr64, n_rows)
    default_block_nnz, default_max_segments = _choose_transpose_family_launch_config(
        indptr64
    )
    cw_plan = {
        "solve_kind": "transpose_cw",
        "default_solve_kind": "transpose_cw",
        "supported_solve_kinds": ("transpose_cw",),
        "kernel_data": data,
        "kernel_indices32": indices64.to(torch.int32),
        "kernel_indptr64": indptr64,
        "lower_eff": lower_eff,
        "default_block_nnz": default_block_nnz,
        "default_max_segments": default_max_segments,
        "cw_worker_count": _cw_worker_count(
            n_rows, matrix_stats["max_frontier"], matrix_stats["avg_nnz_per_row"], 1
        ),
        "matrix_stats": matrix_stats,
        "storage_view": storage_view,
        "route_name": "transpose_cw",
    }
    _attach_spsv_complex_plan_views(cw_plan)
    return cw_plan


def _resolve_spsv_csr_runtime(
    data,
    indices,
    indptr,
    b,
    shape,
    lower,
    transpose,
    unit_diagonal=False,
    requested_solve_kind=None,
    storage_view="csr_as_csc",
):
    input_data = data
    input_indices = indices
    input_indptr = indptr
    trans_mode = _normalize_spsv_transpose_mode(transpose)
    data, input_index_dtype, indices, indptr, b, n_rows, n_cols = _prepare_spsv_inputs(
        data, indices, indptr, b, shape
    )
    original_output_dtype = None
    if n_rows != n_cols:
        raise ValueError(f"A must be square, got shape={shape}")
    if trans_mode == "N":
        _validate_spsv_non_trans_combo(data.dtype, input_index_dtype, "CSR")
    else:
        _validate_spsv_trans_combo(data.dtype, input_index_dtype, "CSR")
    requested_route = _normalize_requested_spsv_route(requested_solve_kind, trans_mode)

    preprocess_key = _csr_preprocess_cache_key(
        input_data,
        input_indices,
        input_indptr,
        (n_rows, n_cols),
        lower,
        trans_mode,
        unit_diagonal,
        requested_route,
        _normalize_spsv_storage_view(storage_view),
    )
    cached = _spsv_cache_get(_SPSV_CSR_PREPROCESS_CACHE, preprocess_key)
    if cached is None:
        cached = _prepare_spsv_csr_system(
            data,
            indices,
            indptr,
            n_rows,
            n_cols,
            lower,
            trans_mode,
            unit_diagonal,
            requested_solve_kind=requested_solve_kind,
            storage_view=storage_view,
        )
        _spsv_cache_put(
            _SPSV_CSR_PREPROCESS_CACHE,
            preprocess_key,
            cached,
            _SPSV_CSR_PREPROCESS_CACHE_SIZE,
        )
    return (
        data,
        b,
        original_output_dtype,
        trans_mode,
        n_rows,
        n_cols,
        cached,
    )


def _select_spsv_runtime_plan(solve_plan, trans_mode, requested_solve_kind=None):
    requested_route = _normalize_requested_spsv_route(requested_solve_kind, trans_mode)
    routed = _clone_spsv_plan(solve_plan)
    if requested_route is None:
        requested_route = str(
            solve_plan.get("default_solve_kind", solve_plan.get("solve_kind", "csr_cw"))
        )
    supported = tuple(solve_plan.get("supported_solve_kinds", (solve_plan.get("solve_kind"),)))
    if requested_route not in supported:
        raise ValueError(
            f"solve_kind={requested_route!r} is not available for this SpSV problem; "
            f"supported routes: {', '.join(str(route) for route in supported if route)}"
        )
    routed["solve_kind"] = requested_route
    if requested_route == "csr_cw":
        routed["route_name"] = str(solve_plan.get("nontrans_variant", requested_route))
    elif requested_route == "csr_roc":
        routed["route_name"] = "csr_n_lo_roc" if bool(routed.get("lower_eff", True)) else "csr_n_up_roc"
    elif requested_route == "csr_smblk":
        routed["route_name"] = "csr_n_lo_smblk" if bool(routed.get("lower_eff", True)) else "csr_n_up_smblk"
    elif requested_route == "csr_cw_levelschd":
        routed["route_name"] = (
            "csr_n_lo_cw_levelschd" if bool(routed.get("lower_eff", True)) else "csr_n_up_cw_levelschd"
        )
    elif requested_route == "csr_nnz_balance":
        routed["route_name"] = (
            "csr_n_lo_nnz_balance" if bool(routed.get("lower_eff", True)) else "csr_n_up_nnz_balance"
        )
    else:
        routed["route_name"] = requested_route
    return routed


def _auto_spsv_launch_config(indptr, block_nnz=None, max_segments=None, *, max_nnz_per_row=None):
    if max_nnz_per_row is None:
        if indptr.numel() <= 1:
            max_nnz_per_row = 0
        else:
            max_nnz_per_row = int((indptr[1:] - indptr[:-1]).max().item())

    auto_block = block_nnz is None
    if block_nnz is None:
        if max_nnz_per_row <= 64:
            block_nnz_use = 64
        elif max_nnz_per_row <= 256:
            block_nnz_use = 128
        elif max_nnz_per_row <= 1024:
            block_nnz_use = 256
        elif max_nnz_per_row <= 4096:
            block_nnz_use = 512
        elif max_nnz_per_row <= 16384:
            block_nnz_use = 1024
        else:
            block_nnz_use = 2048
    else:
        block_nnz_use = int(block_nnz)
        if block_nnz_use <= 0:
            raise ValueError("block_nnz must be a positive integer")

    required_segments = max(
        (max_nnz_per_row + block_nnz_use - 1) // block_nnz_use, 1
    )
    if max_segments is None:
        max_segments_use = required_segments
        if auto_block:
            while max_segments_use > 2048 and block_nnz_use < 65536:
                block_nnz_use *= 2
                max_segments_use = max(
                    (max_nnz_per_row + block_nnz_use - 1) // block_nnz_use, 1
                )
    else:
        max_segments_use = int(max_segments)
        if max_segments_use <= 0:
            raise ValueError("max_segments must be a positive integer")
        if max_segments_use < required_segments:
            raise ValueError(
                f"max_segments={max_segments_use} is too small; at least {required_segments} required"
            )

    return block_nnz_use, max_segments_use


def _triton_spsv_csr_cw_vector(
    data,
    indices,
    indptr,
    b_vec,
    n_rows,
    *,
    lower=True,
    unit_diagonal=False,
    diag_eps=1e-12,
    worker_count=None,
    matrix_stats=None,
    ready_in=None,
    row_counter_in=None,
):
    x = torch.zeros_like(b_vec)
    ready = ready_in if ready_in is not None else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    row_counter = (
        row_counter_in
        if row_counter_in is not None
        else torch.zeros(1, dtype=torch.int32, device=b_vec.device)
    )
    ready.zero_()
    row_counter.zero_()
    if n_rows == 0:
        return x
    if worker_count is None:
        matrix_stats = matrix_stats or {}
        worker_count = _resolve_cw_worker_count(n_rows, matrix_stats, 1)
    use_fp64_acc = data.dtype == torch.float64
    grid = (worker_count,)
    kernel = _spsv_csr_u_cw_kernel if unit_diagonal else _spsv_csr_n_cw_kernel
    kernel[grid](
        data,
        indices,
        indptr,
        b_vec,
        x,
        ready,
        row_counter,
        n_rows,
        LOWER=lower,
        REVERSE_ORDER=not lower,
        UNIT_DIAG=unit_diagonal,
        USE_FP64_ACC=use_fp64_acc,
        DIAG_EPS=diag_eps,
        SERIAL_EXECUTION=(worker_count == 1),
    )
    return x


def _triton_spsv_csr_cw_vector_complex(
    data,
    indices,
    indptr,
    b_vec,
    n_rows,
    *,
    lower=True,
    unit_diagonal=False,
    diag_eps=1e-12,
    worker_count=None,
    matrix_stats=None,
    data_ri_in=None,
    ready_in=None,
    row_counter_in=None,
):
    x = torch.zeros_like(b_vec)
    ready = ready_in if ready_in is not None else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    row_counter = (
        row_counter_in
        if row_counter_in is not None
        else torch.zeros(1, dtype=torch.int32, device=b_vec.device)
    )
    ready.zero_()
    row_counter.zero_()
    if n_rows == 0:
        return x

    data_ri = data_ri_in if data_ri_in is not None else _complex_interleaved_view(data)
    b_ri = torch.view_as_real(b_vec.contiguous()).reshape(-1).contiguous()
    component_dtype = _component_dtype_for_complex(data.dtype)
    use_fp64 = component_dtype == torch.float64
    x_ri = torch.view_as_real(x.contiguous()).reshape(-1).contiguous()

    if worker_count is None:
        matrix_stats = matrix_stats or {}
        worker_count = _resolve_cw_worker_count(n_rows, matrix_stats, 1)
    grid = (worker_count,)
    kernel = (
        _spsv_csr_u_cw_kernel_complex
        if unit_diagonal
        else _spsv_csr_n_cw_kernel_complex
    )
    kernel[grid](
        data_ri,
        indices,
        indptr,
        b_ri,
        x_ri,
        ready,
        row_counter,
        n_rows,
        LOWER=lower,
        REVERSE_ORDER=not lower,
        UNIT_DIAG=unit_diagonal,
        USE_FP64_ACC=use_fp64,
        DIAG_EPS=diag_eps,
        SERIAL_EXECUTION=(worker_count == 1),
    )
    return x

def _triton_spsv_csr_u_lo_cw_vector(*args, **kwargs):
    return _triton_spsv_csr_cw_vector(*args, lower=True, unit_diagonal=True, **kwargs)


def _triton_spsv_csr_n_lo_cw_vector(*args, **kwargs):
    return _triton_spsv_csr_cw_vector(*args, lower=True, unit_diagonal=False, **kwargs)


def _triton_spsv_csr_u_up_cw_vector(*args, **kwargs):
    return _triton_spsv_csr_cw_vector(*args, lower=False, unit_diagonal=True, **kwargs)


def _triton_spsv_csr_n_up_cw_vector(*args, **kwargs):
    return _triton_spsv_csr_cw_vector(*args, lower=False, unit_diagonal=False, **kwargs)


def _triton_spsv_csr_u_lo_cw_vector_complex(*args, **kwargs):
    return _triton_spsv_csr_cw_vector_complex(*args, lower=True, unit_diagonal=True, **kwargs)


def _triton_spsv_csr_n_lo_cw_vector_complex(*args, **kwargs):
    return _triton_spsv_csr_cw_vector_complex(*args, lower=True, unit_diagonal=False, **kwargs)


def _triton_spsv_csr_u_up_cw_vector_complex(*args, **kwargs):
    return _triton_spsv_csr_cw_vector_complex(*args, lower=False, unit_diagonal=True, **kwargs)


def _triton_spsv_csr_n_up_cw_vector_complex(*args, **kwargs):
    return _triton_spsv_csr_cw_vector_complex(*args, lower=False, unit_diagonal=False, **kwargs)


def _triton_spsv_csr_n_lo_roc_vector(
    data,
    indices,
    indptr,
    row_map,
    b_vec,
    n_rows,
    *,
    lower=True,
    unit_diagonal=False,
    diag_eps=1e-12,
    ready_in=None,
    level_ptr=None,
):
    x = torch.zeros_like(b_vec)
    ready = ready_in if ready_in is not None else torch.zeros(
        n_rows, dtype=torch.int32, device=b_vec.device
    )
    ready.zero_()
    if n_rows == 0:
        return x
    use_fp64_acc = data.dtype == torch.float64
    warp_size = SPSV_ROCM_ALG3_WARP_SIZE if _is_rocm_runtime() else 32
    if level_ptr is not None and int(level_ptr.numel()) > 1:
        level_ptr_cpu = level_ptr.detach().to("cpu")
        for level_id in range(int(level_ptr_cpu.numel()) - 1):
            start = int(level_ptr_cpu[level_id].item())
            end = int(level_ptr_cpu[level_id + 1].item())
            count = end - start
            if count <= 0:
                continue
            _spsv_csr_roc_kernel[(count,)](
                data,
                indices,
                indptr,
                row_map[start:end],
                b_vec,
                x,
                ready,
                n_rows,
                LOWER=lower,
                USE_FP64_ACC=use_fp64_acc,
                DIAG_EPS=diag_eps,
                WARP_SIZE=warp_size,
                LEVEL_SCHEDULED=True,
                num_warps=1,
            )
    else:
        _spsv_csr_roc_kernel[(n_rows,)](
            data,
            indices,
            indptr,
            row_map,
            b_vec,
            x,
            ready,
            n_rows,
            LOWER=lower,
            USE_FP64_ACC=use_fp64_acc,
            DIAG_EPS=diag_eps,
            WARP_SIZE=warp_size,
            LEVEL_SCHEDULED=False,
            num_warps=1,
        )
    return x


def _triton_spsv_csr_n_lo_roc_vector_complex(
    data,
    indices,
    indptr,
    row_map,
    b_vec,
    n_rows,
    *,
    lower=True,
    unit_diagonal=False,
    diag_eps=1e-12,
    data_ri_in=None,
    ready_in=None,
    level_ptr=None,
):
    x = torch.zeros_like(b_vec)
    ready = ready_in if ready_in is not None else torch.zeros(
        n_rows, dtype=torch.int32, device=b_vec.device
    )
    ready.zero_()
    if n_rows == 0:
        return x
    data_ri = data_ri_in if data_ri_in is not None else _complex_interleaved_view(data)
    b_ri = torch.view_as_real(b_vec.contiguous()).reshape(-1).contiguous()
    x_ri = torch.view_as_real(x.contiguous()).reshape(-1).contiguous()
    component_dtype = _component_dtype_for_complex(data.dtype)
    use_fp64 = component_dtype == torch.float64
    warp_size = SPSV_ROCM_ALG3_WARP_SIZE if _is_rocm_runtime() else 32
    if level_ptr is not None and int(level_ptr.numel()) > 1:
        level_ptr_cpu = level_ptr.detach().to("cpu")
        for level_id in range(int(level_ptr_cpu.numel()) - 1):
            start = int(level_ptr_cpu[level_id].item())
            end = int(level_ptr_cpu[level_id + 1].item())
            count = end - start
            if count <= 0:
                continue
            _spsv_csr_roc_kernel_complex[(count,)](
                data_ri,
                indices,
                indptr,
                row_map[start:end],
                b_ri,
                x_ri,
                ready,
                n_rows,
                LOWER=lower,
                USE_FP64_ACC=use_fp64,
                DIAG_EPS=diag_eps,
                WARP_SIZE=warp_size,
                LEVEL_SCHEDULED=True,
                num_warps=1,
            )
    else:
        _spsv_csr_roc_kernel_complex[(n_rows,)](
            data_ri,
            indices,
            indptr,
            row_map,
            b_ri,
            x_ri,
            ready,
            n_rows,
            LOWER=lower,
            USE_FP64_ACC=use_fp64,
            DIAG_EPS=diag_eps,
            WARP_SIZE=warp_size,
            LEVEL_SCHEDULED=False,
            num_warps=1,
        )
    return x


def _triton_spsv_csr_n_lo_smblk_vector(
    data,
    indices,
    indptr,
    b_vec,
    n_rows,
    *,
    lower=True,
    diag_eps=1e-12,
    ready_in=None,
):
    x = torch.zeros_like(b_vec)
    ready = ready_in if ready_in is not None else torch.zeros(
        n_rows, dtype=torch.int32, device=b_vec.device
    )
    ready.zero_()
    if n_rows == 0:
        return x
    use_fp64_acc = data.dtype == torch.float64
    _spsv_csr_smblk_kernel[(n_rows,)](
        data,
        indices,
        indptr,
        b_vec,
        x,
        ready,
        n_rows,
        LOWER=lower,
        REVERSE_ORDER=not lower,
        USE_FP64_ACC=use_fp64_acc,
        DIAG_EPS=diag_eps,
        WARP_SIZE=32,
        num_warps=1,
    )
    return x


def _triton_spsv_csr_n_lo_smblk_vector_complex(
    data,
    indices,
    indptr,
    b_vec,
    n_rows,
    *,
    lower=True,
    diag_eps=1e-12,
    data_ri_in=None,
    ready_in=None,
):
    x = torch.zeros_like(b_vec)
    ready = ready_in if ready_in is not None else torch.zeros(
        n_rows, dtype=torch.int32, device=b_vec.device
    )
    ready.zero_()
    if n_rows == 0:
        return x
    data_ri = data_ri_in if data_ri_in is not None else _complex_interleaved_view(data)
    b_ri = torch.view_as_real(b_vec.contiguous()).reshape(-1).contiguous()
    x_ri = torch.view_as_real(x.contiguous()).reshape(-1).contiguous()
    component_dtype = _component_dtype_for_complex(data.dtype)
    use_fp64 = component_dtype == torch.float64
    _spsv_csr_smblk_kernel_complex[(n_rows,)](
        data_ri,
        indices,
        indptr,
        b_ri,
        x_ri,
        ready,
        n_rows,
        LOWER=lower,
        REVERSE_ORDER=not lower,
        USE_FP64_ACC=use_fp64,
        DIAG_EPS=diag_eps,
        WARP_SIZE=32,
        num_warps=1,
    )
    return x


def _triton_spsv_csr_n_lo_cw_levelschd_vector(
    data,
    indices,
    indptr,
    row_map,
    b_vec,
    n_rows,
    *,
    lower=True,
    diag_eps=1e-12,
    ready_in=None,
):
    x = torch.zeros_like(b_vec)
    ready = ready_in if ready_in is not None else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    ready.zero_()
    if n_rows == 0:
        return x
    use_fp64_acc = data.dtype == torch.float64
    grid = (n_rows,)
    _spsv_csr_cw_levelschd_kernel[grid](
        data,
        indices,
        indptr,
        row_map,
        b_vec,
        x,
        ready,
        n_rows,
        USE_FP64_ACC=use_fp64_acc,
        DIAG_EPS=diag_eps,
        num_warps=1,
    )
    return x


def _triton_spsv_csr_n_lo_cw_levelschd_vector_complex(
    data,
    indices,
    indptr,
    row_map,
    b_vec,
    n_rows,
    *,
    lower=True,
    diag_eps=1e-12,
    data_ri_in=None,
    ready_in=None,
):
    x = torch.zeros_like(b_vec)
    ready = (
        ready_in
        if ready_in is not None
        else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    )
    ready.zero_()
    if n_rows == 0:
        return x
    data_ri = data_ri_in if data_ri_in is not None else _complex_interleaved_view(data)
    b_ri = torch.view_as_real(b_vec.contiguous()).reshape(-1).contiguous()
    x_ri = torch.view_as_real(x.contiguous()).reshape(-1).contiguous()
    component_dtype = _component_dtype_for_complex(data.dtype)
    use_fp64 = component_dtype == torch.float64
    grid = (n_rows,)
    _spsv_csr_cw_levelschd_kernel_complex[grid](
        data_ri,
        indices,
        indptr,
        row_map,
        b_ri,
        x_ri,
        ready,
        n_rows,
        USE_FP64_ACC=use_fp64,
        DIAG_EPS=diag_eps,
        num_warps=1,
    )
    return x


def _triton_spsv_csr_n_lo_nnz_balance_vector(
    data,
    indices,
    launch_order,
    row_idx,
    indegree_init,
    b_vec,
    n_rows,
    *,
    lower=True,
    diag_eps=1e-12,
    tmp_sum_in=None,
    ready_in=None,
    indegree_in=None,
):
    x = torch.zeros_like(b_vec)
    if n_rows == 0:
        return x
    tmp_sum = tmp_sum_in if tmp_sum_in is not None else torch.zeros_like(b_vec)
    ready = ready_in if ready_in is not None else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    indegree = (
        indegree_in
        if indegree_in is not None
        else torch.empty(n_rows, dtype=torch.int32, device=b_vec.device)
    )
    tmp_sum.zero_()
    ready.zero_()
    indegree.copy_(indegree_init)
    use_fp64_acc = data.dtype == torch.float64
    grid = (int(data.numel()),)
    _spsv_csr_nnz_balance_kernel[grid](
        launch_order,
        row_idx,
        indices,
        data,
        b_vec,
        x,
        tmp_sum,
        ready,
        indegree,
        int(data.numel()),
        LOWER=lower,
        USE_FP64_ACC=use_fp64_acc,
        DIAG_EPS=diag_eps,
        num_warps=1,
    )
    return x


def _triton_spsv_csr_n_lo_nnz_balance_vector_complex(
    data,
    indices,
    launch_order,
    row_idx,
    indegree_init,
    b_vec,
    n_rows,
    *,
    lower=True,
    diag_eps=1e-12,
    data_ri_in=None,
    tmp_sum_in=None,
    ready_in=None,
    indegree_in=None,
):
    x = torch.zeros_like(b_vec)
    if n_rows == 0:
        return x
    tmp_sum = tmp_sum_in if tmp_sum_in is not None else torch.zeros_like(b_vec)
    ready = (
        ready_in
        if ready_in is not None
        else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    )
    indegree = (
        indegree_in
        if indegree_in is not None
        else torch.empty(n_rows, dtype=torch.int32, device=b_vec.device)
    )
    tmp_sum.zero_()
    ready.zero_()
    indegree.copy_(indegree_init)
    data_ri = data_ri_in if data_ri_in is not None else _complex_interleaved_view(data)
    b_ri = torch.view_as_real(b_vec.contiguous()).reshape(-1).contiguous()
    x_ri = torch.view_as_real(x.contiguous()).reshape(-1).contiguous()
    tmp_sum_ri = torch.view_as_real(tmp_sum.contiguous()).reshape(-1).contiguous()
    component_dtype = _component_dtype_for_complex(data.dtype)
    use_fp64 = component_dtype == torch.float64
    grid = (int(data.numel()),)
    _spsv_csr_nnz_balance_kernel_complex[grid](
        launch_order,
        row_idx,
        indices,
        data_ri,
        b_ri,
        x_ri,
        tmp_sum_ri,
        ready,
        indegree,
        int(data.numel()),
        LOWER=lower,
        USE_FP64_ACC=use_fp64,
        DIAG_EPS=diag_eps,
        num_warps=1,
    )
    return x


def _triton_spsv_csr_transpose_cw_vector(
    data,
    indices,
    indptr,
    b_vec,
    n_rows,
    lower=True,
    unit_diagonal=False,
    block_nnz=None,
    max_segments=None,
    diag_eps=1e-12,
    block_nnz_use=None,
    max_segments_use=None,
    worker_count=None,
    matrix_stats=None,
    residual_in=None,
    indegree_in=None,
    row_counter_in=None,
    preprocessed=False,
):
    x = torch.zeros_like(b_vec)
    if n_rows == 0:
        return x
    residual = residual_in if residual_in is not None else b_vec.clone()
    indegree = (
        indegree_in
        if indegree_in is not None
        else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    )
    row_counter = (
        row_counter_in
        if row_counter_in is not None
        else torch.zeros(1, dtype=torch.int32, device=b_vec.device)
    )
    residual.copy_(b_vec)
    row_counter.zero_()
    if block_nnz_use is None or max_segments_use is None:
        block_nnz_use, max_segments_use = _choose_transpose_family_launch_config(
            indptr, block_nnz=block_nnz, max_segments=max_segments
        )
    if not preprocessed:
        _run_spsv_csc_preprocess(
            indices,
            indptr,
            indegree,
            n_rows,
            lower=lower,
            unit_diagonal=unit_diagonal,
            block_nnz_use=block_nnz_use,
            max_segments_use=max_segments_use,
        )
    if worker_count is None:
        matrix_stats = matrix_stats or {}
        worker_count = _resolve_cw_worker_count(n_rows, matrix_stats, 1)
    grid = (worker_count,)
    kernel = _spsv_csc_u_cw_kernel if unit_diagonal else _spsv_csc_n_cw_kernel
    kernel[grid](
        data,
        indices,
        indptr,
        indegree,
        residual,
        x,
        row_counter,
        n_rows,
        BLOCK_NNZ=block_nnz_use,
        MAX_SEGMENTS=max_segments_use,
        LOWER=lower,
        REVERSE_ORDER=not lower,
        UNIT_DIAG=unit_diagonal,
        DIAG_EPS=diag_eps,
    )
    return x


def _triton_spsv_csr_transpose_cw_vector_complex(
    data,
    indices,
    indptr,
    b_vec,
    n_rows,
    lower=True,
    unit_diagonal=False,
    conjugate=False,
    block_nnz=None,
    max_segments=None,
    diag_eps=1e-12,
    block_nnz_use=None,
    max_segments_use=None,
    worker_count=None,
    matrix_stats=None,
    data_ri_in=None,
    residual_in=None,
    indegree_in=None,
    row_counter_in=None,
    preprocessed=False,
):
    x = torch.zeros_like(b_vec)
    if n_rows == 0:
        return x
    if block_nnz_use is None or max_segments_use is None:
        block_nnz_use, max_segments_use = _choose_transpose_family_launch_config(
            indptr, block_nnz=block_nnz, max_segments=max_segments
        )

    residual_work = (
        residual_in if residual_in is not None else b_vec.contiguous().clone()
    )
    indegree = (
        indegree_in
        if indegree_in is not None
        else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    )
    row_counter = (
        row_counter_in
        if row_counter_in is not None
        else torch.zeros(1, dtype=torch.int32, device=b_vec.device)
    )
    residual_work.copy_(b_vec.contiguous())
    row_counter.zero_()
    if not preprocessed:
        _run_spsv_csc_preprocess(
            indices,
            indptr,
            indegree,
            n_rows,
            lower=lower,
            unit_diagonal=unit_diagonal,
            block_nnz_use=block_nnz_use,
            max_segments_use=max_segments_use,
        )
    data_ri = data_ri_in if data_ri_in is not None else _complex_interleaved_view(data)
    residual_ri = torch.view_as_real(residual_work).reshape(-1).contiguous()
    component_dtype = _component_dtype_for_complex(data.dtype)
    use_fp64 = component_dtype == torch.float64
    if component_dtype == torch.float16:
        x_ri_work = torch.zeros((n_rows, 2), dtype=torch.float32, device=b_vec.device)
        x_ri = x_ri_work.reshape(-1).contiguous()
    else:
        x_ri = torch.view_as_real(x.contiguous()).reshape(-1).contiguous()

    if worker_count is None:
        matrix_stats = matrix_stats or {}
        worker_count = _resolve_cw_worker_count(n_rows, matrix_stats, 1)
    grid = (worker_count,)
    kernel = (
        _spsv_csc_u_cw_kernel_complex
        if unit_diagonal
        else _spsv_csc_n_cw_kernel_complex
    )
    kernel[grid](
        data_ri,
        indices,
        indptr,
        indegree,
        residual_ri,
        x_ri,
        row_counter,
        n_rows,
        BLOCK_NNZ=block_nnz_use,
        MAX_SEGMENTS=max_segments_use,
        LOWER=lower,
        REVERSE_ORDER=not lower,
        UNIT_DIAG=unit_diagonal,
        CONJ_TRANS=conjugate,
        USE_FP64_ACC=use_fp64,
        DIAG_EPS=diag_eps,
    )
    if component_dtype == torch.float16:
        return torch.view_as_complex(x_ri_work.contiguous())
    return x


def _choose_transpose_family_launch_config(indptr, block_nnz=None, max_segments=None):
    if block_nnz is not None or max_segments is not None:
        return _auto_spsv_launch_config(indptr, block_nnz=block_nnz, max_segments=max_segments)

    if indptr.numel() <= 1:
        return 32, 1
    max_nnz_per_row = int((indptr[1:] - indptr[:-1]).max().item())
    for cand in (32, 64, 128, 256, 512, 1024):
        req = max((max_nnz_per_row + cand - 1) // cand, 1)
        if req <= 2048:
            return cand, req
    cand = 2048
    req = max((max_nnz_per_row + cand - 1) // cand, 1)
    return cand, req


def _run_spsv_csc_preprocess(
    indices,
    indptr,
    indegree,
    n_rows,
    *,
    lower,
    unit_diagonal,
    block_nnz_use,
    max_segments_use,
):
    indegree.zero_()
    if n_rows == 0:
        return indegree
    grid = (n_rows,)
    kernel = (
        _spsv_csc_u_preprocess_kernel
        if unit_diagonal
        else _spsv_csc_n_preprocess_kernel
    )
    kernel[grid](
        indices,
        indptr,
        indegree,
        n_rows,
        BLOCK_NNZ=block_nnz_use,
        MAX_SEGMENTS=max_segments_use,
        LOWER=lower,
        UNIT_DIAG=unit_diagonal,
    )
    return indegree


def _prepare_spsv_coo_inputs(data, row, col, b, shape):
    if not all(torch.is_tensor(t) for t in (data, row, col, b)):
        raise TypeError("data, row, col, b must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, row, col, b)):
        raise ValueError("data, row, col, b must all be CUDA tensors")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, col must be 1D")
    if row.numel() != data.numel() or col.numel() != data.numel():
        raise ValueError("data, row, col must have the same length")
    if b.ndim != 1:
        raise ValueError("b must be a 1D dense vector (DnVec)")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if b.numel() != n_rows:
        raise ValueError(f"b length must equal n_rows={n_rows}")

    if data.dtype not in (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ):
        raise TypeError(
            "data dtype must be one of: float32, float64, complex64, complex128"
        )
    if b.dtype != data.dtype:
        raise TypeError("b dtype must match data dtype")
    if row.dtype not in SUPPORTED_SPSV_INDEX_DTYPES:
        raise TypeError("row dtype must be torch.int32 or torch.int64")
    if col.dtype not in SUPPORTED_SPSV_INDEX_DTYPES:
        raise TypeError("col dtype must be torch.int32 or torch.int64")
    input_index_dtype = (
        torch.int64
        if row.dtype == torch.int64 or col.dtype == torch.int64
        else torch.int32
    )
    row64 = row.to(torch.int64).contiguous()
    col64 = col.to(torch.int64).contiguous()
    if col64.numel() > 0 and int(col64.max().item()) > _INDEX_LIMIT_INT32:
        raise ValueError(
            f"int64 index value {int(col64.max().item())} exceeds Triton int32 kernel range"
        )
    if row64.numel() > 0:
        if bool(torch.any(row64 < 0).item()):
            raise IndexError("row indices must be non-negative")
        if bool(torch.any(col64 < 0).item()):
            raise IndexError("col indices must be non-negative")
        max_row = int(row64.max().item())
        max_col = int(col64.max().item())
        if max_row >= n_rows:
            raise IndexError(f"row indices out of range for n_rows={n_rows}")
        if max_col >= n_cols:
            raise IndexError(f"col indices out of range for n_cols={n_cols}")

    return (
        data.contiguous(),
        input_index_dtype,
        row64,
        col64,
        b.contiguous(),
        n_rows,
        n_cols,
    )


def _build_coo_row_ptr(row_sorted, n_rows):
    row_ptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=row_sorted.device)
    if row_sorted.numel() > 0:
        nnz_per_row = torch.bincount(row_sorted, minlength=n_rows)
        row_ptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    return row_ptr


def _coo_order_for_spsv(data, row64, col64):
    if data.numel() == 0:
        return data, row64, col64
    key = row64
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)
    return data[order], row64[order], col64[order]


def _coo2csr_for_spsv(data, row64, col64, n_rows, assume_ordered=False):
    nnz = data.numel()
    if nnz == 0:
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device)
        indices = torch.empty(0, dtype=torch.int64, device=data.device)
        return data, indices, indptr

    if not assume_ordered:
        data, row64, col64 = _coo_order_for_spsv(data, row64, col64)

    indptr = _build_coo_row_ptr(row64, n_rows)
    indices = col64.to(torch.int64).contiguous()
    return data.contiguous(), indices, indptr


def _analyze_spsv_csr_descriptor(
    data,
    indices,
    indptr,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    solve_kind=None,
    compute_dtype=None,
    handle=None,
    workspace=None,
    storage_view="csr_as_csc",
    format_name="csr",
    clear_cache=False,
):
    if clear_cache:
        _clear_spsv_csr_preprocess_cache()
    n_rows = int(shape[0])
    dummy_b = torch.empty(n_rows, dtype=data.dtype, device=data.device)
    (
        matrix_data,
        _dummy_b,
        _original_output_dtype,
        trans_mode,
        n_rows,
        _n_cols,
        solve_plan,
    ) = _resolve_spsv_csr_runtime(
        data,
        indices,
        indptr,
        dummy_b,
        shape,
        lower,
        transpose,
        unit_diagonal,
        requested_solve_kind=solve_kind,
        storage_view=storage_view,
    )
    input_index_dtype = indices.dtype
    solve_plan = _select_spsv_runtime_plan(
        solve_plan, trans_mode, requested_solve_kind=solve_kind
    )
    compute_dtype = _spsv_effective_compute_dtype(
        matrix_data.dtype, trans_mode, compute_dtype=compute_dtype
    )
    layout = _build_spsv_workspace_layout(
        n_rows, solve_plan["solve_kind"], value_dtype=compute_dtype
    )
    if workspace is not None:
        _resolve_spsv_workspace(workspace, layout, matrix_data.device)
    return FlagSparseSpSVDescr(
        format=_normalize_spsv_format(format_name),
        canonical_format="csr",
        shape=(int(shape[0]), int(shape[1])),
        lower=bool(lower),
        unit_diagonal=bool(unit_diagonal),
        fill_mode="lower" if lower else "upper",
        diag_type="unit" if unit_diagonal else "non_unit",
        matrix_type="triangular",
        index_base=0,
        transpose_mode=trans_mode,
        value_dtype=matrix_data.dtype,
        compute_dtype=compute_dtype,
        index_dtype=input_index_dtype,
        solve_kind=solve_plan["solve_kind"],
        route_name=str(solve_plan.get("route_name", solve_plan["solve_kind"])),
        storage_view=str(solve_plan.get("storage_view", "csr")),
        buffer_size=_workspace_size_bytes(layout),
        workspace_layout=layout,
        data=matrix_data,
        indices=indices.contiguous(),
        indptr=indptr.contiguous(),
        solve_plan=_clone_spsv_plan(solve_plan),
    )


def flagsparse_spsv_analysis_csr(
    data,
    indices,
    indptr,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    solve_kind=None,
    compute_dtype=None,
    handle=None,
    workspace=None,
    storage_view="csr_as_csc",
    clear_cache=False,
):
    """Analyze a CSR SpSV problem and return a reusable Triton descriptor."""

    return _analyze_spsv_csr_descriptor(
        data,
        indices,
        indptr,
        shape,
        lower=lower,
        unit_diagonal=unit_diagonal,
        transpose=transpose,
        solve_kind=solve_kind,
        compute_dtype=compute_dtype,
        handle=handle,
        workspace=workspace,
        storage_view=storage_view,
        format_name="csr",
        clear_cache=clear_cache,
    )


def _analyze_spsv_csr(
    data,
    indices,
    indptr,
    b,
    shape,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    solve_kind=None,
    clear_cache=False,
    return_time=False,
):
    if clear_cache:
        _clear_spsv_csr_preprocess_cache()
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    (
        _data,
        _b,
        _original_output_dtype,
        trans_mode,
        _n_rows,
        _n_cols,
        solve_plan,
    ) = _resolve_spsv_csr_runtime(
        data,
        indices,
        indptr,
        b,
        shape,
        lower,
        transpose,
        unit_diagonal,
        requested_solve_kind=solve_kind,
    )
    _select_spsv_runtime_plan(
        solve_plan, trans_mode, requested_solve_kind=solve_kind
    )
    if return_time:
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0


def flagsparse_spsv_analysis_coo(
    data,
    row,
    col,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    solve_kind=None,
    compute_dtype=None,
    handle=None,
    workspace=None,
    storage_view="csr_as_csc",
):
    """Analyze a COO SpSV problem by canonicalizing COO into CSR first."""

    dummy_b = torch.empty(int(shape[0]), dtype=data.dtype, device=data.device)
    data, _input_index_dtype, row64, col64, _b, n_rows, _n_cols = _prepare_spsv_coo_inputs(
        data, row, col, dummy_b, shape
    )
    trans_mode = _normalize_spsv_transpose_mode(transpose)
    if trans_mode == "N":
        _validate_spsv_non_trans_combo(data.dtype, row.dtype, "COO")
    else:
        _validate_spsv_trans_combo(data.dtype, row.dtype, "COO")
    data_csr, indices_csr, indptr_csr = _coo2csr_for_spsv(
        data, row64, col64, n_rows, assume_ordered=False
    )
    return _analyze_spsv_csr_descriptor(
        data_csr,
        indices_csr,
        indptr_csr,
        shape,
        lower=lower,
        unit_diagonal=unit_diagonal,
        transpose=transpose,
        solve_kind=solve_kind,
        compute_dtype=compute_dtype,
        handle=handle,
        workspace=workspace,
        storage_view=storage_view,
        format_name="coo",
        clear_cache=False,
    )


def _execute_spsv_csr_plan(
    data,
    b,
    solve_plan,
    trans_mode,
    n_rows,
    *,
    alpha=1,
    unit_diagonal=False,
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
    workspace=None,
    original_output_dtype=None,
    compute_dtype=None,
    handle=None,
    stream=None,
):
    solve_plan = _clone_spsv_plan(solve_plan)
    solve_kind = solve_plan["solve_kind"]
    kernel_data = solve_plan["kernel_data"]
    kernel_indices32 = solve_plan["kernel_indices32"]
    kernel_indptr64 = solve_plan["kernel_indptr64"]
    default_block_nnz = solve_plan["default_block_nnz"]
    default_max_segments = solve_plan["default_max_segments"]
    cw_worker_count = solve_plan.get("cw_worker_count")
    nontrans_variant = solve_plan.get("nontrans_variant", "csr_n_lo_cw")
    lower_eff = solve_plan["lower_eff"]
    matrix_stats = solve_plan.get("matrix_stats", {})
    level_row_map32 = solve_plan.get("level_row_map32")
    level_ptr32 = solve_plan.get("level_ptr32")
    nnz_balance_row_idx32 = solve_plan.get("nnz_balance_row_idx32")
    nnz_balance_indegree32 = solve_plan.get("nnz_balance_indegree32")
    nnz_balance_launch_order32 = solve_plan.get("nnz_balance_launch_order32")
    kernel_indices = kernel_indices32
    kernel_indptr = kernel_indptr64
    compute_dtype = _spsv_effective_compute_dtype(
        data.dtype, trans_mode, compute_dtype=compute_dtype
    )
    data_in = kernel_data
    b_in = b
    if compute_dtype != data.dtype:
        data_in = kernel_data.to(compute_dtype)
        b_in = b.to(compute_dtype)
    alpha_in = _coerce_spsv_alpha(alpha, compute_dtype, b.device)
    if not _spsv_alpha_is_identity(alpha):
        b_in = b_in * alpha_in
    solve_stream = _resolve_spsv_stream(handle, stream, b.device)

    if solve_kind == "transpose_cw":
        if block_nnz is None and max_segments is None:
            block_nnz_use, max_segments_use = default_block_nnz, default_max_segments
        else:
            block_nnz_use, max_segments_use = _choose_transpose_family_launch_config(
                kernel_indptr, block_nnz=block_nnz, max_segments=max_segments
            )
        vec_real = _triton_spsv_csr_transpose_cw_vector
        vec_complex = _triton_spsv_csr_transpose_cw_vector_complex
    elif solve_kind == "csr_cw":
        block_nnz_use, max_segments_use = default_block_nnz, default_max_segments
        nontrans_real_wrappers = {
            "csr_u_lo_cw": _triton_spsv_csr_u_lo_cw_vector,
            "csr_n_lo_cw": _triton_spsv_csr_n_lo_cw_vector,
            "csr_u_up_cw": _triton_spsv_csr_u_up_cw_vector,
            "csr_n_up_cw": _triton_spsv_csr_n_up_cw_vector,
        }
        nontrans_complex_wrappers = {
            "csr_u_lo_cw": _triton_spsv_csr_u_lo_cw_vector_complex,
            "csr_n_lo_cw": _triton_spsv_csr_n_lo_cw_vector_complex,
            "csr_u_up_cw": _triton_spsv_csr_u_up_cw_vector_complex,
            "csr_n_up_cw": _triton_spsv_csr_n_up_cw_vector_complex,
        }
        vec_real = nontrans_real_wrappers[nontrans_variant]
        vec_complex = nontrans_complex_wrappers[nontrans_variant]
    elif solve_kind == "csr_roc":
        block_nnz_use, max_segments_use = default_block_nnz, default_max_segments
        vec_real = _triton_spsv_csr_n_lo_roc_vector
        vec_complex = _triton_spsv_csr_n_lo_roc_vector_complex
    elif solve_kind == "csr_smblk":
        block_nnz_use, max_segments_use = default_block_nnz, default_max_segments
        vec_real = _triton_spsv_csr_n_lo_smblk_vector
        vec_complex = _triton_spsv_csr_n_lo_smblk_vector_complex
    elif solve_kind == "csr_cw_levelschd":
        block_nnz_use, max_segments_use = default_block_nnz, default_max_segments
        vec_real = _triton_spsv_csr_n_lo_cw_levelschd_vector
        vec_complex = _triton_spsv_csr_n_lo_cw_levelschd_vector_complex
    elif solve_kind == "csr_nnz_balance":
        block_nnz_use, max_segments_use = default_block_nnz, default_max_segments
        vec_real = _triton_spsv_csr_n_lo_nnz_balance_vector
        vec_complex = _triton_spsv_csr_n_lo_nnz_balance_vector_complex
    else:
        raise RuntimeError(f"unexpected SpSV solve kind: {solve_kind}")
    diag_eps = _spsv_diag_eps_for_dtype(compute_dtype)

    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    worker_count_use = cw_worker_count
    matrix_stats_use = dict(matrix_stats)
    if solve_kind in ("csr_cw", "transpose_cw"):
        worker_count_use = _resolve_cw_worker_count(
            n_rows,
            matrix_stats_use,
            1,
            cached_worker_count=cw_worker_count,
        )
    # Keep ALG1 serial on ROCm/DCU because cross-program ready-flag polling does
    # not make forward progress reliably on the current gfx936 Triton stack.
    if solve_kind == "csr_cw" and _is_rocm_runtime():
        worker_count_use = 1
    complex_kernel_data_ri = None
    if torch.is_complex(data_in):
        if compute_dtype == solve_plan["kernel_data"].dtype:
            complex_kernel_data_ri = solve_plan.get("kernel_data_ri")
        if complex_kernel_data_ri is None:
            complex_kernel_data_ri = _complex_interleaved_view(data_in)
    workspace_buffers = _resolve_spsv_workspace(
        workspace,
        _build_spsv_workspace_layout(n_rows, solve_kind, value_dtype=compute_dtype),
        b.device,
    )
    ready_buf = workspace_buffers.get("ready")
    tmp_sum_buf = workspace_buffers.get("tmp_sum")
    residual_buf = workspace_buffers.get("residual")
    indegree_buf = workspace_buffers.get("indegree")
    row_counter_buf = workspace_buffers.get("row_counter")
    transpose_preprocessed = False
    if solve_kind == "csr_nnz_balance":
        if tmp_sum_buf is None or ready_buf is None or indegree_buf is None:
            raise RuntimeError("csr_nnz_balance workspace is missing required buffers")
        tmp_sum_buf.zero_()
        ready_buf.zero_()
        indegree_buf.copy_(nnz_balance_indegree32)
    if solve_kind == "transpose_cw":
        if residual_buf is None or indegree_buf is None or row_counter_buf is None:
            raise RuntimeError("transpose_cw workspace is missing required buffers")
        transpose_sig = _transpose_cw_preprocess_signature(
            solve_plan,
            n_rows,
            unit_diagonal,
            block_nnz_use,
            max_segments_use,
        )
        preprocess_stream_ctx = (
            torch.cuda.stream(solve_stream)
            if solve_stream is not None
            else nullcontext()
        )
        with preprocess_stream_ctx:
            _run_spsv_csc_preprocess(
                kernel_indices,
                kernel_indptr,
                indegree_buf,
                n_rows,
                lower=lower_eff,
                unit_diagonal=unit_diagonal,
                block_nnz_use=block_nnz_use,
                max_segments_use=max_segments_use,
            )
        transpose_preprocessed = True
        if isinstance(workspace, FlagSparseSpSVWorkspace):
            workspace.prepared_solve_kind = "transpose_cw"
            workspace.prepared_signature = transpose_sig
    stream_ctx = (
        torch.cuda.stream(solve_stream)
        if solve_stream is not None
        else nullcontext()
    )
    with stream_ctx:
        if torch.is_complex(data_in):
            if vec_complex is None:
                raise ValueError(f"solve_kind={solve_kind!r} currently supports real dtypes only")
            if solve_kind == "transpose_cw":
                x = vec_complex(
                data_in,
                kernel_indices,
                kernel_indptr,
                b_in,
                n_rows,
                lower=lower_eff,
                unit_diagonal=unit_diagonal,
                conjugate=(trans_mode == "C"),
                block_nnz=block_nnz,
                max_segments=max_segments,
                diag_eps=diag_eps,
                block_nnz_use=block_nnz_use,
                max_segments_use=max_segments_use,
                worker_count=worker_count_use,
                matrix_stats=matrix_stats_use,
                data_ri_in=complex_kernel_data_ri,
                residual_in=residual_buf,
                indegree_in=indegree_buf,
                row_counter_in=row_counter_buf,
                preprocessed=transpose_preprocessed,
                )
            else:
                if solve_kind == "csr_roc":
                    x = vec_complex(
                    data_in,
                    kernel_indices,
                    kernel_indptr,
                    level_row_map32,
                    b_in,
                    n_rows,
                    lower=lower_eff,
                    unit_diagonal=unit_diagonal,
                    diag_eps=diag_eps,
                    data_ri_in=complex_kernel_data_ri,
                    ready_in=ready_buf,
                    level_ptr=level_ptr32,
                    )
                elif solve_kind == "csr_smblk":
                    x = vec_complex(
                    data_in,
                    kernel_indices,
                    kernel_indptr,
                    b_in,
                    n_rows,
                    lower=lower_eff,
                    diag_eps=diag_eps,
                    data_ri_in=complex_kernel_data_ri,
                    ready_in=ready_buf,
                    )
                elif solve_kind == "csr_cw_levelschd":
                    x = vec_complex(
                    data_in,
                    kernel_indices,
                    kernel_indptr,
                    level_row_map32,
                    b_in,
                    n_rows,
                    lower=lower_eff,
                    diag_eps=diag_eps,
                    data_ri_in=complex_kernel_data_ri,
                    ready_in=ready_buf,
                    )
                elif solve_kind == "csr_nnz_balance":
                    x = vec_complex(
                    data_in,
                    kernel_indices,
                    nnz_balance_launch_order32,
                    nnz_balance_row_idx32,
                    nnz_balance_indegree32,
                    b_in,
                    n_rows,
                    lower=lower_eff,
                    diag_eps=diag_eps,
                    data_ri_in=complex_kernel_data_ri,
                    tmp_sum_in=tmp_sum_buf,
                    ready_in=ready_buf,
                    indegree_in=indegree_buf,
                    )
                else:
                    x = vec_complex(
                    data_in,
                    kernel_indices,
                    kernel_indptr,
                    b_in,
                    n_rows,
                    diag_eps=diag_eps,
                    worker_count=worker_count_use,
                    matrix_stats=matrix_stats_use,
                    data_ri_in=complex_kernel_data_ri,
                    ready_in=ready_buf,
                    row_counter_in=row_counter_buf,
                    )
        else:
            if solve_kind == "transpose_cw":
                x = vec_real(
                data_in,
                kernel_indices,
                kernel_indptr,
                b_in,
                n_rows,
                lower=lower_eff,
                unit_diagonal=unit_diagonal,
                block_nnz=block_nnz,
                max_segments=max_segments,
                diag_eps=diag_eps,
                block_nnz_use=block_nnz_use,
                max_segments_use=max_segments_use,
                worker_count=worker_count_use,
                matrix_stats=matrix_stats_use,
                residual_in=residual_buf,
                indegree_in=indegree_buf,
                row_counter_in=row_counter_buf,
                preprocessed=transpose_preprocessed,
                )
            elif solve_kind == "csr_roc":
                x = vec_real(
                data_in,
                kernel_indices,
                kernel_indptr,
                level_row_map32,
                b_in,
                n_rows,
                lower=lower_eff,
                unit_diagonal=unit_diagonal,
                diag_eps=diag_eps,
                ready_in=ready_buf,
                level_ptr=level_ptr32,
                )
            elif solve_kind == "csr_smblk":
                x = vec_real(
                data_in,
                kernel_indices,
                kernel_indptr,
                b_in,
                n_rows,
                lower=lower_eff,
                diag_eps=diag_eps,
                ready_in=ready_buf,
                )
            elif solve_kind == "csr_cw_levelschd":
                x = vec_real(
                data_in,
                kernel_indices,
                kernel_indptr,
                level_row_map32,
                b_in,
                n_rows,
                lower=lower_eff,
                diag_eps=diag_eps,
                ready_in=ready_buf,
                )
            elif solve_kind == "csr_nnz_balance":
                x = vec_real(
                data_in,
                kernel_indices,
                nnz_balance_launch_order32,
                nnz_balance_row_idx32,
                nnz_balance_indegree32,
                b_in,
                n_rows,
                lower=lower_eff,
                diag_eps=diag_eps,
                tmp_sum_in=tmp_sum_buf,
                ready_in=ready_buf,
                indegree_in=indegree_buf,
                )
            else:
                x = vec_real(
                data_in,
                kernel_indices,
                kernel_indptr,
                b_in,
                n_rows,
                diag_eps=diag_eps,
                worker_count=worker_count_use,
                matrix_stats=matrix_stats_use,
                ready_in=ready_buf,
                row_counter_in=row_counter_buf,
                )
    target_dtype = original_output_dtype if original_output_dtype is not None else data.dtype
    if x.dtype != target_dtype:
        x = x.to(target_dtype)
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if out is not None:
        if out.shape != x.shape or out.dtype != x.dtype:
            raise ValueError("out shape/dtype must match result")
        out.copy_(x)
        x = out

    if return_time:
        return x, elapsed_ms
    return x


def flagsparse_spsv_solve_csr(
    descr,
    b,
    *,
    alpha=1,
    compute_dtype=None,
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
    workspace=None,
    handle=None,
    stream=None,
):
    """Solve a previously analyzed CSR SpSV problem."""

    if not isinstance(descr, FlagSparseSpSVDescr):
        raise TypeError("descr must be a FlagSparseSpSVDescr")
    if descr.canonical_format != "csr":
        raise ValueError("descr must reference a CSR-canonicalized SpSV analysis")
    if not torch.is_tensor(b):
        raise TypeError("b must be a torch.Tensor")
    if not b.is_cuda:
        raise ValueError("b must be a CUDA tensor")
    if b.ndim != 1:
        raise ValueError("b must be a 1D dense vector (DnVec)")
    if int(b.numel()) != int(descr.shape[0]):
        raise ValueError(f"b length must equal n_rows={descr.shape[0]}")
    if b.dtype != descr.value_dtype:
        raise TypeError("b dtype must match the analyzed matrix dtype")
    return _execute_spsv_csr_plan(
        descr.data,
        b.contiguous(),
        descr.solve_plan,
        descr.transpose_mode,
        int(descr.shape[0]),
        alpha=alpha,
        unit_diagonal=descr.unit_diagonal,
        block_nnz=block_nnz,
        max_segments=max_segments,
        out=out,
        return_time=return_time,
        workspace=workspace,
        original_output_dtype=descr.value_dtype,
        compute_dtype=compute_dtype if compute_dtype is not None else descr.compute_dtype,
        handle=handle,
        stream=stream,
    )


def flagsparse_spsv_solve_coo(
    descr,
    b,
    *,
    alpha=1,
    compute_dtype=None,
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
    workspace=None,
    handle=None,
    stream=None,
):
    """Solve a previously analyzed COO SpSV problem via its CSR canonical form."""

    return flagsparse_spsv_solve_csr(
        descr,
        b,
        alpha=alpha,
        compute_dtype=compute_dtype,
        block_nnz=block_nnz,
        max_segments=max_segments,
        out=out,
        return_time=return_time,
        workspace=workspace,
        handle=handle,
        stream=stream,
    )


def _materialize_spsv_workspace_state(descr, workspace=None):
    if not isinstance(descr, FlagSparseSpSVDescr):
        raise TypeError("descr must be a FlagSparseSpSVDescr")
    buffers = _resolve_spsv_workspace(
        workspace, descr.workspace_layout, descr.data.device
    )
    solve_kind = descr.solve_kind
    preprocess_sig = None
    if solve_kind in {
        "csr_cw",
        "csr_roc",
        "csr_smblk",
        "csr_cw_levelschd",
        "csr_nnz_balance",
    }:
        if isinstance(workspace, FlagSparseSpSVWorkspace):
            workspace.prepared_solve_kind = ""
            workspace.prepared_signature = None
    elif solve_kind == "transpose_cw":
        residual = buffers.get("residual")
        indegree = buffers.get("indegree")
        row_counter = buffers.get("row_counter")
        block_nnz_use = int(descr.solve_plan["default_block_nnz"])
        max_segments_use = int(descr.solve_plan["default_max_segments"])
        preprocess_sig = _transpose_cw_preprocess_signature(
            descr.solve_plan,
            int(descr.shape[0]),
            bool(descr.unit_diagonal),
            block_nnz_use,
            max_segments_use,
        )
        if residual is not None:
            residual.zero_()
        if indegree is not None:
            _run_spsv_csc_preprocess(
                descr.solve_plan["kernel_indices32"],
                descr.solve_plan["kernel_indptr64"],
                indegree,
                int(descr.shape[0]),
                lower=bool(descr.solve_plan["lower_eff"]),
                unit_diagonal=bool(descr.unit_diagonal),
                block_nnz_use=block_nnz_use,
                max_segments_use=max_segments_use,
            )
        if row_counter is not None:
            row_counter.zero_()
        if isinstance(workspace, FlagSparseSpSVWorkspace):
            workspace.prepared_solve_kind = "transpose_cw"
            workspace.prepared_signature = preprocess_sig
    else:
        raise RuntimeError(f"unexpected SpSV solve kind: {solve_kind}")
    if workspace is None:
        return FlagSparseSpSVWorkspace(
            buffer_size=int(descr.buffer_size),
            layout=tuple(descr.workspace_layout),
            device=descr.data.device,
            buffers=buffers,
            prepared_solve_kind=(
                "transpose_cw" if solve_kind == "transpose_cw" else ""
            ),
            prepared_signature=preprocess_sig,
        )
    return workspace


def flagsparse_spsv_preprocess_csr(descr, *, workspace=None):
    """Materialize caller-managed workspace for a CSR SpSV descriptor."""

    return _materialize_spsv_workspace_state(descr, workspace=workspace)


def flagsparse_spsv_preprocess_coo(descr, *, workspace=None):
    """Materialize caller-managed workspace for a COO SpSV descriptor."""

    return _materialize_spsv_workspace_state(descr, workspace=workspace)


def flagsparse_spsv_buffer_size_ex(
    handle,
    opA,
    alpha,
    matA,
    vecX,
    vecY=None,
    *,
    compute_dtype=None,
    solve_kind=None,
    storage_view="csr_as_csc",
):
    if not isinstance(matA, FlagSparseSpMatDescr):
        raise TypeError("matA must be a FlagSparseSpMatDescr")
    if not isinstance(vecX, FlagSparseDnVecDescr):
        raise TypeError("vecX must be a FlagSparseDnVecDescr")
    return flagsparse_spsv_buffer_size(
        matA.shape,
        matA.values.dtype,
        format=matA.format,
        transpose=opA,
        solve_kind=solve_kind,
        compute_dtype=compute_dtype,
        alpha=alpha,
        handle=handle,
        vecX=vecX,
        vecY=vecY,
        storage_view=storage_view,
    )


def flagsparse_spsv_analysis_ex(
    handle,
    opA,
    alpha,
    matA,
    vecX,
    vecY=None,
    *,
    compute_dtype=None,
    solve_kind=None,
    workspace=None,
    storage_view="csr_as_csc",
    clear_cache=False,
):
    if not isinstance(matA, FlagSparseSpMatDescr):
        raise TypeError("matA must be a FlagSparseSpMatDescr")
    if not isinstance(vecX, FlagSparseDnVecDescr):
        raise TypeError("vecX must be a FlagSparseDnVecDescr")
    if matA.format == "csr":
        return flagsparse_spsv_analysis_csr(
            matA.values,
            matA.indices,
            matA.indptr_or_col,
            matA.shape,
            lower=matA.lower,
            unit_diagonal=matA.unit_diagonal,
            transpose=opA,
            solve_kind=solve_kind,
            compute_dtype=compute_dtype,
            handle=handle,
            workspace=workspace,
            storage_view=storage_view,
            clear_cache=clear_cache,
        )
    if matA.format == "coo":
        return flagsparse_spsv_analysis_coo(
            matA.values,
            matA.indices,
            matA.indptr_or_col,
            matA.shape,
            lower=matA.lower,
            unit_diagonal=matA.unit_diagonal,
            transpose=opA,
            solve_kind=solve_kind,
            compute_dtype=compute_dtype,
            handle=handle,
            workspace=workspace,
            storage_view=storage_view,
        )
    raise ValueError("matA.format must be 'csr' or 'coo'")


def flagsparse_spsv_solve_ex(
    handle,
    opA,
    alpha,
    matA,
    vecX,
    vecY=None,
    descr=None,
    *,
    compute_dtype=None,
    solve_kind=None,
    workspace=None,
    stream=None,
    storage_view="csr_as_csc",
    block_nnz=None,
    max_segments=None,
    return_time=False,
):
    if not isinstance(matA, FlagSparseSpMatDescr):
        raise TypeError("matA must be a FlagSparseSpMatDescr")
    if not isinstance(vecX, FlagSparseDnVecDescr):
        raise TypeError("vecX must be a FlagSparseDnVecDescr")
    out_tensor = None if vecY is None else vecY.values
    if matA.format == "csr":
        return flagsparse_spsv_csr(
            matA.values,
            matA.indices,
            matA.indptr_or_col,
            vecX.values,
            matA.shape,
            lower=matA.lower,
            unit_diagonal=matA.unit_diagonal,
            transpose=opA,
            alpha=alpha,
            compute_dtype=compute_dtype,
            block_nnz=block_nnz,
            max_segments=max_segments,
            out=out_tensor,
            return_time=return_time,
            descr=descr,
            workspace=workspace,
            solve_kind=solve_kind,
            handle=handle,
            stream=stream,
            storage_view=storage_view,
        )
    if matA.format == "coo":
        return flagsparse_spsv_coo(
            matA.values,
            matA.indices,
            matA.indptr_or_col,
            vecX.values,
            matA.shape,
            lower=matA.lower,
            unit_diagonal=matA.unit_diagonal,
            transpose=opA,
            alpha=alpha,
            compute_dtype=compute_dtype,
            block_nnz=block_nnz,
            max_segments=max_segments,
            out=out_tensor,
            return_time=return_time,
            descr=descr,
            workspace=workspace,
            solve_kind=solve_kind,
            handle=handle,
            stream=stream,
            storage_view=storage_view,
        )
    raise ValueError("matA.format must be 'csr' or 'coo'")


def flagsparse_spsv_csr(
    data,
    indices,
    indptr,
    b,
    shape,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    alpha=1,
    compute_dtype=None,
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
    descr=None,
    workspace=None,
    solve_kind=None,
    handle=None,
    stream=None,
    storage_view="csr_as_csc",
):
    """Sparse triangular solve using Triton CSR CW kernels.

    Current support matrix:
    - NON_TRANS: float32/float64/complex64/complex128 with int32/int64 indices
    - TRANS/CONJ: float32/float64/complex64/complex128 with int32/int64 indices
    """
    if descr is not None:
        if not isinstance(descr, FlagSparseSpSVDescr):
            raise TypeError("descr must be a FlagSparseSpSVDescr or None")
        return flagsparse_spsv_solve_csr(
            descr,
            b,
            alpha=alpha,
            compute_dtype=compute_dtype,
            block_nnz=block_nnz,
            max_segments=max_segments,
            out=out,
            return_time=return_time,
            workspace=workspace,
            handle=handle,
            stream=stream,
        )
    (
        data,
        b,
        original_output_dtype,
        trans_mode,
        n_rows,
        _n_cols,
        solve_plan,
    ) = _resolve_spsv_csr_runtime(
        data,
        indices,
        indptr,
        b,
        shape,
        lower,
        transpose,
        unit_diagonal,
        requested_solve_kind=solve_kind,
        storage_view=storage_view,
    )
    solve_plan = _select_spsv_runtime_plan(
        solve_plan, trans_mode, requested_solve_kind=solve_kind
    )
    return _execute_spsv_csr_plan(
        data,
        b,
        solve_plan,
        trans_mode,
        n_rows,
        alpha=alpha,
        unit_diagonal=unit_diagonal,
        block_nnz=block_nnz,
        max_segments=max_segments,
        out=out,
        return_time=return_time,
        workspace=workspace,
        original_output_dtype=original_output_dtype,
        compute_dtype=compute_dtype,
        handle=handle,
        stream=stream,
    )

def flagsparse_spsv_coo(
    data,
    row,
    col,
    b,
    shape,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    alpha=1,
    compute_dtype=None,
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
    descr=None,
    workspace=None,
    solve_kind=None,
    handle=None,
    stream=None,
    storage_view="csr_as_csc",
):
    """COO SpSV by canonicalizing COO into CSR, then reusing CSR SpSV."""
    if descr is not None:
        if not isinstance(descr, FlagSparseSpSVDescr):
            raise TypeError("descr must be a FlagSparseSpSVDescr or None")
        return flagsparse_spsv_solve_coo(
            descr,
            b,
            alpha=alpha,
            compute_dtype=compute_dtype,
            block_nnz=block_nnz,
            max_segments=max_segments,
            out=out,
            return_time=return_time,
            workspace=workspace,
            handle=handle,
            stream=stream,
        )
    data, input_index_dtype, row64, col64, b, n_rows, n_cols = _prepare_spsv_coo_inputs(
        data, row, col, b, shape
    )
    if n_rows != n_cols:
        raise ValueError(f"A must be square, got shape={shape}")

    trans_mode = _normalize_spsv_transpose_mode(transpose)
    if trans_mode == "N":
        _validate_spsv_non_trans_combo(data.dtype, input_index_dtype, "COO")
    else:
        _validate_spsv_trans_combo(data.dtype, input_index_dtype, "COO")
    data_csr, indices_csr, indptr_csr = _coo2csr_for_spsv(
        data, row64, col64, n_rows, assume_ordered=False
    )
    return flagsparse_spsv_csr(
        data_csr,
        indices_csr,
        indptr_csr,
        b,
        shape,
        lower=lower,
        unit_diagonal=unit_diagonal,
        transpose=transpose,
        alpha=alpha,
        compute_dtype=compute_dtype,
        block_nnz=block_nnz,
        max_segments=max_segments,
        out=out,
        return_time=return_time,
        workspace=workspace,
        solve_kind=solve_kind,
        handle=handle,
        stream=stream,
        storage_view=storage_view,
    )
