"""Sparse triangular matrix-matrix solve (SpSM) for CSR/COO."""

from collections import OrderedDict
import os

from ._common import *


SUPPORTED_SPSM_VALUE_DTYPES = (torch.float32, torch.float64)
SUPPORTED_SPSM_INDEX_DTYPES = (torch.int32, torch.int64)
SPSM_NON_TRANS_PRIMARY_COMBOS = (
    ("csr", torch.float32, torch.int32),
    ("csr", torch.float64, torch.int32),
    ("coo", torch.float32, torch.int32),
    ("coo", torch.float64, torch.int32),
)
def _spsm_env_flag(name, default="0"):
    return str(os.environ.get(name, default)).lower() in ("1", "true", "yes", "on")


SPSM_ENABLE_TILED_KERNEL = _spsm_env_flag("FLAGSPARSE_SPSM_ENABLE_TILED_KERNEL", "1")
SPSM_TILED_MIN_RHS = 16
_SPSM_PREPROCESS_CACHE = OrderedDict()
_SPSM_PREPROCESS_CACHE_SIZE = 8


def _clear_spsm_preprocess_cache():
    _SPSM_PREPROCESS_CACHE.clear()


def _is_non_transpose(op):
    op_str = str(op).upper()
    return op_str in ("NON_TRANS", "NON_TRANSPOSE", "N")


def _validate_spsm_non_trans_combo(fmt_name, value_dtype, index_dtype):
    combo = (str(fmt_name).lower(), value_dtype, index_dtype)
    if combo not in SPSM_NON_TRANS_PRIMARY_COMBOS:
        raise TypeError(
            f"{fmt_name} SpSM currently supports NON_TRANS combinations with int32 kernel indices: "
            "(float32, int32), (float64, int32)"
        )


def _validate_spsm_op_and_layout(opA, opB, major):
    if not _is_non_transpose(opA):
        raise NotImplementedError("Only op(A)=NON_TRANS is supported")
    if not _is_non_transpose(opB):
        raise NotImplementedError("Only op(B)=NON_TRANS is supported")
    if str(major).lower() != "row":
        raise NotImplementedError("Only row-major dense layout is supported")


def _prepare_spsm_csr_inputs(data, indices, indptr, B, shape, opA, opB, major):
    if not all(torch.is_tensor(t) for t in (data, indices, indptr, B)):
        raise TypeError("data, indices, indptr, B must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, indices, indptr, B)):
        raise ValueError("data, indices, indptr, B must all be CUDA tensors")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, indptr must be 1D")
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense matrix")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows != n_cols:
        raise ValueError(f"SpSM requires square A, got shape={shape}")
    if indptr.numel() != n_rows + 1:
        raise ValueError(f"indptr length must be n_rows+1={n_rows + 1}")
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")
    if B.shape[0] != n_rows:
        raise ValueError(f"B.shape[0] must equal n_rows={n_rows}")
    _validate_spsm_op_and_layout(opA, opB, major)
    if data.dtype not in SUPPORTED_SPSM_VALUE_DTYPES:
        raise TypeError("data dtype must be float32 or float64")
    if B.dtype != data.dtype:
        raise TypeError("B dtype must match data dtype")
    if indices.dtype not in SUPPORTED_SPSM_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")
    if indptr.dtype not in SUPPORTED_SPSM_INDEX_DTYPES:
        raise TypeError("indptr dtype must be torch.int32 or torch.int64")
    indices64 = indices.to(torch.int64).contiguous()
    indptr64 = indptr.to(torch.int64).contiguous()
    if data.numel() > 0:
        if bool(torch.any(indices64 < 0).item()):
            raise IndexError("indices must be non-negative")
        if int(indices64.max().item()) >= n_cols:
            raise IndexError("indices out of range")
        if int(indices64.max().item()) > _INDEX_LIMIT_INT32:
            raise ValueError("index value exceeds int32 kernel range")
    if indptr64.numel() > 0:
        if int(indptr64[0].item()) != 0:
            raise ValueError("indptr[0] must be 0")
        if int(indptr64[-1].item()) != int(data.numel()):
            raise ValueError("indptr[-1] must equal nnz")
        if bool(torch.any(indptr64[1:] < indptr64[:-1]).item()):
            raise ValueError("indptr must be non-decreasing")
    _validate_spsm_non_trans_combo("csr", data.dtype, torch.int32)
    return data.contiguous(), indices64, indptr64, B.contiguous(), n_rows, n_cols


def _prepare_spsm_coo_inputs(data, row, col, B, shape, opA, opB, major):
    if not all(torch.is_tensor(t) for t in (data, row, col, B)):
        raise TypeError("data, row, col, B must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, row, col, B)):
        raise ValueError("data, row, col, B must all be CUDA tensors")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, col must be 1D")
    if data.numel() != row.numel() or data.numel() != col.numel():
        raise ValueError("data, row, col must have same length")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows != n_cols:
        raise ValueError(f"SpSM requires square A, got shape={shape}")
    if B.ndim != 2 or B.shape[0] != n_rows:
        raise ValueError("B must be 2D and B.shape[0] == n_rows")
    _validate_spsm_op_and_layout(opA, opB, major)
    if data.dtype not in SUPPORTED_SPSM_VALUE_DTYPES:
        raise TypeError("data dtype must be float32 or float64")
    if B.dtype != data.dtype:
        raise TypeError("B dtype must match data dtype")
    if row.dtype not in SUPPORTED_SPSM_INDEX_DTYPES or col.dtype not in SUPPORTED_SPSM_INDEX_DTYPES:
        raise TypeError("row/col dtype must be torch.int32 or torch.int64")
    row64 = row.to(torch.int64).contiguous()
    col64 = col.to(torch.int64).contiguous()
    if row64.numel() > 0:
        if bool(torch.any(row64 < 0).item()) or bool(torch.any(col64 < 0).item()):
            raise IndexError("row/col must be non-negative")
        if int(row64.max().item()) >= n_rows:
            raise IndexError("row index out of range")
        if int(col64.max().item()) >= n_cols:
            raise IndexError("col index out of range")
        if max(int(row64.max().item()), int(col64.max().item())) > _INDEX_LIMIT_INT32:
            raise ValueError("row/col value exceeds int32 kernel range")
    _validate_spsm_non_trans_combo("coo", data.dtype, torch.int32)
    return data.contiguous(), row64, col64, B.contiguous(), n_rows, n_cols


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


def _spsm_cache_get(cache, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _spsm_cache_put(cache, key, value, max_entries):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_entries:
        cache.popitem(last=False)


def _spsm_preprocess_cache_key(fmt_name, tensors, shape, lower, unit_diagonal):
    return (
        str(fmt_name).lower(),
        bool(lower),
        bool(unit_diagonal),
        int(shape[0]),
        int(shape[1]),
        *(_tensor_cache_token(t) for t in tensors),
    )


def _prepare_spsm_diag(data, indices64, indptr64, n_rows, unit_diagonal=False):
    diag = torch.ones(n_rows, dtype=data.dtype, device=data.device)
    if unit_diagonal or n_rows == 0 or data.numel() == 0:
        return diag
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, device=data.device, dtype=torch.int64),
        indptr64[1:] - indptr64[:-1],
    )
    diag_mask = indices64 == row_ids
    if bool(torch.any(diag_mask).item()):
        diag.scatter_(0, row_ids[diag_mask], data[diag_mask])
    return diag


def _prepare_spsm_kernel_row_ptr(indptr64):
    if indptr64.numel() == 0:
        return indptr64.to(torch.int32)
    if int(indptr64[-1].item()) <= _INDEX_LIMIT_INT32:
        return indptr64.to(torch.int32)
    return indptr64


def _csr_row_ids_from_indptr(indptr64, n_rows):
    if n_rows == 0 or indptr64.numel() <= 1:
        return torch.empty(0, dtype=torch.int64, device=indptr64.device)
    return torch.repeat_interleave(
        torch.arange(n_rows, device=indptr64.device, dtype=torch.int64),
        indptr64[1:] - indptr64[:-1],
    )


def _csr_rows_sorted_by_col(indices64, indptr64, n_rows):
    if indices64.numel() <= 1:
        return True
    row_ids = _csr_row_ids_from_indptr(indptr64, n_rows)
    if row_ids.numel() <= 1:
        return True
    same_row = row_ids[1:] == row_ids[:-1]
    if not bool(torch.any(same_row).item()):
        return True
    return bool(torch.all(indices64[1:][same_row] >= indices64[:-1][same_row]).item())


def _prepare_spsm_csr_sorted_view(data, indices64, indptr64, n_rows, n_cols):
    row_ids = _csr_row_ids_from_indptr(indptr64, n_rows)
    if data.numel() <= 1:
        return data, indices64, row_ids
    if _csr_rows_sorted_by_col(indices64, indptr64, n_rows):
        return data, indices64, row_ids
    key = row_ids * max(1, n_cols) + indices64
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)
    return data[order].contiguous(), indices64[order].contiguous(), row_ids


def _prepare_spsm_csr_dependency_bounds(indices64, indptr64, n_rows, lower, row_ids=None):
    row_begin = indptr64[:-1].clone()
    row_end = indptr64[1:].clone()
    if n_rows == 0 or indices64.numel() == 0:
        return row_begin, row_begin if lower else row_end
    if row_ids is None:
        row_ids = _csr_row_ids_from_indptr(indptr64, n_rows)
    dep_mask = indices64 < row_ids if lower else indices64 > row_ids
    dep_counts = torch.bincount(row_ids[dep_mask], minlength=n_rows)
    if lower:
        dep_begin = row_begin
        dep_end = row_begin + dep_counts
    else:
        dep_begin = row_end - dep_counts
        dep_end = row_end
    return dep_begin, dep_end


def _prepare_spsm_csr_dependency_view(
    data, indices64, indptr64, n_rows, lower, row_ids=None
):
    dep_begin64, dep_end64 = _prepare_spsm_csr_dependency_bounds(
        indices64, indptr64, n_rows, lower=lower, row_ids=row_ids
    )
    dep_counts = dep_end64 - dep_begin64
    dep_ptr64 = torch.zeros(n_rows + 1, dtype=torch.int64, device=indptr64.device)
    if n_rows > 0:
        dep_ptr64[1:] = torch.cumsum(dep_counts, dim=0)
    total_dep_nnz = int(dep_ptr64[-1].item()) if dep_ptr64.numel() > 0 else 0
    if total_dep_nnz == 0:
        empty_data = torch.empty(0, dtype=data.dtype, device=data.device)
        empty_indices = torch.empty(0, dtype=torch.int64, device=indices64.device)
        return empty_data, empty_indices, dep_ptr64

    if row_ids is None:
        row_ids = _csr_row_ids_from_indptr(indptr64, n_rows)
    dep_mask = indices64 < row_ids if lower else indices64 > row_ids
    dep_data = data[dep_mask].contiguous()
    dep_indices64 = indices64[dep_mask].contiguous()
    return dep_data, dep_indices64, dep_ptr64


def _alpha_is_one(alpha):
    if torch.is_tensor(alpha):
        if alpha.numel() != 1:
            return False
        return bool((alpha.detach().cpu() == 1).item())
    return alpha == 1 or alpha == 1.0


def _alpha_to_host_scalar(alpha):
    if torch.is_tensor(alpha):
        if alpha.numel() != 1:
            raise ValueError("alpha tensor must be scalar")
        return float(alpha.detach().cpu().item())
    return float(alpha)


def _should_use_spsm_tiled_kernel(fmt_name, rhs, dtype):
    if not SPSM_ENABLE_TILED_KERNEL:
        return False
    if dtype not in (torch.float32, torch.float64):
        return False
    if rhs.ndim != 2:
        return False
    n_rhs = int(rhs.shape[1])
    if n_rhs < SPSM_TILED_MIN_RHS:
        return False
    fmt = str(fmt_name).lower()
    if fmt == "csr":
        return True
    if fmt == "coo":
        return True
    return False


@triton.jit
def _spsm_pack_rhs_work_kernel(
    src_ptr,
    dst_ptr,
    n_rows,
    n_rhs,
    stride_src0,
    stride_dst0,
    alpha,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_rhs = tl.program_id(1)

    row_offsets = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    rhs_offsets = pid_rhs * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    mask = (row_offsets[:, None] < n_rows) & (rhs_offsets[None, :] < n_rhs)

    src_ptrs = src_ptr + row_offsets[:, None] * stride_src0 + rhs_offsets[None, :]
    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    if USE_FP64_ACC:
        vals = vals.to(tl.float64) * alpha
    else:
        vals = vals.to(tl.float32) * alpha

    dst_ptrs = dst_ptr + row_offsets[:, None] * stride_dst0 + rhs_offsets[None, :]
    tl.store(dst_ptrs, vals, mask=mask)


@triton.jit
def _spsm_csr_level_kernel_real(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    diag_ptr,
    b_ptr,
    x_ptr,
    rows_ptr,
    n_level_rows,
    n_rhs,
    stride_b0,
    stride_x0,
    BLOCK_NNZ: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_rhs = tl.program_id(1)
    if pid_row >= n_level_rows:
        return

    row = tl.load(rows_ptr + pid_row)
    rhs_offsets = pid_rhs * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_offsets < n_rhs

    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)

    if USE_FP64_ACC:
        acc = tl.zeros((BLOCK_RHS,), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_RHS,), dtype=tl.float32)

    for seg in range(MAX_SEGMENTS):
        nnz_offsets = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
        nnz_mask = nnz_offsets < end
        a = tl.load(data_ptr + nnz_offsets, mask=nnz_mask, other=0.0)
        col = tl.load(indices_ptr + nnz_offsets, mask=nnz_mask, other=0)

        if USE_FP64_ACC:
            a = a.to(tl.float64)
        else:
            a = a.to(tl.float32)

        solved_mask = col < row if LOWER else col > row

        x_ptrs = x_ptr + col[:, None] * stride_x0 + rhs_offsets[None, :]
        x_mask = nnz_mask[:, None] & rhs_mask[None, :]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
        if USE_FP64_ACC:
            x_vals = x_vals.to(tl.float64)
        else:
            x_vals = x_vals.to(tl.float32)

        contrib = tl.where((nnz_mask & solved_mask)[:, None], a[:, None] * x_vals, 0.0)
        acc += tl.sum(contrib, axis=0)

    b_ptrs = b_ptr + row * stride_b0 + rhs_offsets
    rhs = tl.load(b_ptrs, mask=rhs_mask, other=0.0)
    rhs = rhs.to(tl.float64) if USE_FP64_ACC else rhs.to(tl.float32)

    diag = tl.load(diag_ptr + row)
    diag = diag.to(tl.float64) if USE_FP64_ACC else diag.to(tl.float32)
    diag_safe = tl.where(tl.abs(diag) < DIAG_EPS, 1.0, diag)
    out = (rhs - acc) / diag_safe
    out = tl.where(out == out, out, 0.0)

    out_ptrs = x_ptr + row * stride_x0 + rhs_offsets
    tl.store(out_ptrs, out, mask=rhs_mask)


@triton.jit
def _spsm_coo_level_kernel_real(
    data_ptr,
    row_ptr_ptr,
    col_ptr,
    diag_ptr,
    b_ptr,
    x_ptr,
    rows_ptr,
    n_level_rows,
    n_rhs,
    stride_b0,
    stride_x0,
    BLOCK_NNZ: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_rhs = tl.program_id(1)
    if pid_row >= n_level_rows:
        return

    row = tl.load(rows_ptr + pid_row)
    rhs_offsets = pid_rhs * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_offsets < n_rhs

    start = tl.load(row_ptr_ptr + row)
    end = tl.load(row_ptr_ptr + row + 1)

    if USE_FP64_ACC:
        acc = tl.zeros((BLOCK_RHS,), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_RHS,), dtype=tl.float32)

    for seg in range(MAX_SEGMENTS):
        nnz_offsets = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
        nnz_mask = nnz_offsets < end
        a = tl.load(data_ptr + nnz_offsets, mask=nnz_mask, other=0.0)
        col = tl.load(col_ptr + nnz_offsets, mask=nnz_mask, other=0)

        if USE_FP64_ACC:
            a = a.to(tl.float64)
        else:
            a = a.to(tl.float32)

        solved_mask = col < row if LOWER else col > row

        x_ptrs = x_ptr + col[:, None] * stride_x0 + rhs_offsets[None, :]
        x_mask = nnz_mask[:, None] & rhs_mask[None, :]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
        if USE_FP64_ACC:
            x_vals = x_vals.to(tl.float64)
        else:
            x_vals = x_vals.to(tl.float32)

        contrib = tl.where((nnz_mask & solved_mask)[:, None], a[:, None] * x_vals, 0.0)
        acc += tl.sum(contrib, axis=0)

    b_ptrs = b_ptr + row * stride_b0 + rhs_offsets
    rhs = tl.load(b_ptrs, mask=rhs_mask, other=0.0)
    rhs = rhs.to(tl.float64) if USE_FP64_ACC else rhs.to(tl.float32)

    diag = tl.load(diag_ptr + row)
    diag = diag.to(tl.float64) if USE_FP64_ACC else diag.to(tl.float32)
    diag_safe = tl.where(tl.abs(diag) < DIAG_EPS, 1.0, diag)
    out = (rhs - acc) / diag_safe
    out = tl.where(out == out, out, 0.0)

    out_ptrs = x_ptr + row * stride_x0 + rhs_offsets
    tl.store(out_ptrs, out, mask=rhs_mask)


@triton.jit
def _spsm_csr_tiled_kernel_real(
    data_ptr,
    indices_ptr,
    dep_ptr_ptr,
    diag_ptr,
    y_ptr,
    ready_ptr,
    n_rows,
    n_rhs,
    stride_y0,
    ready_stride0,
    BLOCK_NNZ: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    row = tl.program_id(0)
    pid_rhs = tl.program_id(1)
    if row >= n_rows:
        return

    rhs_offsets = pid_rhs * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    rhs_mask = rhs_offsets < n_rhs
    ready_row_ptr = ready_ptr + pid_rhs * ready_stride0

    start = tl.load(dep_ptr_ptr + row)
    end = tl.load(dep_ptr_ptr + row + 1)

    if USE_FP64_ACC:
        acc = tl.zeros((BLOCK_RHS,), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_RHS,), dtype=tl.float32)

    for seg in range(MAX_SEGMENTS):
        nnz_offsets = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
        nnz_mask = nnz_offsets < end
        a = tl.load(data_ptr + nnz_offsets, mask=nnz_mask, other=0.0)
        col = tl.load(indices_ptr + nnz_offsets, mask=nnz_mask, other=0)

        if USE_FP64_ACC:
            a = a.to(tl.float64)
        else:
            a = a.to(tl.float32)

        for k in range(BLOCK_NNZ):
            if nnz_mask[k]:
                dep_col = col[k]
                while tl.load(ready_row_ptr + dep_col) == 0:
                    pass
                x_ptrs = y_ptr + dep_col * stride_y0 + rhs_offsets
                x_vals = tl.load(x_ptrs, mask=rhs_mask, other=0.0)
                if USE_FP64_ACC:
                    x_vals = x_vals.to(tl.float64)
                else:
                    x_vals = x_vals.to(tl.float32)
                acc += a[k] * x_vals

    rhs_ptrs = y_ptr + row * stride_y0 + rhs_offsets
    rhs = tl.load(rhs_ptrs, mask=rhs_mask, other=0.0)
    rhs = rhs.to(tl.float64) if USE_FP64_ACC else rhs.to(tl.float32)

    diag = tl.load(diag_ptr + row)
    diag = diag.to(tl.float64) if USE_FP64_ACC else diag.to(tl.float32)
    diag_safe = tl.where(tl.abs(diag) < DIAG_EPS, 1.0, diag)
    out = (rhs - acc) / diag_safe
    out = tl.where(out == out, out, 0.0)

    out_ptrs = y_ptr + row * stride_y0 + rhs_offsets
    tl.store(out_ptrs, out, mask=rhs_mask)
    tl.debug_barrier()
    tl.store(ready_row_ptr + row, 1)


def _build_spsm_levels(indptr, indices, n_rows, lower=True):
    if n_rows == 0:
        return []
    indptr_h = indptr.to(torch.int64).cpu()
    indices_h = indices.to(torch.int64).cpu()
    levels = [0] * n_rows

    if lower:
        for i in range(n_rows):
            start = int(indptr_h[i].item())
            end = int(indptr_h[i + 1].item())
            level = 0
            for p in range(start, end):
                col = int(indices_h[p].item())
                if col < i:
                    level = max(level, levels[col] + 1)
            levels[i] = level
    else:
        for i in range(n_rows - 1, -1, -1):
            start = int(indptr_h[i].item())
            end = int(indptr_h[i + 1].item())
            level = 0
            for p in range(start, end):
                col = int(indices_h[p].item())
                if col > i:
                    level = max(level, levels[col] + 1)
            levels[i] = level

    max_level = max(levels)
    buckets = [[] for _ in range(max_level + 1)]
    for row, level in enumerate(levels):
        buckets[level].append(row)

    device = indptr.device
    return [torch.tensor(rows, dtype=torch.int32, device=device) for rows in buckets if rows]


def _auto_spsm_launch_config(indptr, block_nnz=None, max_segments=None):
    if indptr.numel() <= 1:
        max_nnz_per_row = 0
    else:
        row_lengths = indptr[1:] - indptr[:-1]
        max_nnz_per_row = int(row_lengths.max().item())

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
        else:
            block_nnz_use = 1024
    else:
        block_nnz_use = int(block_nnz)
        if block_nnz_use <= 0:
            raise ValueError("block_nnz must be positive")

    required_segments = max((max_nnz_per_row + block_nnz_use - 1) // block_nnz_use, 1)
    if max_segments is None:
        max_segments_use = required_segments
        if auto_block:
            while max_segments_use > 2048 and block_nnz_use < 65536:
                block_nnz_use *= 2
                max_segments_use = max((max_nnz_per_row + block_nnz_use - 1) // block_nnz_use, 1)
    else:
        max_segments_use = int(max_segments)
        if max_segments_use <= 0:
            raise ValueError("max_segments must be positive")
        if max_segments_use < required_segments:
            raise ValueError(
                f"max_segments={max_segments_use} is too small; at least {required_segments} required"
            )
    return block_nnz_use, max_segments_use


def _auto_spsm_launch_config_from_row_lengths(row_lengths, block_nnz=None, max_segments=None):
    if row_lengths.numel() == 0:
        max_nnz_per_row = 0
    else:
        max_nnz_per_row = int(row_lengths.max().item())

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
        else:
            block_nnz_use = 1024
    else:
        block_nnz_use = int(block_nnz)
        if block_nnz_use <= 0:
            raise ValueError("block_nnz must be positive")

    required_segments = max((max_nnz_per_row + block_nnz_use - 1) // block_nnz_use, 1)
    if max_segments is None:
        max_segments_use = required_segments
        if auto_block:
            while max_segments_use > 2048 and block_nnz_use < 65536:
                block_nnz_use *= 2
                max_segments_use = max((max_nnz_per_row + block_nnz_use - 1) // block_nnz_use, 1)
    else:
        max_segments_use = int(max_segments)
        if max_segments_use <= 0:
            raise ValueError("max_segments must be positive")
        if max_segments_use < required_segments:
            raise ValueError(
                f"max_segments={max_segments_use} is too small; at least {required_segments} required"
            )
    return block_nnz_use, max_segments_use


def _auto_rhs_block(n_rhs):
    n_rhs = int(n_rhs)
    if n_rhs <= 4:
        return 4
    if n_rhs <= 8:
        return 8
    if n_rhs <= 16:
        return 16
    if n_rhs <= 32:
        return 32
    return 64


def _pack_spsm_rhs_work_buffer(rhs, alpha):
    if rhs.ndim != 2:
        raise ValueError("rhs must be 2D")
    n_rows, n_rhs = int(rhs.shape[0]), int(rhs.shape[1])
    work = torch.empty_like(rhs, memory_format=torch.contiguous_format)
    if n_rows == 0 or n_rhs == 0:
        return work

    block_rows = 32
    block_rhs = _auto_rhs_block(n_rhs)
    alpha_value = _alpha_to_host_scalar(alpha)
    use_fp64 = rhs.dtype == torch.float64
    grid = (triton.cdiv(n_rows, block_rows), triton.cdiv(n_rhs, block_rhs))
    _spsm_pack_rhs_work_kernel[grid](
        rhs,
        work,
        n_rows,
        n_rhs,
        rhs.stride(0),
        work.stride(0),
        alpha_value,
        BLOCK_ROWS=block_rows,
        BLOCK_RHS=block_rhs,
        USE_FP64_ACC=use_fp64,
    )
    return work


def _coo_to_csr_sorted_unique(data, row64, col64, n_rows, n_cols):
    if data.numel() == 0:
        return (
            data,
            torch.empty(0, dtype=torch.int64, device=data.device),
            torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device),
        )
    key = row64 * max(1, n_cols) + col64
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)
    key_s = key[order]
    data_s = data[order]
    unique_key, inverse = torch.unique_consecutive(key_s, return_inverse=True)
    out_nnz = unique_key.numel()
    data_u = torch.zeros(out_nnz, dtype=data.dtype, device=data.device)
    data_u.scatter_add_(0, inverse, data_s)
    row_u = torch.div(unique_key, max(1, n_cols), rounding_mode="floor")
    col_u = unique_key - row_u * max(1, n_cols)
    indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device)
    if out_nnz > 0:
        nnz_per_row = torch.bincount(row_u, minlength=n_rows)
        indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    return data_u, col_u.to(torch.int64), indptr


def _prepare_spsm_csr_system(data, indices64, indptr64, n_rows, lower, unit_diagonal):
    sorted_data, sorted_indices64, row_ids = _prepare_spsm_csr_sorted_view(
        data, indices64, indptr64, n_rows, n_rows
    )
    dep_data, dep_indices64, dep_ptr64 = _prepare_spsm_csr_dependency_view(
        sorted_data, sorted_indices64, indptr64, n_rows, lower=lower, row_ids=row_ids
    )
    levels = _build_spsm_levels(indptr64, sorted_indices64, n_rows, lower=lower)
    default_block_nnz, default_max_segments = _auto_spsm_launch_config(indptr64)
    tiled_block_nnz, tiled_max_segments = _auto_spsm_launch_config_from_row_lengths(
        dep_ptr64[1:] - dep_ptr64[:-1]
    )
    return {
        "kernel_data": sorted_data,
        "kernel_indices32": sorted_indices64.to(torch.int32),
        "kernel_dep_data": dep_data,
        "kernel_dep_indices32": dep_indices64.to(torch.int32),
        "kernel_dep_ptr": _prepare_spsm_kernel_row_ptr(dep_ptr64),
        "kernel_diag": _prepare_spsm_diag(
            sorted_data, sorted_indices64, indptr64, n_rows, unit_diagonal=unit_diagonal
        ),
        "kernel_indptr": _prepare_spsm_kernel_row_ptr(indptr64),
        "launch_groups": levels,
        "default_block_nnz": default_block_nnz,
        "default_max_segments": default_max_segments,
        "tiled_block_nnz": tiled_block_nnz,
        "tiled_max_segments": tiled_max_segments,
        "lower_eff": bool(lower),
    }


def _prepare_spsm_coo_system(data, row64, col64, n_rows, n_cols, lower, unit_diagonal):
    data_u, col_u64, row_ptr = _coo_to_csr_sorted_unique(data, row64, col64, n_rows, n_cols)
    dep_data, dep_indices64, dep_ptr64 = _prepare_spsm_csr_dependency_view(
        data_u, col_u64, row_ptr, n_rows, lower=lower
    )
    levels = _build_spsm_levels(row_ptr, col_u64, n_rows, lower=lower)
    default_block_nnz, default_max_segments = _auto_spsm_launch_config(row_ptr)
    tiled_block_nnz, tiled_max_segments = _auto_spsm_launch_config_from_row_lengths(
        dep_ptr64[1:] - dep_ptr64[:-1]
    )
    return {
        "kernel_data": data_u,
        "kernel_cols32": col_u64.to(torch.int32),
        "kernel_dep_data": dep_data,
        "kernel_dep_indices32": dep_indices64.to(torch.int32),
        "kernel_dep_ptr": _prepare_spsm_kernel_row_ptr(dep_ptr64),
        "kernel_diag": _prepare_spsm_diag(
            data_u, col_u64, row_ptr, n_rows, unit_diagonal=unit_diagonal
        ),
        "kernel_row_ptr": _prepare_spsm_kernel_row_ptr(row_ptr),
        "launch_groups": levels,
        "default_block_nnz": default_block_nnz,
        "default_max_segments": default_max_segments,
        "tiled_block_nnz": tiled_block_nnz,
        "tiled_max_segments": tiled_max_segments,
        "lower_eff": bool(lower),
    }


def _resolve_spsm_csr_runtime(data, indices, indptr, B, shape, lower, unit_diagonal, opA, opB, major):
    data, indices64, indptr64, B, n_rows, n_cols = _prepare_spsm_csr_inputs(
        data, indices, indptr, B, shape, opA, opB, major
    )
    cache_key = _spsm_preprocess_cache_key(
        "csr",
        (data, indices64, indptr64),
        shape,
        lower,
        unit_diagonal,
    )
    solve_plan = _spsm_cache_get(_SPSM_PREPROCESS_CACHE, cache_key)
    if solve_plan is None:
        solve_plan = _prepare_spsm_csr_system(
            data, indices64, indptr64, n_rows, lower, unit_diagonal
        )
        _spsm_cache_put(_SPSM_PREPROCESS_CACHE, cache_key, solve_plan, _SPSM_PREPROCESS_CACHE_SIZE)
    return data, B, n_rows, n_cols, solve_plan


def _resolve_spsm_coo_runtime(data, row, col, B, shape, lower, unit_diagonal, opA, opB, major):
    data, row64, col64, B, n_rows, n_cols = _prepare_spsm_coo_inputs(
        data, row, col, B, shape, opA, opB, major
    )
    cache_key = _spsm_preprocess_cache_key(
        "coo",
        (data, row64, col64),
        shape,
        lower,
        unit_diagonal,
    )
    solve_plan = _spsm_cache_get(_SPSM_PREPROCESS_CACHE, cache_key)
    if solve_plan is None:
        solve_plan = _prepare_spsm_coo_system(
            data, row64, col64, n_rows, n_cols, lower, unit_diagonal
        )
        _spsm_cache_put(_SPSM_PREPROCESS_CACHE, cache_key, solve_plan, _SPSM_PREPROCESS_CACHE_SIZE)
    return data, B, n_rows, n_cols, solve_plan


def _run_spsm_csr_core(
    data,
    indices32,
    indptr,
    diag,
    rhs,
    n_rows,
    *,
    lower=True,
    unit_diagonal=False,
    block_nnz=None,
    max_segments=None,
    block_rhs=None,
    launch_groups=None,
    block_nnz_use=None,
    max_segments_use=None,
):
    if rhs.ndim != 2:
        raise ValueError("rhs must be 2D")
    rhs = rhs.contiguous()
    if rhs.shape[0] != n_rows:
        raise ValueError("rhs first dim must equal n_rows")
    n_rhs = int(rhs.shape[1])
    x = torch.zeros_like(rhs)
    if n_rows == 0 or n_rhs == 0:
        return x

    if launch_groups is None:
        levels = _build_spsm_levels(indptr, indices32, n_rows, lower=lower)
        launch_groups = levels
    if block_nnz_use is None or max_segments_use is None:
        block_nnz_use, max_segments_use = _auto_spsm_launch_config(
            indptr, block_nnz=block_nnz, max_segments=max_segments
        )
    block_rhs_use = _auto_rhs_block(n_rhs) if block_rhs is None else int(block_rhs)
    if block_rhs_use <= 0:
        raise ValueError("block_rhs must be positive")

    use_fp64 = data.dtype == torch.float64
    diag_eps = 1e-12 if use_fp64 else 1e-6

    for rows_lv in launch_groups:
        n_lv = rows_lv.numel()
        if n_lv == 0:
            continue
        grid = (n_lv, triton.cdiv(n_rhs, block_rhs_use))
        _spsm_csr_level_kernel_real[grid](
            data,
            indices32,
            indptr,
            diag,
            rhs,
            x,
            rows_lv,
            n_level_rows=n_lv,
            n_rhs=n_rhs,
            stride_b0=rhs.stride(0),
            stride_x0=x.stride(0),
            BLOCK_NNZ=block_nnz_use,
            BLOCK_RHS=block_rhs_use,
            MAX_SEGMENTS=max_segments_use,
            LOWER=lower,
            USE_FP64_ACC=use_fp64,
            DIAG_EPS=diag_eps,
        )
    return x


def _run_spsm_csr_tiled_core(
    data,
    indices32,
    dep_ptr,
    diag,
    rhs,
    n_rows,
    *,
    alpha=1.0,
    block_nnz=None,
    max_segments=None,
    block_rhs=None,
    block_nnz_use=None,
    max_segments_use=None,
):
    if rhs.ndim != 2:
        raise ValueError("rhs must be 2D")
    rhs = rhs.contiguous()
    if rhs.shape[0] != n_rows:
        raise ValueError("rhs first dim must equal n_rows")
    n_rhs = int(rhs.shape[1])
    y = _pack_spsm_rhs_work_buffer(rhs, alpha)
    if n_rows == 0 or n_rhs == 0:
        return y

    if block_nnz_use is None or max_segments_use is None:
        block_nnz_use, max_segments_use = _auto_spsm_launch_config_from_row_lengths(
            dep_ptr[1:] - dep_ptr[:-1],
            block_nnz=block_nnz,
            max_segments=max_segments,
        )
    block_rhs_use = _auto_rhs_block(n_rhs) if block_rhs is None else int(block_rhs)
    if block_rhs_use <= 0:
        raise ValueError("block_rhs must be positive")

    use_fp64 = data.dtype == torch.float64
    diag_eps = 1e-12 if use_fp64 else 1e-6
    rhs_tiles = triton.cdiv(n_rhs, block_rhs_use)
    ready = torch.zeros((rhs_tiles, n_rows), dtype=torch.int32, device=rhs.device)

    grid = (n_rows, rhs_tiles)
    _spsm_csr_tiled_kernel_real[grid](
        data,
        indices32,
        dep_ptr,
        diag,
        y,
        ready,
        n_rows,
        n_rhs,
        y.stride(0),
        ready.stride(0),
        BLOCK_NNZ=block_nnz_use,
        BLOCK_RHS=block_rhs_use,
        MAX_SEGMENTS=max_segments_use,
        USE_FP64_ACC=use_fp64,
        DIAG_EPS=diag_eps,
    )
    return y


def _run_spsm_coo_core(
    data,
    cols32,
    row_ptr,
    diag,
    rhs,
    n_rows,
    *,
    lower=True,
    unit_diagonal=False,
    block_nnz=None,
    max_segments=None,
    block_rhs=None,
    launch_groups=None,
    block_nnz_use=None,
    max_segments_use=None,
):
    if rhs.ndim != 2:
        raise ValueError("rhs must be 2D")
    rhs = rhs.contiguous()
    if rhs.shape[0] != n_rows:
        raise ValueError("rhs first dim must equal n_rows")
    n_rhs = int(rhs.shape[1])
    x = torch.zeros_like(rhs)
    if n_rows == 0 or n_rhs == 0:
        return x

    if launch_groups is None:
        levels = _build_spsm_levels(row_ptr, cols32, n_rows, lower=lower)
        launch_groups = levels
    if block_nnz_use is None or max_segments_use is None:
        block_nnz_use, max_segments_use = _auto_spsm_launch_config(
            row_ptr, block_nnz=block_nnz, max_segments=max_segments
        )
    block_rhs_use = _auto_rhs_block(n_rhs) if block_rhs is None else int(block_rhs)
    if block_rhs_use <= 0:
        raise ValueError("block_rhs must be positive")

    use_fp64 = data.dtype == torch.float64
    diag_eps = 1e-12 if use_fp64 else 1e-6

    for rows_lv in launch_groups:
        n_lv = rows_lv.numel()
        if n_lv == 0:
            continue
        grid = (n_lv, triton.cdiv(n_rhs, block_rhs_use))
        _spsm_coo_level_kernel_real[grid](
            data,
            row_ptr,
            cols32,
            diag,
            rhs,
            x,
            rows_lv,
            n_level_rows=n_lv,
            n_rhs=n_rhs,
            stride_b0=rhs.stride(0),
            stride_x0=x.stride(0),
            BLOCK_NNZ=block_nnz_use,
            BLOCK_RHS=block_rhs_use,
            MAX_SEGMENTS=max_segments_use,
            LOWER=lower,
            USE_FP64_ACC=use_fp64,
            DIAG_EPS=diag_eps,
        )
    return x




def flagsparse_spsm_csr(
    data,
    indices,
    indptr,
    B,
    shape,
    alpha=1.0,
    lower=True,
    unit_diagonal=False,
    opA="NON_TRANS",
    opB="NON_TRANS",
    major="row",
    out=None,
    return_time=False,
):
    data, B, n_rows, _n_cols, solve_plan = _resolve_spsm_csr_runtime(
        data, indices, indptr, B, shape, lower, unit_diagonal, opA, opB, major
    )
    use_tiled_kernel = _should_use_spsm_tiled_kernel("csr", B, data.dtype)
    alpha_value = 1.0 if _alpha_is_one(alpha) else _alpha_to_host_scalar(alpha)
    rhs = B if use_tiled_kernel else (B if alpha_value == 1.0 else torch.as_tensor(alpha, dtype=B.dtype, device=B.device) * B)
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    if use_tiled_kernel:
        x = _run_spsm_csr_tiled_core(
            solve_plan["kernel_dep_data"],
            solve_plan["kernel_dep_indices32"],
            solve_plan["kernel_dep_ptr"],
            solve_plan["kernel_diag"],
            rhs,
            n_rows,
            alpha=alpha_value,
            block_nnz_use=solve_plan["tiled_block_nnz"],
            max_segments_use=solve_plan["tiled_max_segments"],
        )
    else:
        x = _run_spsm_csr_core(
            solve_plan["kernel_data"],
            solve_plan["kernel_indices32"],
            solve_plan["kernel_indptr"],
            solve_plan["kernel_diag"],
            rhs,
            n_rows,
            lower=solve_plan["lower_eff"],
            unit_diagonal=unit_diagonal,
            launch_groups=solve_plan["launch_groups"],
            block_nnz_use=solve_plan["default_block_nnz"],
            max_segments_use=solve_plan["default_max_segments"],
        )
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


def flagsparse_spsm_coo(
    data,
    row,
    col,
    B,
    shape,
    alpha=1.0,
    lower=True,
    unit_diagonal=False,
    opA="NON_TRANS",
    opB="NON_TRANS",
    major="row",
    out=None,
    return_time=False,
):
    data, B, n_rows, _n_cols, solve_plan = _resolve_spsm_coo_runtime(
        data, row, col, B, shape, lower, unit_diagonal, opA, opB, major
    )
    use_tiled_kernel = _should_use_spsm_tiled_kernel("coo", B, data.dtype)
    alpha_value = 1.0 if _alpha_is_one(alpha) else _alpha_to_host_scalar(alpha)
    rhs = B if use_tiled_kernel else (B if alpha_value == 1.0 else torch.as_tensor(alpha, dtype=B.dtype, device=B.device) * B)
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    if use_tiled_kernel:
        x = _run_spsm_csr_tiled_core(
            solve_plan["kernel_dep_data"],
            solve_plan["kernel_dep_indices32"],
            solve_plan["kernel_dep_ptr"],
            solve_plan["kernel_diag"],
            rhs,
            n_rows,
            alpha=alpha_value,
            block_nnz_use=solve_plan["tiled_block_nnz"],
            max_segments_use=solve_plan["tiled_max_segments"],
        )
    else:
        x = _run_spsm_coo_core(
            solve_plan["kernel_data"],
            solve_plan["kernel_cols32"],
            solve_plan["kernel_row_ptr"],
            solve_plan["kernel_diag"],
            rhs,
            n_rows,
            lower=solve_plan["lower_eff"],
            unit_diagonal=unit_diagonal,
            launch_groups=solve_plan["launch_groups"],
            block_nnz_use=solve_plan["default_block_nnz"],
            max_segments_use=solve_plan["default_max_segments"],
        )
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


def _analyze_spsm_csr(
    data,
    indices,
    indptr,
    B,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    opA="NON_TRANS",
    opB="NON_TRANS",
    major="row",
    clear_cache=False,
    return_time=False,
):
    if clear_cache:
        _clear_spsm_preprocess_cache()
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    _resolve_spsm_csr_runtime(data, indices, indptr, B, shape, lower, unit_diagonal, opA, opB, major)
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms
    return None


def _analyze_spsm_coo(
    data,
    row,
    col,
    B,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    opA="NON_TRANS",
    opB="NON_TRANS",
    major="row",
    clear_cache=False,
    return_time=False,
):
    if clear_cache:
        _clear_spsm_preprocess_cache()
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    _resolve_spsm_coo_runtime(data, row, col, B, shape, lower, unit_diagonal, opA, opB, major)
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms
    return None


def benchmark_spsm_case(
    fmt="csr",
    n_rows=1024,
    n_rhs=32,
    nnz=8192,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    alpha=1.0,
    lower=True,
    unit_diagonal=False,
    warmup=10,
    iters=50,
):
    """Pure FlagSparse SpSM benchmark entry for one configuration."""
    device = torch.device("cuda")
    data, indices, indptr = _build_random_csr(
        n_rows, n_rows, nnz, value_dtype, index_dtype, device
    )
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, device=device, dtype=torch.int64),
        indptr.to(torch.int64)[1:] - indptr.to(torch.int64)[:-1],
    )
    col_ids = indices.to(torch.int64)
    tri_mask = (col_ids <= row_ids) if lower else (col_ids >= row_ids)
    data = data[tri_mask]
    row_ids = row_ids[tri_mask]
    col_ids = col_ids[tri_mask]
    data, col_ids, indptr = _coo_to_csr_sorted_unique(data, row_ids, col_ids, n_rows, n_rows)
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, device=device, dtype=torch.int64),
        indptr[1:] - indptr[:-1],
    )
    diag_mask = col_ids == row_ids
    diag_present = torch.zeros(n_rows, dtype=torch.bool, device=device)
    if diag_mask.numel() > 0 and bool(torch.any(diag_mask).item()):
        diag_present[row_ids[diag_mask]] = True
    missing_diag = torch.nonzero(~diag_present, as_tuple=False).reshape(-1).to(torch.int64)
    if missing_diag.numel() > 0:
        diag_values = torch.ones(missing_diag.numel(), dtype=value_dtype, device=device)
        data = torch.cat([data, diag_values], dim=0)
        row_ids = torch.cat([row_ids, missing_diag], dim=0)
        col_ids = torch.cat([col_ids, missing_diag], dim=0)
        data, col_ids, indptr = _coo_to_csr_sorted_unique(data, row_ids, col_ids, n_rows, n_rows)
    B = torch.randn((n_rows, n_rhs), dtype=value_dtype, device=device).contiguous()
    shape = (n_rows, n_rows)

    if str(fmt).lower() == "coo":
        row_ids = torch.repeat_interleave(
            torch.arange(n_rows, device=device, dtype=torch.int64),
            indptr[1:] - indptr[:-1],
        )
        analysis_ms = _analyze_spsm_coo(
            data,
            row_ids,
            col_ids,
            B,
            shape,
            lower=lower,
            unit_diagonal=unit_diagonal,
            clear_cache=True,
            return_time=True,
        )
        solve_call = lambda: flagsparse_spsm_coo(
            data,
            row_ids,
            col_ids,
            B,
            shape,
            alpha=alpha,
            lower=lower,
            unit_diagonal=unit_diagonal,
            opA="NON_TRANS",
            opB="NON_TRANS",
            major="row",
        )
    else:
        analysis_ms = _analyze_spsm_csr(
            data,
            col_ids.to(index_dtype),
            indptr,
            B,
            shape,
            lower=lower,
            unit_diagonal=unit_diagonal,
            clear_cache=True,
            return_time=True,
        )
        solve_call = lambda: flagsparse_spsm_csr(
            data,
            col_ids.to(index_dtype),
            indptr,
            B,
            shape,
            alpha=alpha,
            lower=lower,
            unit_diagonal=unit_diagonal,
            opA="NON_TRANS",
            opB="NON_TRANS",
            major="row",
        )
    C_fs, solve_ms = _benchmark_cuda_op(solve_call, warmup=warmup, iters=iters)
    return {
        "parameters": {
            "format": str(fmt).lower(),
            "n_rows": n_rows,
            "n_rhs": n_rhs,
            "nnz": int(data.numel()),
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
            "opA": "NON_TRANS",
            "opB": "NON_TRANS",
            "major": "row",
        },
        "performance": {
            "triton_analysis_ms": analysis_ms,
            "triton_solve_ms": solve_ms,
            "triton_time_total_ms": (
                analysis_ms + solve_ms if analysis_ms is not None and solve_ms is not None else None
            ),
        },
        "samples": {"flagsparse": C_fs},
    }
