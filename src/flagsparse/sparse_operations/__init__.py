"""FlagSparse sparse operations (gather, scatter, SpMV)."""

import time

try:
    import torch
    import triton
    import triton.language as tl
except ImportError as exc:
    raise ImportError(
        "Runtime dependencies are missing. Install them manually: pip install torch triton"
    ) from exc

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
except ImportError:
    cp = None
    cpx_sparse = None


SUPPORTED_VALUE_DTYPES = (
    torch.float32,
    torch.float64,
)
SUPPORTED_INDEX_DTYPES = (torch.int32,)
_INDEX_LIMIT_INT32 = 2**31 - 1


@triton.jit
def _gather_real_kernel(
    sparse_values_ptr,
    dense_values_ptr,
    indices_ptr,
    nnz,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < nnz
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    gathered_values = tl.load(dense_values_ptr + indices, mask=mask, other=0.0)
    tl.store(sparse_values_ptr + offsets, gathered_values, mask=mask)


@triton.jit
def _gather_complex_kernel(
    sparse_values_ri_ptr,
    dense_values_ri_ptr,
    indices_ptr,
    nnz,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < nnz
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)

    dense_offsets = indices * 2
    sparse_offsets = offsets * 2

    gathered_real = tl.load(dense_values_ri_ptr + dense_offsets, mask=mask, other=0.0)
    gathered_imag = tl.load(dense_values_ri_ptr + dense_offsets + 1, mask=mask, other=0.0)

    tl.store(sparse_values_ri_ptr + sparse_offsets, gathered_real, mask=mask)
    tl.store(sparse_values_ri_ptr + sparse_offsets + 1, gathered_imag, mask=mask)


@triton.jit
def _scatter_real_kernel(
    dense_values_ptr,
    sparse_values_ptr,
    indices_ptr,
    nnz,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < nnz

    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    values = tl.load(sparse_values_ptr + offsets, mask=mask, other=0.0)
    tl.store(dense_values_ptr + indices, values, mask=mask)


@triton.jit
def _scatter_complex_kernel(
    dense_values_ri_ptr,
    sparse_values_ri_ptr,
    indices_ptr,
    nnz,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < nnz

    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    dense_offsets = indices * 2
    sparse_offsets = offsets * 2

    values_real = tl.load(sparse_values_ri_ptr + sparse_offsets, mask=mask, other=0.0)
    values_imag = tl.load(sparse_values_ri_ptr + sparse_offsets + 1, mask=mask, other=0.0)

    tl.store(dense_values_ri_ptr + dense_offsets, values_real, mask=mask)
    tl.store(dense_values_ri_ptr + dense_offsets + 1, values_imag, mask=mask)


@triton.jit
def _spmv_csr_real_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    acc = tl.load(data_ptr + start, mask=start < end, other=0.0) * 0
    for seg in range(MAX_SEGMENTS):
        idx = start + seg * BLOCK_NNZ
        offsets = idx + tl.arange(0, BLOCK_NNZ)
        mask = offsets < end
        a = tl.load(data_ptr + offsets, mask=mask, other=0.0)
        col = tl.load(indices_ptr + offsets, mask=mask, other=0)
        x_vals = tl.load(x_ptr + col, mask=mask, other=0.0)
        part = tl.where(mask, a * x_vals, 0.0)
        acc = acc + tl.sum(part)
    tl.store(y_ptr + row, acc)


@triton.jit
def _spmv_csr_complex_kernel(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    x_ri_ptr,
    y_ri_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    acc_re = tl.load(data_ri_ptr + start * 2, mask=start < end, other=0.0) * 0
    acc_im = tl.load(data_ri_ptr + start * 2 + 1, mask=start < end, other=0.0) * 0
    for seg in range(MAX_SEGMENTS):
        idx = start + seg * BLOCK_NNZ
        offsets = idx + tl.arange(0, BLOCK_NNZ)
        mask = offsets < end
        a_re = tl.load(data_ri_ptr + offsets * 2, mask=mask, other=0.0)
        a_im = tl.load(data_ri_ptr + offsets * 2 + 1, mask=mask, other=0.0)
        col = tl.load(indices_ptr + offsets, mask=mask, other=0)
        x_re = tl.load(x_ri_ptr + col * 2, mask=mask, other=0.0)
        x_im = tl.load(x_ri_ptr + col * 2 + 1, mask=mask, other=0.0)
        prod_re = tl.where(mask, a_re * x_re - a_im * x_im, 0.0)
        prod_im = tl.where(mask, a_re * x_im + a_im * x_re, 0.0)
        acc_re = acc_re + tl.sum(prod_re)
        acc_im = acc_im + tl.sum(prod_im)
    tl.store(y_ri_ptr + row * 2, acc_re)
    tl.store(y_ri_ptr + row * 2 + 1, acc_im)


def _is_complex_dtype(value_dtype):
    return value_dtype in (torch.complex64, torch.complex128)


def _component_dtype_for_complex(value_dtype):
    if value_dtype == torch.complex64:
        return torch.float32
    if value_dtype == torch.complex128:
        return torch.float64
    raise TypeError(f"Unsupported complex dtype: {value_dtype}")


def _tolerance_for_dtype(value_dtype):
    if value_dtype == torch.float16:
        return 2e-3, 2e-3
    if value_dtype == torch.bfloat16:
        return 1e-1, 1e-1
    if value_dtype in (torch.float32, torch.complex64):
        return 1e-6, 1e-5
    if value_dtype in (torch.float64, torch.complex128):
        return 1e-10, 1e-8
    return 1e-6, 1e-5


def _require_cupy():
    if cp is None or cpx_sparse is None:
        raise RuntimeError(
            "CuPy is required for cuSPARSE baseline. "
            "Install a CUDA-matched wheel, for example: pip install cupy-cuda12x"
        )


def _cupy_dtype_from_torch(torch_dtype):
    _require_cupy()
    mapping = {
        torch.float16: cp.float16,
        # Keep cuSPARSE baseline stable for bf16 by computing in fp32 on CuPy path.
        torch.bfloat16: cp.float32,
        torch.float32: cp.float32,
        torch.float64: cp.float64,
        torch.complex64: cp.complex64,
        torch.complex128: cp.complex128,
        torch.int32: cp.int32,
        torch.int64: cp.int64,
    }
    if torch_dtype not in mapping:
        raise TypeError(f"Unsupported dtype conversion to CuPy: {torch_dtype}")
    return mapping[torch_dtype]


def _cupy_from_torch(tensor):
    _require_cupy()
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))


def _torch_from_cupy(array):
    try:
        dlpack_capsule = array.toDlpack()
    except AttributeError:
        dlpack_capsule = array.to_dlpack()
    return torch.utils.dlpack.from_dlpack(dlpack_capsule)


def _to_torch_tensor(x, name):
    if torch.is_tensor(x):
        return x, "torch"
    if cp is not None and isinstance(x, cp.ndarray):
        return _torch_from_cupy(x), "cupy"
    raise TypeError(f"{name} must be a torch.Tensor or cupy.ndarray")


def _to_backend_like(torch_tensor, ref_obj):
    if cp is not None and isinstance(ref_obj, cp.ndarray):
        return _cupy_from_torch(torch_tensor)
    return torch_tensor


def _cusparse_baseline_skip_reason(value_dtype):
    if value_dtype == torch.bfloat16:
        return "bfloat16 is not supported by the cuSPARSE baseline path; skipped"
    if cp is None and value_dtype == torch.float16:
        return "float16 is not supported by torch sparse fallback when CuPy is unavailable; skipped"
    return None


def _build_random_dense(dense_size, value_dtype, device):
    if value_dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return torch.randn(dense_size, dtype=value_dtype, device=device)
    if _is_complex_dtype(value_dtype):
        component_dtype = _component_dtype_for_complex(value_dtype)
        real = torch.randn(dense_size, dtype=component_dtype, device=device)
        imag = torch.randn(dense_size, dtype=component_dtype, device=device)
        return torch.complex(real, imag)
    raise TypeError(f"Unsupported value dtype: {value_dtype}")


def _build_indices(nnz, dense_size, index_dtype, device, unique=False):
    if unique and nnz <= dense_size:
        return torch.randperm(dense_size, device=device)[:nnz].to(index_dtype)
    return torch.randint(0, dense_size, (nnz,), dtype=index_dtype, device=device)


def _build_random_csr(n_rows, n_cols, nnz, value_dtype, index_dtype, device):
    if nnz <= 0 or n_rows <= 0 or n_cols <= 0:
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=device)
        return (
            torch.empty(0, dtype=value_dtype, device=device),
            torch.empty(0, dtype=index_dtype, device=device),
            indptr,
        )
    row_choices = torch.randint(0, n_rows, (nnz,), device=device)
    row_choices, _ = torch.sort(row_choices)
    nnz_per_row = torch.bincount(row_choices, minlength=n_rows)
    indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=device)
    indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    indices = torch.randint(0, n_cols, (nnz,), dtype=index_dtype, device=device)
    data = _build_random_dense(nnz, value_dtype, device)
    return data, indices, indptr


def _validate_common_inputs(dense_vector, indices):
    if dense_vector.ndim != 1:
        raise ValueError("dense_vector must be a 1D tensor")
    if indices.ndim != 1:
        raise ValueError("indices must be a 1D tensor")
    if not dense_vector.is_cuda or not indices.is_cuda:
        raise ValueError("dense_vector and indices must both be CUDA tensors")
    if dense_vector.dtype not in SUPPORTED_VALUE_DTYPES:
        raise TypeError(
            "dense_vector dtype must be one of: torch.float16, torch.bfloat16, "
            "torch.float32, torch.float64, torch.complex64, torch.complex128"
        )
    if indices.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")


def _prepare_inputs(dense_vector, indices):
    _validate_common_inputs(dense_vector, indices)

    dense_vector = dense_vector.contiguous()
    indices = indices.contiguous()

    max_index = -1
    if indices.numel() > 0:
        if torch.any(indices < 0).item():
            raise IndexError("indices must be non-negative")
        max_index = int(indices.max().item())
        if max_index >= dense_vector.numel():
            raise IndexError(
                f"indices out of range: max index {max_index}, dense size {dense_vector.numel()}"
            )

    kernel_indices = indices
    if indices.dtype == torch.int64:
        if max_index > _INDEX_LIMIT_INT32:
            raise ValueError(
                f"int64 index value {max_index} exceeds Triton int32 kernel range"
            )
        kernel_indices = indices.to(torch.int32)

    return dense_vector, indices, kernel_indices


def _prepare_scatter_inputs(sparse_values, indices, dense_size=None, out=None):
    if sparse_values.ndim != 1:
        raise ValueError("sparse_values must be a 1D tensor")
    if indices.ndim != 1:
        raise ValueError("indices must be a 1D tensor")
    if sparse_values.numel() != indices.numel():
        raise ValueError("sparse_values and indices must have the same number of elements")
    if not sparse_values.is_cuda or not indices.is_cuda:
        raise ValueError("sparse_values and indices must both be CUDA tensors")
    if sparse_values.dtype not in SUPPORTED_VALUE_DTYPES:
        raise TypeError(
            "sparse_values dtype must be one of: torch.float16, torch.bfloat16, "
            "torch.float32, torch.float64, torch.complex64, torch.complex128"
        )
    if indices.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")

    sparse_values = sparse_values.contiguous()
    indices = indices.contiguous()

    if dense_size is None:
        dense_size = int(indices.max().item()) + 1 if indices.numel() > 0 else 0
    dense_size = int(dense_size)
    if dense_size < 0:
        raise ValueError("dense_size must be non-negative")

    max_index = -1
    if indices.numel() > 0:
        if torch.any(indices < 0).item():
            raise IndexError("indices must be non-negative")
        max_index = int(indices.max().item())
        if max_index >= dense_size:
            raise IndexError(
                f"indices out of range: max index {max_index}, dense size {dense_size}"
            )

    kernel_indices = indices
    if indices.dtype == torch.int64:
        if max_index > _INDEX_LIMIT_INT32:
            raise ValueError(
                f"int64 index value {max_index} exceeds Triton int32 kernel range"
            )
        kernel_indices = indices.to(torch.int32)

    if out is not None:
        if out.ndim != 1:
            raise ValueError("out must be a 1D tensor")
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor")
        if out.dtype != sparse_values.dtype:
            raise TypeError("out dtype must match sparse_values dtype")
        if out.numel() != dense_size:
            raise ValueError("out size must equal dense_size")
        if out.device != sparse_values.device:
            raise ValueError("out must be on the same device as sparse_values")

    return sparse_values, indices, kernel_indices, dense_size


def _triton_gather_impl(dense_vector, kernel_indices, block_size=1024):
    nnz = kernel_indices.numel()
    if nnz == 0:
        return torch.empty(0, dtype=dense_vector.dtype, device=dense_vector.device)

    grid = lambda meta: (triton.cdiv(nnz, meta["BLOCK_SIZE"]),)

    if not _is_complex_dtype(dense_vector.dtype):
        sparse_values = torch.empty(nnz, dtype=dense_vector.dtype, device=dense_vector.device)
        _gather_real_kernel[grid](
            sparse_values,
            dense_vector,
            kernel_indices,
            nnz,
            BLOCK_SIZE=block_size,
        )
        return sparse_values

    sparse_values = torch.empty(nnz, dtype=dense_vector.dtype, device=dense_vector.device)
    dense_values_ri = torch.view_as_real(dense_vector).reshape(-1)
    sparse_values_ri = torch.view_as_real(sparse_values).reshape(-1)

    _gather_complex_kernel[grid](
        sparse_values_ri,
        dense_values_ri,
        kernel_indices,
        nnz,
        BLOCK_SIZE=block_size,
    )
    return sparse_values


def _triton_scatter_impl(sparse_values, kernel_indices, dense_size, out=None, block_size=1024):
    if out is None:
        dense_values = torch.zeros(
            dense_size, dtype=sparse_values.dtype, device=sparse_values.device
        )
    else:
        dense_values = out
        dense_values.zero_()

    nnz = kernel_indices.numel()
    if nnz == 0:
        return dense_values

    grid = lambda meta: (triton.cdiv(nnz, meta["BLOCK_SIZE"]),)

    if not _is_complex_dtype(sparse_values.dtype):
        _scatter_real_kernel[grid](
            dense_values,
            sparse_values,
            kernel_indices,
            nnz,
            BLOCK_SIZE=block_size,
        )
        return dense_values

    dense_values_ri = torch.view_as_real(dense_values).reshape(-1)
    sparse_values_ri = torch.view_as_real(sparse_values).reshape(-1)
    _scatter_complex_kernel[grid](
        dense_values_ri,
        sparse_values_ri,
        kernel_indices,
        nnz,
        BLOCK_SIZE=block_size,
    )
    return dense_values


def _prepare_spmv_csr_inputs(data, indices, indptr, x, shape):
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1 or x.ndim != 1:
        raise ValueError("data, indices, indptr, and x must be 1D tensors")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if indptr.numel() != n_rows + 1:
        raise ValueError(
            f"indptr length must be n_rows+1={n_rows + 1}, got {indptr.numel()}"
        )
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")
    if x.numel() != n_cols:
        raise ValueError(f"x length must be n_cols={n_cols}, got {x.numel()}")
    if not all(t.is_cuda for t in (data, indices, indptr, x)):
        raise ValueError("data, indices, indptr, and x must be CUDA tensors")
    if data.dtype not in SUPPORTED_VALUE_DTYPES:
        raise TypeError(
            "data dtype must be one of: float16, bfloat16, float32, float64, complex64, complex128"
        )
    if x.dtype != data.dtype:
        raise TypeError("x dtype must match data dtype")
    if indices.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")
    nnz = data.numel()
    if nnz > 0 and (int(indices.min().item()) < 0 or int(indices.max().item()) >= n_cols):
        raise IndexError("indices out of range for n_cols")
    data = data.contiguous()
    indices = indices.contiguous()
    indptr = indptr.contiguous()
    x = x.contiguous()
    kernel_indices = indices.to(torch.int32) if indices.dtype == torch.int64 else indices
    kernel_indptr = indptr.to(torch.int64)
    return data, kernel_indices, kernel_indptr, x, n_rows, n_cols


def _triton_spmv_csr_impl(
    data, indices, indptr, x, n_rows, block_nnz=256, max_segments=64
):
    device = data.device
    dtype = data.dtype
    y = torch.empty(n_rows, dtype=dtype, device=device)
    if n_rows == 0:
        return y
    compute_dtype = dtype
    data_in = data
    x_in = x
    if dtype == torch.float16 or dtype == torch.bfloat16:
        compute_dtype = torch.float32
        data_in = data.to(torch.float32)
        x_in = x.to(torch.float32)
    if not _is_complex_dtype(compute_dtype):
        y_out = torch.empty(n_rows, dtype=compute_dtype, device=device)
        grid = (n_rows,)
        _spmv_csr_real_kernel[grid](
            data_in,
            indices,
            indptr,
            x_in,
            y_out,
            n_rows=n_rows,
            BLOCK_NNZ=block_nnz,
            MAX_SEGMENTS=max_segments,
        )
        if dtype == torch.float16 or dtype == torch.bfloat16:
            y_out = y_out.to(dtype)
        y.copy_(y_out)
        return y
    data_ri = torch.view_as_real(data_in).reshape(-1)
    x_ri = torch.view_as_real(x_in).reshape(-1)
    comp_dtype = data_ri.dtype
    y_ri = torch.empty(n_rows * 2, dtype=comp_dtype, device=device)
    grid = (n_rows,)
    _spmv_csr_complex_kernel[grid](
        data_ri,
        indices,
        indptr,
        x_ri,
        y_ri,
        n_rows=n_rows,
        BLOCK_NNZ=block_nnz,
        MAX_SEGMENTS=max_segments,
    )
    y_ri = y_ri.reshape(n_rows, 2)
    y.copy_(torch.view_as_complex(y_ri))
    return y


def flagsparse_spmv_csr(
    data,
    indices,
    indptr,
    x,
    shape,
    block_nnz=256,
    max_segments=64,
    out=None,
    return_time=False,
):
    """
    CSR SpMV: y = A @ x using Triton (cuSPARSE-aligned dtypes).
    data, indices, indptr: CSR arrays; x: dense vector; shape: (n_rows, n_cols).
    """
    data, kernel_indices, kernel_indptr, x, n_rows, n_cols = _prepare_spmv_csr_inputs(
        data, indices, indptr, x, shape
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    y = _triton_spmv_csr_impl(
        data,
        kernel_indices,
        kernel_indptr,
        x,
        n_rows,
        block_nnz=block_nnz,
        max_segments=max_segments,
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if out is not None:
        if out.shape != y.shape or out.dtype != y.dtype:
            raise ValueError("out shape/dtype must match result")
        out.copy_(y)
        y = out
    if return_time:
        return y, elapsed_ms
    return y


def flagsparse_spmv_coo(
    data,
    row,
    col,
    x,
    shape,
    block_nnz=256,
    max_segments=64,
    out=None,
    return_time=False,
):
    """COO SpMV: y = A @ x by converting COO (row, col, data) to CSR and calling flagsparse_spmv_csr.

    Only float32/float64 values and int32 indices are supported (aligned with SUPPORTED_VALUE_DTYPES/INDEX_DTYPES).
    """
    if not all(torch.is_tensor(t) for t in (data, row, col, x)):
        raise TypeError("data, row, col, x must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, row, col, x)):
        raise ValueError("data, row, col, x must all be CUDA tensors")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1 or x.ndim != 1:
        raise ValueError("data, row, col, x must all be 1D tensors")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if data.dtype not in SUPPORTED_VALUE_DTYPES:
        raise TypeError("data dtype must be float32 or float64")
    if x.dtype != data.dtype:
        raise TypeError("x dtype must match data dtype")

    # Sort by (row, col).
    row64 = row.to(torch.int64)
    col64 = col.to(torch.int64)
    if row64.numel() > 0:
        key = row64 * max(1, n_cols) + col64
        order = torch.argsort(key)
        row_s = row64[order]
        col_s = col64[order]
        data_s = data[order].to(data.dtype)
    else:
        row_s = row64
        col_s = col64
        data_s = data

    nnz = data_s.numel()
    indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device)
    if nnz > 0:
        nnz_per_row = torch.bincount(row_s, minlength=n_rows)
        indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    indices = col_s.to(torch.int32)

    return flagsparse_spmv_csr(
        data_s,
        indices,
        indptr,
        x,
        shape,
        block_nnz=block_nnz,
        max_segments=max_segments,
        out=out,
        return_time=return_time,
    )


def _csr_to_dense(data, indices, indptr, shape):
    """Convert CSR (torch CUDA tensors) to dense matrix on the same device."""
    device = data.device
    dtype = data.dtype
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows == 0 or n_cols == 0:
        return torch.zeros((n_rows, n_cols), dtype=dtype, device=device)
    row_ind = torch.repeat_interleave(
        torch.arange(n_rows, device=device, dtype=torch.int64),
        indptr[1:] - indptr[:-1],
    )
    col_ind = indices.to(torch.int64)
    coo = torch.sparse_coo_tensor(
        torch.stack([row_ind, col_ind]),
        data,
        (n_rows, n_cols),
        device=device,
    ).coalesce()
    return coo.to_dense()


def _prepare_spsv_inputs(data, indices, indptr, b, shape):
    """Validate and normalize inputs for sparse solve A x = b with CSR A."""
    if not all(torch.is_tensor(t) for t in (data, indices, indptr, b)):
        raise TypeError("data, indices, indptr, b must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, indices, indptr, b)):
        raise ValueError("data, indices, indptr, b must all be CUDA tensors")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, indptr must be 1D")
    if b.ndim not in (1, 2):
        raise ValueError("b must be 1D or 2D (vector or multiple RHS)")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if indptr.numel() != n_rows + 1:
        raise ValueError(f"indptr length must be n_rows+1={n_rows + 1}")
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")
    if b.ndim == 1 and b.numel() != n_rows:
        raise ValueError(f"b length must equal n_rows={n_rows}")
    if b.ndim == 2 and b.shape[0] != n_rows:
        raise ValueError(f"b.shape[0] must equal n_rows={n_rows}")

    if data.dtype not in SUPPORTED_VALUE_DTYPES:
        raise TypeError(
            "data dtype must be one of: float16, bfloat16, float32, float64, complex64, complex128"
        )
    if indices.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")
    if b.dtype != data.dtype:
        raise TypeError("b dtype must match data dtype")

    return (
        data.contiguous(),
        indices.contiguous(),
        indptr.contiguous(),
        b.contiguous(),
        n_rows,
        n_cols,
    )


def flagsparse_spsv_csr(
    data,
    indices,
    indptr,
    b,
    shape,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    out=None,
    return_time=False,
):
    """Sparse solve: solve A x = b where A is CSR (triangular or general).

    Current implementation is functionality-first:
    - Convert CSR to dense on GPU
    - Use torch.linalg.solve (with optional triangular masking)
    - Support all value/index dtypes aligned with cuSPARSE SpMV.
    """
    data, indices, indptr, b, n_rows, n_cols = _prepare_spsv_inputs(
        data, indices, indptr, b, shape
    )
    if n_rows != n_cols:
        raise ValueError(f"A must be square, got shape={shape}")

    dense = _csr_to_dense(data, indices, indptr, (n_rows, n_cols))
    cast_back = dense.dtype in (torch.float16, torch.bfloat16)

    if cast_back:
        dense_ = dense.to(torch.float32)
        b_ = b.to(torch.float32)
    else:
        dense_ = dense
        b_ = b

    # Apply lower/upper/transpose masks for triangular systems.
    if lower and not transpose:
        tri = torch.tril(dense_)
    elif (not lower) and not transpose:
        tri = torch.triu(dense_)
    elif lower and transpose:
        tri = torch.triu(dense_.mT)
    else:
        tri = torch.tril(dense_.mT)

    if unit_diagonal:
        # Force diagonal to 1 without changing RHS; this matches unit-diagonal assumption.
        tri = tri.clone()
        tri.diagonal().fill_(1.0)

    if b_.ndim == 1:
        b_mat = b_.unsqueeze(1)
    else:
        b_mat = b_

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    x_mat = torch.linalg.solve(tri, b_mat)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if cast_back:
        x_mat = x_mat.to(dense.dtype)

    x = x_mat.squeeze(1) if b.ndim == 1 else x_mat
    if out is not None:
        if out.shape != x.shape or out.dtype != x.dtype:
            raise ValueError("out shape/dtype must match result")
        out.copy_(x)
        x = out

    if return_time:
        return x, elapsed_ms
    return x


def flagsparse_spsv_coo(
    data,
    row,
    col,
    b,
    shape,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    **kwargs,
):
    """COO variant: convert COO (data, row, col) to CSR then call flagsparse_spsv_csr."""
    if not all(torch.is_tensor(t) for t in (data, row, col, b)):
        raise TypeError("data, row, col, b must all be torch.Tensor")
    device = data.device
    dtype = data.dtype
    n_rows, n_cols = int(shape[0]), int(shape[1])

    # Ensure sorted by row, then col (PyTorch has no lexsort API).
    row64 = row.to(torch.int64)
    col64 = col.to(torch.int64)
    key = row64 * max(1, n_cols) + col64
    order = torch.argsort(key)
    row_s = row64[order]
    col_s = col64[order]
    data_s = data[order].to(dtype)

    nnz = data_s.numel()
    indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=device)
    if nnz > 0:
        nnz_per_row = torch.bincount(row_s, minlength=n_rows)
        indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    indices = col_s.to(torch.int64)

    return flagsparse_spsv_csr(
        data_s,
        indices,
        indptr,
        b,
        shape,
        lower=lower,
        unit_diagonal=unit_diagonal,
        transpose=transpose,
        **kwargs,
    )


def _benchmark_cuda_op(op, warmup, iters):
    warmup = max(0, int(warmup))
    iters = max(1, int(iters))

    output = None
    for _ in range(warmup):
        output = op()

    torch.cuda.synchronize()
    if cp is not None:
        cp.cuda.runtime.deviceSynchronize()
    start_time = time.perf_counter()
    for _ in range(iters):
        output = op()
    torch.cuda.synchronize()
    if cp is not None:
        cp.cuda.runtime.deviceSynchronize()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / iters
    return output, elapsed_ms


def _cusparse_spmv(selector_matrix, dense_vector):
    if cp is not None and cpx_sparse is not None and isinstance(selector_matrix, cpx_sparse.spmatrix):
        if torch.is_tensor(dense_vector):
            out_dtype = dense_vector.dtype
            dense_for_compute = (
                dense_vector.to(torch.float32)
                if dense_vector.dtype == torch.bfloat16
                else dense_vector
            )
            dense_cp = _cupy_from_torch(dense_for_compute)
            out_cp = selector_matrix @ dense_cp
            out_torch = _torch_from_cupy(out_cp)
            if out_dtype == torch.bfloat16:
                out_torch = out_torch.to(torch.bfloat16)
            return out_torch

        if cp is not None and isinstance(dense_vector, cp.ndarray):
            return selector_matrix @ dense_vector

        raise TypeError("dense_vector must be torch.Tensor or cupy.ndarray")

    # Fallback path: torch sparse SpMV (still CUDA-backed).
    if torch.is_tensor(selector_matrix) and selector_matrix.is_sparse:
        if not torch.is_tensor(dense_vector):
            raise TypeError("dense_vector must be torch.Tensor for torch sparse fallback")
        out_dtype = dense_vector.dtype
        dense_for_compute = (
            dense_vector.to(torch.float32)
            if dense_vector.dtype == torch.bfloat16
            else dense_vector
        )
        out = torch.sparse.mm(selector_matrix, dense_for_compute.unsqueeze(1)).squeeze(1)
        if out_dtype == torch.bfloat16:
            out = out.to(torch.bfloat16)
        return out

    if cp is None or cpx_sparse is None:
        raise RuntimeError(
            "CuPy is not available and torch sparse fallback selector is not provided"
        )
    raise TypeError(
        "selector_matrix must be a cupyx sparse matrix or torch sparse tensor"
    )


def _make_gather_selector_matrix(indices, dense_size, value_dtype):
    if cp is not None and cpx_sparse is not None:
        rows_cp = cp.arange(indices.numel(), dtype=cp.int64)
        cols_cp = _cupy_from_torch(indices.to(torch.int64))
        vals_cp = cp.ones(indices.numel(), dtype=_cupy_dtype_from_torch(value_dtype))
        return cpx_sparse.coo_matrix(
            (vals_cp, (rows_cp, cols_cp)),
            shape=(indices.numel(), dense_size),
        )

    rows = torch.arange(indices.numel(), dtype=torch.int64, device=indices.device)
    cols = indices.to(torch.int64)
    coords = torch.stack([rows, cols], dim=0)
    values = torch.ones(indices.numel(), dtype=value_dtype, device=indices.device)
    return torch.sparse_coo_tensor(
        coords, values, size=(indices.numel(), dense_size), device=indices.device
    ).coalesce()


def _make_scatter_selector_matrix(indices, dense_size, value_dtype):
    if cp is not None and cpx_sparse is not None:
        rows_cp = _cupy_from_torch(indices.to(torch.int64))
        cols_cp = cp.arange(indices.numel(), dtype=cp.int64)
        vals_cp = cp.ones(indices.numel(), dtype=_cupy_dtype_from_torch(value_dtype))
        return cpx_sparse.coo_matrix(
            (vals_cp, (rows_cp, cols_cp)),
            shape=(dense_size, indices.numel()),
        )

    rows = indices.to(torch.int64)
    cols = torch.arange(indices.numel(), dtype=torch.int64, device=indices.device)
    coords = torch.stack([rows, cols], dim=0)
    values = torch.ones(indices.numel(), dtype=value_dtype, device=indices.device)
    return torch.sparse_coo_tensor(
        coords, values, size=(dense_size, indices.numel()), device=indices.device
    ).coalesce()


def _pytorch_scatter_impl(sparse_values, indices, dense_size, out=None):
    if out is None:
        dense_values = torch.zeros(
            dense_size, dtype=sparse_values.dtype, device=sparse_values.device
        )
    else:
        dense_values = out
        dense_values.zero_()
    dense_values.index_copy_(0, indices.to(torch.int64), sparse_values)
    return dense_values


def flagsparse_gather(a, indices, out=None, mode="raise", block_size=1024, return_time=False):
    """CuPy-style gather (take): out = a[indices]."""
    if mode != "raise":
        raise NotImplementedError("Only mode='raise' is currently supported")

    dense_vector, dense_backend = _to_torch_tensor(a, "a")
    indices_tensor, _ = _to_torch_tensor(indices, "indices")
    dense_vector, indices_tensor, kernel_indices = _prepare_inputs(dense_vector, indices_tensor)

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    sparse_values = _triton_gather_impl(dense_vector, kernel_indices, block_size=block_size)
    torch.cuda.synchronize()
    execution_time_ms = (time.perf_counter() - start_time) * 1000.0

    if out is not None:
        out_tensor, _ = _to_torch_tensor(out, "out")
        if out_tensor.shape != sparse_values.shape:
            raise ValueError("out shape must match gather output shape")
        if out_tensor.dtype != sparse_values.dtype:
            raise TypeError("out dtype must match gather output dtype")
        out_tensor.copy_(sparse_values)
        result = out if dense_backend == "cupy" else out_tensor
    else:
        result = _to_backend_like(sparse_values, a)

    if return_time:
        return result, execution_time_ms
    return result


def flagsparse_scatter(
    a,
    indices,
    values,
    mode="raise",
    block_size=1024,
    return_time=False,
):
    """CuPy-style scatter (put): a[indices] = values (in-place)."""
    if mode != "raise":
        raise NotImplementedError("Only mode='raise' is currently supported")

    dense_tensor, dense_backend = _to_torch_tensor(a, "a")
    values_tensor, _ = _to_torch_tensor(values, "values")
    indices_tensor, _ = _to_torch_tensor(indices, "indices")
    values_tensor, _, kernel_indices, dense_size = _prepare_scatter_inputs(
        values_tensor,
        indices_tensor,
        dense_size=dense_tensor.numel(),
        out=dense_tensor,
    )

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    _ = _triton_scatter_impl(
        values_tensor,
        kernel_indices,
        dense_size=dense_size,
        out=dense_tensor,
        block_size=block_size,
    )
    torch.cuda.synchronize()
    execution_time_ms = (time.perf_counter() - start_time) * 1000.0

    if dense_backend == "cupy":
        # DLPack view updates dense_tensor and cupy array shares memory.
        pass

    if return_time:
        return execution_time_ms
    return None


# Backward compatibility wrappers.
def triton_cusparse_gather(dense_vector, indices, block_size=1024):
    return flagsparse_gather(
        dense_vector, indices, block_size=block_size, return_time=True
    )


def triton_cusparse_scatter(sparse_values, indices, dense_size=None, out=None, block_size=1024):
    sparse_values_t, sparse_backend = _to_torch_tensor(sparse_values, "sparse_values")
    indices_t, _ = _to_torch_tensor(indices, "indices")
    if out is None:
        if dense_size is None:
            dense_size = int(indices_t.max().item()) + 1 if indices_t.numel() > 0 else 0
        out = torch.zeros(
            int(dense_size), dtype=sparse_values_t.dtype, device=sparse_values_t.device
        )
    elapsed_ms = flagsparse_scatter(
        out, indices_t, sparse_values_t, block_size=block_size, return_time=True
    )
    if sparse_backend == "cupy":
        return _to_backend_like(out, sparse_values), elapsed_ms
    return out, elapsed_ms


def pytorch_index_gather(dense_vector, indices):
    """Baseline gather using PyTorch native indexing."""
    dense_vector, indices, _ = _prepare_inputs(dense_vector, indices)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    sparse_values = dense_vector[indices]
    torch.cuda.synchronize()
    execution_time_ms = (time.perf_counter() - start_time) * 1000.0
    return sparse_values, execution_time_ms


def pytorch_index_scatter(sparse_values, indices, dense_size=None, out=None):
    """Baseline scatter using PyTorch index_copy_."""
    sparse_values, indices, _, dense_size = _prepare_scatter_inputs(
        sparse_values, indices, dense_size=dense_size, out=out
    )
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    dense_values = _pytorch_scatter_impl(sparse_values, indices, dense_size, out=out)
    torch.cuda.synchronize()
    execution_time_ms = (time.perf_counter() - start_time) * 1000.0
    return dense_values, execution_time_ms


def cusparse_spmv_gather(dense_vector, indices, selector_matrix=None):
    """Equivalent gather baseline via cuSPARSE-backed COO SpMV."""
    dense_vector, indices, _ = _prepare_inputs(dense_vector, indices)
    skip_reason = _cusparse_baseline_skip_reason(dense_vector.dtype)
    if skip_reason:
        raise RuntimeError(skip_reason)

    if selector_matrix is None:
        selector_matrix = _make_gather_selector_matrix(
            indices, dense_vector.numel(), dense_vector.dtype
        )

    try:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        sparse_values = _cusparse_spmv(selector_matrix, dense_vector)
        torch.cuda.synchronize()
        execution_time_ms = (time.perf_counter() - start_time) * 1000.0
    except Exception as exc:
        raise RuntimeError(
            "cuSPARSE gather baseline is unavailable in this PyTorch/CUDA environment"
        ) from exc

    return sparse_values, execution_time_ms, selector_matrix


def cusparse_spmv_scatter(sparse_values, indices, dense_size=None, selector_matrix=None):
    """Equivalent scatter baseline via cuSPARSE-backed COO SpMV."""
    sparse_values, indices, _, dense_size = _prepare_scatter_inputs(
        sparse_values, indices, dense_size=dense_size, out=None
    )
    skip_reason = _cusparse_baseline_skip_reason(sparse_values.dtype)
    if skip_reason:
        raise RuntimeError(skip_reason)

    if selector_matrix is None:
        selector_matrix = _make_scatter_selector_matrix(indices, dense_size, sparse_values.dtype)

    try:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        dense_values = _cusparse_spmv(selector_matrix, sparse_values)
        torch.cuda.synchronize()
        execution_time_ms = (time.perf_counter() - start_time) * 1000.0
    except Exception as exc:
        raise RuntimeError(
            "cuSPARSE scatter baseline is unavailable in this PyTorch/CUDA environment"
        ) from exc

    return dense_values, execution_time_ms, selector_matrix


def benchmark_gather_case(
    dense_size=65536,
    nnz=4096,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_size=1024,
    run_cusparse=True,
):
    """Benchmark Triton vs PyTorch indexing vs cuSPARSE-backed COO SpMV."""
    device = torch.device("cuda")
    dense_vector = _build_random_dense(dense_size, value_dtype, device)
    indices = _build_indices(nnz, dense_size, index_dtype, device, unique=False)

    dense_vector, indices, kernel_indices = _prepare_inputs(dense_vector, indices)
    expected = dense_vector[indices]

    pytorch_op = lambda: dense_vector[indices]
    triton_op = lambda: _triton_gather_impl(dense_vector, kernel_indices, block_size=block_size)

    pytorch_values, pytorch_ms = _benchmark_cuda_op(pytorch_op, warmup=warmup, iters=iters)
    triton_values, triton_ms = _benchmark_cuda_op(triton_op, warmup=warmup, iters=iters)

    atol, rtol = _tolerance_for_dtype(value_dtype)
    triton_match = torch.allclose(triton_values, expected, atol=atol, rtol=rtol)
    triton_max_error = (
        float(torch.max(torch.abs(triton_values - expected)).item())
        if nnz > 0
        else 0.0
    )

    cusparse_ms = None
    cusparse_match = None
    cusparse_max_error = None
    cusparse_reason = None
    if run_cusparse:
        skip_reason = _cusparse_baseline_skip_reason(value_dtype)
        if skip_reason:
            cusparse_reason = skip_reason
        else:
            try:
                selector_matrix = _make_gather_selector_matrix(
                    indices, dense_vector.numel(), dense_vector.dtype
                )
                cusparse_op = lambda: _cusparse_spmv(selector_matrix, dense_vector)
                cusparse_values, cusparse_ms = _benchmark_cuda_op(
                    cusparse_op, warmup=warmup, iters=iters
                )
                cusparse_match = torch.allclose(
                    cusparse_values, expected, atol=atol, rtol=rtol
                )
                cusparse_max_error = (
                    float(torch.max(torch.abs(cusparse_values - expected)).item())
                    if nnz > 0
                    else 0.0
                )
            except Exception as exc:
                cusparse_reason = str(exc)

    triton_speedup_vs_pytorch = (
        pytorch_ms / triton_ms if triton_ms > 0 else float("inf")
    )
    triton_speedup_vs_cusparse = (
        cusparse_ms / triton_ms
        if (cusparse_ms is not None and triton_ms > 0)
        else None
    )

    return {
        "parameters": {
            "dense_size": dense_size,
            "nnz": nnz,
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
            "warmup": warmup,
            "iters": iters,
        },
        "performance": {
            "pytorch_ms": pytorch_ms,
            "triton_ms": triton_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_pytorch": triton_speedup_vs_pytorch,
            "triton_speedup_vs_cusparse": triton_speedup_vs_cusparse,
        },
        "verification": {
            "triton_match_pytorch": triton_match,
            "triton_max_error": triton_max_error,
            "cusparse_match_pytorch": cusparse_match,
            "cusparse_max_error": cusparse_max_error,
        },
        "backend_status": {
            "cusparse_unavailable_reason": cusparse_reason,
        },
        "samples": {
            "pytorch": pytorch_values,
            "triton": triton_values,
        },
    }


def benchmark_scatter_case(
    dense_size=65536,
    nnz=4096,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_size=1024,
    run_cusparse=True,
    unique_indices=True,
):
    """Benchmark Triton scatter vs PyTorch index_copy vs cuSPARSE-backed COO SpMV."""
    device = torch.device("cuda")
    sparse_values = _build_random_dense(nnz, value_dtype, device)
    indices = _build_indices(nnz, dense_size, index_dtype, device, unique=unique_indices)

    sparse_values, indices, kernel_indices, dense_size = _prepare_scatter_inputs(
        sparse_values, indices, dense_size=dense_size, out=None
    )
    expected = _pytorch_scatter_impl(sparse_values, indices, dense_size)

    pytorch_op = lambda: _pytorch_scatter_impl(sparse_values, indices, dense_size)
    triton_op = lambda: _triton_scatter_impl(
        sparse_values,
        kernel_indices,
        dense_size=dense_size,
        out=None,
        block_size=block_size,
    )

    pytorch_values, pytorch_ms = _benchmark_cuda_op(pytorch_op, warmup=warmup, iters=iters)
    triton_values, triton_ms = _benchmark_cuda_op(triton_op, warmup=warmup, iters=iters)

    atol, rtol = _tolerance_for_dtype(value_dtype)
    triton_match = torch.allclose(triton_values, expected, atol=atol, rtol=rtol)
    triton_max_error = (
        float(torch.max(torch.abs(triton_values - expected)).item())
        if dense_size > 0
        else 0.0
    )

    cusparse_ms = None
    cusparse_match = None
    cusparse_max_error = None
    cusparse_reason = None
    if run_cusparse:
        skip_reason = _cusparse_baseline_skip_reason(value_dtype)
        if skip_reason:
            cusparse_reason = skip_reason
        else:
            try:
                selector_matrix = _make_scatter_selector_matrix(
                    indices, dense_size, sparse_values.dtype
                )
                cusparse_op = lambda: _cusparse_spmv(selector_matrix, sparse_values)
                cusparse_values, cusparse_ms = _benchmark_cuda_op(
                    cusparse_op, warmup=warmup, iters=iters
                )
                cusparse_match = torch.allclose(
                    cusparse_values, expected, atol=atol, rtol=rtol
                )
                cusparse_max_error = (
                    float(torch.max(torch.abs(cusparse_values - expected)).item())
                    if dense_size > 0
                    else 0.0
                )
            except Exception as exc:
                cusparse_reason = str(exc)

    triton_speedup_vs_pytorch = (
        pytorch_ms / triton_ms if triton_ms > 0 else float("inf")
    )
    triton_speedup_vs_cusparse = (
        cusparse_ms / triton_ms
        if (cusparse_ms is not None and triton_ms > 0)
        else None
    )

    return {
        "parameters": {
            "dense_size": dense_size,
            "nnz": nnz,
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
            "warmup": warmup,
            "iters": iters,
            "unique_indices": unique_indices,
        },
        "performance": {
            "pytorch_ms": pytorch_ms,
            "triton_ms": triton_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_pytorch": triton_speedup_vs_pytorch,
            "triton_speedup_vs_cusparse": triton_speedup_vs_cusparse,
        },
        "verification": {
            "triton_match_pytorch": triton_match,
            "triton_max_error": triton_max_error,
            "cusparse_match_pytorch": cusparse_match,
            "cusparse_max_error": cusparse_max_error,
        },
        "backend_status": {
            "cusparse_unavailable_reason": cusparse_reason,
        },
        "samples": {
            "pytorch": pytorch_values,
            "triton": triton_values,
        },
    }


def benchmark_spmv_case(
    n_rows=4096,
    n_cols=4096,
    nnz=65536,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_nnz=256,
    max_segments=64,
    run_cusparse=True,
):
    """Benchmark Triton CSR SpMV vs cuSPARSE (CuPy CSR @ x)."""
    device = torch.device("cuda")
    data, indices, indptr = _build_random_csr(
        n_rows, n_cols, nnz, value_dtype, index_dtype, device
    )
    x = _build_random_dense(n_cols, value_dtype, device)
    shape = (n_rows, n_cols)
    data, kernel_indices, kernel_indptr, x, _, _ = _prepare_spmv_csr_inputs(
        data, indices, indptr, x, shape
    )
    triton_y, triton_ms = flagsparse_spmv_csr(
        data,
        kernel_indices,
        kernel_indptr,
        x,
        shape,
        block_nnz=block_nnz,
        max_segments=max_segments,
        return_time=True,
    )
    _cupy_supported_dtypes = (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    if (
        cp is not None
        and cpx_sparse is not None
        and value_dtype in _cupy_supported_dtypes
    ):
        data_cp = _cupy_from_torch(data)
        indices_cp = _cupy_from_torch(indices.to(torch.int64))
        indptr_cp = _cupy_from_torch(indptr)
        x_cp = _cupy_from_torch(x)
        A_csr = cpx_sparse.csr_matrix(
            (data_cp, indices_cp, indptr_cp), shape=shape
        )
        ref_y = A_csr @ x_cp
        expected = _torch_from_cupy(ref_y)
    else:
        row_indices = torch.repeat_interleave(
            torch.arange(n_rows, device=device, dtype=torch.int64),
            indptr[1:] - indptr[:-1],
        )
        col_ind = indices.to(torch.int64)
        coo = torch.sparse_coo_tensor(
            torch.stack([row_indices, col_ind]),
            data,
            shape,
            device=device,
        ).coalesce()
        x_2d = x.unsqueeze(1)
        if value_dtype in (torch.float16, torch.bfloat16):
            coo_f32 = coo.to(torch.float32)
            x_2d_f32 = x_2d.to(torch.float32)
            expected = torch.sparse.mm(coo_f32, x_2d_f32).squeeze(1).to(value_dtype)
        else:
            expected = torch.sparse.mm(coo, x_2d).squeeze(1)
    atol, rtol = _tolerance_for_dtype(value_dtype)
    triton_match = torch.allclose(triton_y, expected, atol=atol, rtol=rtol)
    triton_max_error = (
        float(torch.max(torch.abs(triton_y - expected)).item())
        if n_rows > 0
        else 0.0
    )
    cusparse_ms = None
    cusparse_match = None
    cusparse_max_error = None
    cusparse_reason = None
    if (
        run_cusparse
        and cp is not None
        and cpx_sparse is not None
        and value_dtype in _cupy_supported_dtypes
    ):
        skip_reason = _cusparse_baseline_skip_reason(value_dtype)
        if skip_reason:
            cusparse_reason = skip_reason
        else:
            try:
                cusparse_op = lambda: _torch_from_cupy(
                    A_csr @ _cupy_from_torch(x)
                )
                cusparse_values, cusparse_ms = _benchmark_cuda_op(
                    cusparse_op, warmup=warmup, iters=iters
                )
                cusparse_match = torch.allclose(
                    cusparse_values, expected, atol=atol, rtol=rtol
                )
                cusparse_max_error = (
                    float(torch.max(torch.abs(cusparse_values - expected)).item())
                    if n_rows > 0
                    else 0.0
                )
            except Exception as exc:
                cusparse_reason = str(exc)
    elif run_cusparse and value_dtype not in _cupy_supported_dtypes:
        cusparse_reason = (
            "float16/bfloat16 not supported by CuPy sparse; skipped"
        )
    triton_speedup_vs_cusparse = (
        cusparse_ms / triton_ms
        if (cusparse_ms is not None and triton_ms > 0)
        else None
    )
    return {
        "parameters": {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "nnz": nnz,
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
            "warmup": warmup,
            "iters": iters,
        },
        "performance": {
            "triton_ms": triton_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_cusparse": triton_speedup_vs_cusparse,
        },
        "verification": {
            "triton_match_reference": triton_match,
            "triton_max_error": triton_max_error,
            "cusparse_match_reference": cusparse_match,
            "cusparse_max_error": cusparse_max_error,
        },
        "backend_status": {
            "cusparse_unavailable_reason": cusparse_reason,
        },
        "samples": {"triton": triton_y, "reference": expected},
    }


def benchmark_performance(
    dense_size=65536,
    nnz=4096,
    dtype=torch.float32,
    index_dtype=torch.int32,
):
    """Backward-compatible benchmark entry."""
    result = benchmark_gather_case(
        dense_size=dense_size,
        nnz=nnz,
        value_dtype=dtype,
        index_dtype=index_dtype,
        warmup=10,
        iters=100,
        run_cusparse=False,
    )
    return {
        "triton_time_ms": result["performance"]["triton_ms"],
        "results_match": result["verification"]["triton_match_pytorch"],
        "dtype": str(dtype),
        "index_dtype": str(index_dtype),
        "dense_size": dense_size,
        "nnz": nnz,
    }


def comprehensive_gather_test(
    dense_size=100000,
    nnz=10000,
    dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    run_cusparse=True,
):
    """Full test entry for one configuration."""
    return benchmark_gather_case(
        dense_size=dense_size,
        nnz=nnz,
        value_dtype=dtype,
        index_dtype=index_dtype,
        warmup=warmup,
        iters=iters,
        run_cusparse=run_cusparse,
    )


def comprehensive_scatter_test(
    dense_size=100000,
    nnz=10000,
    dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    run_cusparse=True,
    unique_indices=True,
):
    """Full scatter test entry for one configuration."""
    return benchmark_scatter_case(
        dense_size=dense_size,
        nnz=nnz,
        value_dtype=dtype,
        index_dtype=index_dtype,
        warmup=warmup,
        iters=iters,
        run_cusparse=run_cusparse,
        unique_indices=unique_indices,
    )
