import pytest
import torch

from flagsparse import flagsparse_spsv_coo, flagsparse_spsv_csr

from tests.pytest.param_shapes import SPSV_N

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
    from cupyx.scipy.sparse.linalg import spsolve_triangular as cpx_spsolve_triangular
except Exception:
    cp = None
    cpx_sparse = None
    cpx_spsolve_triangular = None


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
<<<<<<< HEAD
COMPLEX32_DTYPE = getattr(torch, "complex32", None)
if COMPLEX32_DTYPE is None:
    COMPLEX32_DTYPE = getattr(torch, "chalf", None)

SUPPORTED_COMPLEX_DTYPES = []
if COMPLEX32_DTYPE is not None:
    SUPPORTED_COMPLEX_DTYPES.append(COMPLEX32_DTYPE)
SUPPORTED_COMPLEX_DTYPES.append(torch.complex64)

SUPPORTED_DTYPES = [torch.float32, torch.float64, *SUPPORTED_COMPLEX_DTYPES]
=======
SUPPORTED_COMPLEX_DTYPES = [torch.complex64, torch.complex128]

SUPPORTED_DTYPES = [torch.float32, torch.float64, *SUPPORTED_COMPLEX_DTYPES]
NON_TRANS_DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]
TRANS_DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]
>>>>>>> 5a83e0f (test)


def _dtype_id(dtype):
    return str(dtype).replace("torch.", "")


def _tol(dtype):
<<<<<<< HEAD
    if COMPLEX32_DTYPE is not None and dtype == COMPLEX32_DTYPE:
        return 5e-3, 5e-3
=======
>>>>>>> 5a83e0f (test)
    if dtype in (torch.float32, torch.complex64):
        return 1e-4, 1e-3
    return 1e-10, 1e-8


def _rand_like(dtype, shape, device):
    if dtype in (torch.float32, torch.float64):
        return torch.randn(shape, dtype=dtype, device=device)
<<<<<<< HEAD
    if COMPLEX32_DTYPE is not None and dtype == COMPLEX32_DTYPE:
        pair = torch.randn((*shape, 2), dtype=torch.float16, device=device) * 0.1
        return torch.view_as_complex(pair)
    base = torch.float32
=======
    base = torch.float32 if dtype == torch.complex64 else torch.float64
>>>>>>> 5a83e0f (test)
    r = torch.randn(shape, dtype=base, device=device)
    i = torch.randn(shape, dtype=base, device=device)
    return torch.complex(r, i)


def _ref_dtype(dtype):
<<<<<<< HEAD
    if COMPLEX32_DTYPE is not None and dtype == COMPLEX32_DTYPE:
        return torch.complex64
=======
>>>>>>> 5a83e0f (test)
    return dtype


def _safe_cast_tensor(tensor, dtype):
<<<<<<< HEAD
    if COMPLEX32_DTYPE is not None and dtype == COMPLEX32_DTYPE:
        real = tensor.real.to(torch.float16)
        imag = tensor.imag.to(torch.float16)
        return torch.view_as_complex(torch.stack([real, imag], dim=-1).contiguous())
=======
>>>>>>> 5a83e0f (test)
    return tensor.to(dtype)


def _cmp_view(tensor, dtype):
<<<<<<< HEAD
    if COMPLEX32_DTYPE is not None and dtype == COMPLEX32_DTYPE:
        return tensor.to(torch.complex64)
    return tensor


def _build_lower_triangular(n, dtype, device):
    off = _rand_like(dtype, (n, n), device) * 0.02
    A = torch.tril(off)
=======
    return tensor


def _build_triangular(n, dtype, device, lower=True):
    off = _rand_like(dtype, (n, n), device) * 0.02
    A = torch.tril(off) if lower else torch.triu(off)
>>>>>>> 5a83e0f (test)
    if torch.is_complex(A):
        diag = (torch.rand(n, device=device, dtype=A.real.dtype) + 2.0).to(A.real.dtype)
        A = A + torch.diag(torch.complex(diag, torch.zeros_like(diag)))
    else:
        diag = torch.rand(n, device=device, dtype=A.dtype) + 2.0
        A = A + torch.diag(diag)
    return A


def _cupy_csr_from_torch(data, indices, indptr, shape):
    if cp is None or cpx_sparse is None:
        return None
<<<<<<< HEAD
    data_ref = data.to(torch.complex64) if COMPLEX32_DTYPE is not None and data.dtype == COMPLEX32_DTYPE else data
    data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data_ref.contiguous()))
=======
    data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data.contiguous()))
>>>>>>> 5a83e0f (test)
    idx_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices.to(torch.int64).contiguous()))
    ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr.to(torch.int64).contiguous()))
    return cpx_sparse.csr_matrix((data_cp, idx_cp, ptr_cp), shape=shape)


def _cupy_ref_spsv(A_cp, b_t, *, lower, unit_diagonal=False):
    if cp is None or cpx_spsolve_triangular is None:
        return None
<<<<<<< HEAD
    b_ref = b_t.to(torch.complex64) if COMPLEX32_DTYPE is not None and b_t.dtype == COMPLEX32_DTYPE else b_t
    b_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(b_ref.contiguous()))
    x_cp = cpx_spsolve_triangular(A_cp, b_cp, lower=lower, unit_diagonal=unit_diagonal)
    x_t = torch.utils.dlpack.from_dlpack(x_cp.toDlpack())
    if COMPLEX32_DTYPE is not None and b_t.dtype == COMPLEX32_DTYPE:
        return x_t.to(torch.complex64)
=======
    b_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(b_t.contiguous()))
    x_cp = cpx_spsolve_triangular(A_cp, b_cp, lower=lower, unit_diagonal=unit_diagonal)
    x_t = torch.utils.dlpack.from_dlpack(x_cp.toDlpack())
>>>>>>> 5a83e0f (test)
    return x_t.to(b_t.dtype)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
    ids=["float32", "float64"],
)
def test_spsv_csr_lower_matches_dense(n, dtype):
    # Keep the original baseline test case untouched in semantics.
    device = torch.device("cuda")
    base = torch.tril(torch.randn(n, n, dtype=dtype, device=device))
    eye = torch.eye(n, dtype=dtype, device=device)
    A = base + eye * (float(n) * 0.5 + 2.0)
    b = torch.randn(n, dtype=dtype, device=device)
    x_ref = torch.linalg.solve_triangular(
        A, b.unsqueeze(-1), upper=False
    ).squeeze(-1)
    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices()
    indptr = Asp.crow_indices().to(torch.int64)
    x = flagsparse_spsv_csr(
        data,
        indices,
        indptr,
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
    )
    rtol = 1e-4 if dtype == torch.float32 else 1e-10
    atol = 1e-5 if dtype == torch.float32 else 1e-10
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
<<<<<<< HEAD
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_spsv_csr_non_trans_supported_combos(n, dtype, index_dtype):
    device = torch.device("cuda")
    A = _build_lower_triangular(n, dtype, device)
=======
@pytest.mark.parametrize("dtype", NON_TRANS_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_spsv_csr_non_trans_supported_combos(n, dtype, index_dtype):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=True)
>>>>>>> 5a83e0f (test)
    b = _rand_like(dtype, (n,), device)
    x_ref = torch.linalg.solve_triangular(
        A.to(_ref_dtype(dtype)), b.to(_ref_dtype(dtype)).unsqueeze(-1), upper=False
    ).squeeze(-1)

    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(index_dtype)
    indptr = Asp.crow_indices().to(index_dtype)

    x = flagsparse_spsv_csr(
        data,
        indices,
        indptr,
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(_cmp_view(x, dtype), _cmp_view(x_ref, dtype), rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
<<<<<<< HEAD
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES, ids=_dtype_id)
def test_spsv_csr_trans_int32_supported_combos(n, dtype):
    device = torch.device("cuda")
    A = _build_lower_triangular(n, dtype, device)
=======
@pytest.mark.parametrize("dtype", TRANS_DTYPES, ids=_dtype_id)
def test_spsv_csr_trans_int32_supported_combos(n, dtype):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=True)
>>>>>>> 5a83e0f (test)
    b = _rand_like(dtype, (n,), device)
    A_ref = A.to(_ref_dtype(dtype))
    b_ref = b.to(_ref_dtype(dtype))
    x_ref = torch.linalg.solve_triangular(
        A_ref.transpose(-2, -1), b_ref.unsqueeze(-1), upper=True
    ).squeeze(-1)

    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(torch.int32)
    indptr = Asp.crow_indices().to(torch.int32)

    x = flagsparse_spsv_csr(
        data,
        indices,
        indptr,
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=True,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(_cmp_view(x, dtype), _cmp_view(x_ref, dtype), rtol=rtol, atol=atol)


@pytest.mark.spsv
<<<<<<< HEAD
@pytest.mark.skipif(cp is None or cpx_spsolve_triangular is None, reason="CuPy/cuSPARSE required")
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES, ids=_dtype_id)
def test_spsv_csr_matches_cusparse_non_trans_and_trans(n, dtype):
    device = torch.device("cuda")
    A = _build_lower_triangular(n, dtype, device)
=======
@pytest.mark.skipif(
    cp is None or cpx_sparse is None or cpx_spsolve_triangular is None,
    reason="CuPy/cuSPARSE required",
)
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", NON_TRANS_DTYPES, ids=_dtype_id)
def test_spsv_csr_matches_cusparse_non_trans(n, dtype):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=True)
>>>>>>> 5a83e0f (test)
    b = _rand_like(dtype, (n,), device)

    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(torch.int32)
    indptr = Asp.crow_indices().to(torch.int32)
    A_cp = _cupy_csr_from_torch(data, indices, indptr, (n, n))

    x_non = flagsparse_spsv_csr(
        data, indices, indptr, b, (n, n), lower=True, unit_diagonal=False, transpose=False
    )
    x_non_ref = _cupy_ref_spsv(A_cp, b, lower=True, unit_diagonal=False)

<<<<<<< HEAD
=======
    rtol, atol = _tol(dtype)
    assert torch.allclose(_cmp_view(x_non, dtype), _cmp_view(x_non_ref, dtype), rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.skipif(
    cp is None or cpx_sparse is None or cpx_spsolve_triangular is None,
    reason="CuPy/cuSPARSE required",
)
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", TRANS_DTYPES, ids=_dtype_id)
def test_spsv_csr_matches_cusparse_trans(n, dtype):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)

    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(torch.int32)
    indptr = Asp.crow_indices().to(torch.int32)
    A_cp = _cupy_csr_from_torch(data, indices, indptr, (n, n))

>>>>>>> 5a83e0f (test)
    x_trans = flagsparse_spsv_csr(
        data, indices, indptr, b, (n, n), lower=True, unit_diagonal=False, transpose=True
    )
    x_trans_ref = _cupy_ref_spsv(A_cp.transpose().tocsr(), b, lower=False, unit_diagonal=False)

    rtol, atol = _tol(dtype)
<<<<<<< HEAD
    assert torch.allclose(_cmp_view(x_non, dtype), _cmp_view(x_non_ref, dtype), rtol=rtol, atol=atol)
    assert torch.allclose(_cmp_view(x_trans, dtype), _cmp_view(x_trans_ref, dtype), rtol=rtol, atol=atol)
=======
    assert torch.allclose(_cmp_view(x_trans, dtype), _cmp_view(x_trans_ref, dtype), rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", NON_TRANS_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_spsv_csr_non_trans_upper_supported_combos(n, dtype, index_dtype):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=False)
    b = _rand_like(dtype, (n,), device)
    x_ref = torch.linalg.solve_triangular(
        A.to(_ref_dtype(dtype)), b.to(_ref_dtype(dtype)).unsqueeze(-1), upper=True
    ).squeeze(-1)

    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(index_dtype)
    indptr = Asp.crow_indices().to(index_dtype)

    x = flagsparse_spsv_csr(
        data,
        indices,
        indptr,
        b,
        (n, n),
        lower=False,
        unit_diagonal=False,
        transpose=False,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(_cmp_view(x, dtype), _cmp_view(x_ref, dtype), rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", TRANS_DTYPES, ids=_dtype_id)
def test_spsv_csr_trans_upper_int32_supported_combos(n, dtype):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=False)
    b = _rand_like(dtype, (n,), device)
    A_ref = A.to(_ref_dtype(dtype))
    b_ref = b.to(_ref_dtype(dtype))
    x_ref = torch.linalg.solve_triangular(
        A_ref.transpose(-2, -1), b_ref.unsqueeze(-1), upper=False
    ).squeeze(-1)

    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(torch.int32)
    indptr = Asp.crow_indices().to(torch.int32)

    x = flagsparse_spsv_csr(
        data,
        indices,
        indptr,
        b,
        (n, n),
        lower=False,
        unit_diagonal=False,
        transpose=True,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(_cmp_view(x, dtype), _cmp_view(x_ref, dtype), rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.skipif(
    cp is None or cpx_sparse is None or cpx_spsolve_triangular is None,
    reason="CuPy/cuSPARSE required",
)
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", NON_TRANS_DTYPES, ids=_dtype_id)
def test_spsv_csr_matches_cusparse_upper_non_trans(n, dtype):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=False)
    b = _rand_like(dtype, (n,), device)

    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(torch.int32)
    indptr = Asp.crow_indices().to(torch.int32)
    A_cp = _cupy_csr_from_torch(data, indices, indptr, (n, n))

    x_non = flagsparse_spsv_csr(
        data, indices, indptr, b, (n, n), lower=False, unit_diagonal=False, transpose=False
    )
    x_non_ref = _cupy_ref_spsv(A_cp, b, lower=False, unit_diagonal=False)

    rtol, atol = _tol(dtype)
    assert torch.allclose(_cmp_view(x_non, dtype), _cmp_view(x_non_ref, dtype), rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.skipif(
    cp is None or cpx_sparse is None or cpx_spsolve_triangular is None,
    reason="CuPy/cuSPARSE required",
)
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", TRANS_DTYPES, ids=_dtype_id)
def test_spsv_csr_matches_cusparse_upper_trans(n, dtype):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=False)
    b = _rand_like(dtype, (n,), device)

    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(torch.int32)
    indptr = Asp.crow_indices().to(torch.int32)
    A_cp = _cupy_csr_from_torch(data, indices, indptr, (n, n))

    x_trans = flagsparse_spsv_csr(
        data, indices, indptr, b, (n, n), lower=False, unit_diagonal=False, transpose=True
    )
    x_trans_ref = _cupy_ref_spsv(A_cp.transpose().tocsr(), b, lower=True, unit_diagonal=False)

    rtol, atol = _tol(dtype)
    assert torch.allclose(_cmp_view(x_trans, dtype), _cmp_view(x_trans_ref, dtype), rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
def test_spsv_coo_transpose_complex128_routes_through_csr(n):
    device = torch.device("cuda")
    dtype = torch.complex128
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)
    x_ref = torch.linalg.solve_triangular(
        A.transpose(-2, -1), b.unsqueeze(-1), upper=True
    ).squeeze(-1)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices()
    data = A_coo.values()

    x = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=True,
        coo_mode="auto",
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)
>>>>>>> 5a83e0f (test)
