import pytest
import torch

from flagsparse import flagsparse_spsm_coo, flagsparse_spsm_csr
from flagsparse.sparse_operations import spsm as fs_spsm_impl

from tests.pytest.param_shapes import SPSM_N_RHS


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


SPSM_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
SPSM_DTYPE_IDS = ("float32", "float64", "complex64", "complex128")


def _build_lower_dense(n, dtype, device):
    base = torch.tril(torch.randn(n, n, dtype=dtype, device=device))
    eye = torch.eye(n, dtype=dtype, device=device)
    return base + eye * (float(n) * 0.5 + 2.0)


def _tol(dtype):
    if dtype in (torch.float32, torch.complex64):
        return 1e-4, 1e-5
    return 1e-10, 1e-10


@pytest.mark.spsm
@pytest.mark.spsm_csr
@pytest.mark.parametrize("n, n_rhs", SPSM_N_RHS)
@pytest.mark.parametrize("dtype", SPSM_DTYPES, ids=SPSM_DTYPE_IDS)
def test_spsm_csr_lower_matches_dense(n, n_rhs, dtype):
    device = torch.device("cuda")
    A = _build_lower_dense(n, dtype, device)
    B = torch.randn(n, n_rhs, dtype=dtype, device=device)
    ref = torch.linalg.solve_triangular(A, B, upper=False)
    Acsr = A.to_sparse_csr()
    out = flagsparse_spsm_csr(
        Acsr.values(),
        Acsr.col_indices().to(torch.int32),
        Acsr.crow_indices().to(torch.int32),
        B,
        (n, n),
        lower=True,
        unit_diagonal=False,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol)


@pytest.mark.spsm
@pytest.mark.spsm_coo
@pytest.mark.parametrize("n, n_rhs", SPSM_N_RHS)
@pytest.mark.parametrize("dtype", SPSM_DTYPES, ids=SPSM_DTYPE_IDS)
def test_spsm_coo_lower_matches_dense(n, n_rhs, dtype):
    device = torch.device("cuda")
    A = _build_lower_dense(n, dtype, device)
    B = torch.randn(n, n_rhs, dtype=dtype, device=device)
    ref = torch.linalg.solve_triangular(A, B, upper=False)
    Acoo = A.to_sparse_coo().coalesce()
    indices = Acoo.indices()
    out = flagsparse_spsm_coo(
        Acoo.values(),
        indices[0].to(torch.int32).contiguous(),
        indices[1].to(torch.int32).contiguous(),
        B,
        (n, n),
        lower=True,
        unit_diagonal=False,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol)


@pytest.mark.spsm
def test_spsm_level_schedule_lower_orders_independent_rows_together():
    device = torch.device("cuda")
    indices = torch.tensor([0, 0, 1, 0, 2, 1, 2, 3], dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 1, 3, 5, 8], dtype=torch.int32, device=device)

    schedule = fs_spsm_impl._build_spsm_level_schedule(indices, indptr, 4, lower=True)

    assert schedule["level_row_map32"].tolist() == [0, 1, 2, 3]
    assert schedule["level_ptr"] == (0, 1, 3, 4)


@pytest.mark.spsm
def test_spsm_level_schedule_upper_orders_independent_rows_together():
    device = torch.device("cuda")
    indices = torch.tensor([3, 2, 0, 3, 1, 3, 2, 3], dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 3, 5, 7, 8], dtype=torch.int32, device=device)

    schedule = fs_spsm_impl._build_spsm_level_schedule(indices, indptr, 4, lower=False)

    assert schedule["level_row_map32"].tolist() == [3, 1, 2, 0]
    assert schedule["level_ptr"] == (0, 1, 3, 4)
