import pytest
import torch

from flagsparse import flagsparse_sddmm_csr, flagsparse_spgemm_csr
import flagsparse.sparse_operations._common as common_mod
import flagsparse.sparse_operations.spgemm_csr as spgemm_mod

from tests.pytest.param_shapes import (
    SDDMM_DTYPES,
    SDDMM_DTYPE_IDS,
    SDDMM_MNK_SHAPES,
    SPGEMM_DTYPES,
    SPGEMM_DTYPE_IDS,
    SPGEMM_MNK_SHAPES,
)


CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _random_csr(rows, cols, dtype, device):
    denom = max(rows * cols, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(rows, cols, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    vals = torch.randn(rows, cols, dtype=dtype, device=device) * mask.to(dtype=dtype)
    return vals.to_sparse_csr()


def _csr_to_dense(data, indices, indptr, shape):
    csr = torch.sparse_csr_tensor(
        indptr,
        indices,
        data,
        size=shape,
        dtype=data.dtype,
        device=data.device,
    )
    return csr.to_dense()


def _tol(dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 1e-10, 1e-8


def test_spgemm_csr_sparse_ref_backend_selects_hipsparse_when_direct_supported(monkeypatch):
    monkeypatch.setattr(common_mod, "_IS_ROCM_RUNTIME", True, raising=False)
    monkeypatch.setattr(
        spgemm_mod,
        "_hipsparse_spgemm_csr_skip_reason",
        lambda *args, **kwargs: None,
    )
    backend, reason = spgemm_mod._spgemm_csr_sparse_ref_backend(
        torch.float32,
        torch.int32,
        torch.int32,
        torch.int32,
        torch.int32,
    )
    assert backend == "hipsparse"
    assert reason is None


def test_spgemm_csr_sparse_ref_backend_reports_direct_reason_when_unsupported(monkeypatch):
    monkeypatch.setattr(common_mod, "_IS_ROCM_RUNTIME", True, raising=False)
    monkeypatch.setattr(
        spgemm_mod,
        "_hipsparse_spgemm_csr_skip_reason",
        lambda *args, **kwargs: "direct hipSPARSE CSR SpGEMM unsupported",
    )
    backend, reason = spgemm_mod._spgemm_csr_sparse_ref_backend(
        torch.float32,
        torch.int32,
        torch.int32,
        torch.int32,
        torch.int32,
    )
    assert backend is None
    assert reason == "direct hipSPARSE CSR SpGEMM unsupported"


def test_spgemm_csr_sparse_ref_backend_reports_cupy_unavailable(monkeypatch):
    monkeypatch.setattr(common_mod, "_IS_ROCM_RUNTIME", False, raising=False)
    monkeypatch.setattr(spgemm_mod, "cp", None, raising=False)
    monkeypatch.setattr(spgemm_mod, "cpx_sparse", None, raising=False)
    backend, reason = spgemm_mod._spgemm_csr_sparse_ref_backend()
    assert backend is None
    assert reason == "CuPy/cuSPARSE is not available"


def test_spgemm_csr_reference_returns_csr_tuple_and_metadata():
    a_data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    a_indices = torch.tensor([0, 1, 1], dtype=torch.int32)
    a_indptr = torch.tensor([0, 2, 3], dtype=torch.int64)
    b_data = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
    b_indices = torch.tensor([0, 0, 1], dtype=torch.int32)
    b_indptr = torch.tensor([0, 1, 3], dtype=torch.int64)

    reference, metadata = spgemm_mod._spgemm_csr_reference(
        a_data,
        a_indices,
        a_indptr,
        (2, 2),
        b_data,
        b_indices,
        b_indptr,
        (2, 2),
        return_metadata=True,
    )

    c_data, c_indices, c_indptr, c_shape = reference
    dense = _csr_to_dense(c_data, c_indices, c_indptr, c_shape)
    expected = torch.tensor([[4.0, 12.0], [15.0, 18.0]], dtype=torch.float32)

    assert metadata["backend"] == "torch"
    assert metadata["pytorch_sparse_format"] in ("CSR", "COO")
    assert c_shape == (2, 2)
    assert torch.allclose(dense, expected, rtol=1e-6, atol=1e-6)


@CUDA_REQUIRED
@pytest.mark.spgemm_csr
@pytest.mark.parametrize("M, N, K", SPGEMM_MNK_SHAPES)
@pytest.mark.parametrize("dtype", SPGEMM_DTYPES, ids=SPGEMM_DTYPE_IDS)
def test_spgemm_csr_matches_torch(M, N, K, dtype):
    device = torch.device("cuda")
    A = _random_csr(M, K, dtype, device)
    B = _random_csr(K, N, dtype, device)
    c_data, c_indices, c_indptr, c_shape = flagsparse_spgemm_csr(
        A.values(),
        A.col_indices().to(torch.int32),
        A.crow_indices(),
        (M, K),
        B.values(),
        B.col_indices().to(torch.int32),
        B.crow_indices(),
        (K, N),
    )
    got = _csr_to_dense(c_data, c_indices, c_indptr, c_shape)
    ref = torch.sparse.mm(A, B.to_dense())
    rtol, atol = _tol(dtype)
    assert torch.allclose(got, ref, rtol=rtol, atol=atol)


@CUDA_REQUIRED
@pytest.mark.sddmm_csr
@pytest.mark.parametrize("M, N, K", SDDMM_MNK_SHAPES)
@pytest.mark.parametrize("dtype", SDDMM_DTYPES, ids=SDDMM_DTYPE_IDS)
def test_sddmm_csr_matches_sampled_dense_reference(M, N, K, dtype):
    device = torch.device("cuda")
    pattern = _random_csr(M, N, dtype, device)
    indices = pattern.col_indices().to(torch.int32)
    indptr = pattern.crow_indices()
    data = pattern.values()
    x = torch.randn(M, K, dtype=dtype, device=device)
    y = torch.randn(N, K, dtype=dtype, device=device)
    alpha = 1.25
    beta = 0.5

    got = flagsparse_sddmm_csr(
        data=data,
        indices=indices,
        indptr=indptr,
        x=x,
        y=y,
        shape=(M, N),
        alpha=alpha,
        beta=beta,
    )
    row_ids = torch.repeat_interleave(
        torch.arange(M, dtype=torch.int64, device=device),
        indptr[1:] - indptr[:-1],
    )
    ref = alpha * torch.sum(x[row_ids] * y[indices.to(torch.int64)], dim=1) + beta * data
    rtol, atol = _tol(dtype)
    assert torch.allclose(got, ref, rtol=rtol, atol=atol)
