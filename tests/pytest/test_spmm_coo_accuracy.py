import pytest
import torch

from flagsparse import flagsparse_spmm_coo
import flagsparse.sparse_operations.spmm_coo as ast_ops

from tests.pytest.accuracy_utils import close_tolerances
from tests.pytest.param_shapes import MNK_SHAPES


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/PyTorch GPU required")


def _value_dtype_cases():
    return [
        ("float32", torch.float32),
        ("float64", torch.float64),
        ("complex64", torch.complex64),
        ("complex128", torch.complex128),
    ]


def _random_dense(shape, dtype, device):
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype == torch.complex64:
        real = torch.randn(shape, dtype=torch.float32, device=device)
        imag = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.complex(real, imag)
    if dtype == torch.complex128:
        real = torch.randn(shape, dtype=torch.float64, device=device)
        imag = torch.randn(shape, dtype=torch.float64, device=device)
        return torch.complex(real, imag)
    raise TypeError(f"unsupported dtype: {dtype}")


def _reference_dtype(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if dtype == torch.float32:
        return torch.float64
    if dtype == torch.complex64:
        return torch.complex128
    return dtype


def _random_coo_mk(M, K, dtype, device):
    denom = max(M * K, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(M, K, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    vals = torch.randn(M, K, dtype=dtype, device=device) * mask.to(dtype=dtype)
    return vals.to_sparse_coo().coalesce()


def _tol(dtype):
    return close_tolerances(dtype)


def _reference(Asp, B, op):
    if op == "non":
        return torch.sparse.mm(Asp, B)
    if op == "trans":
        return torch.sparse.mm(Asp.transpose(0, 1), B)
    if op == "conj":
        return torch.sparse.mm(Asp.conj().transpose(0, 1), B)
    raise ValueError(op)


@pytest.mark.spmm_coo
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize(
    "dtype_name,dtype",
    _value_dtype_cases(),
    ids=[name for name, _dtype in _value_dtype_cases()],
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmm_coo_matches_torch(M, N, K, dtype_name, dtype, index_dtype, op):
    device = torch.device("cuda")
    Asp = _random_coo_mk(M, K, dtype, device)
    indices = Asp.indices()
    data = Asp.values()
    row = indices[0].to(index_dtype).contiguous()
    col = indices[1].to(index_dtype).contiguous()
    b_rows = M if op in ("trans", "conj") else K
    B = _random_dense((b_rows, N), dtype, device)
    ref_dtype = _reference_dtype(dtype)
    ref = _reference(Asp.to(ref_dtype), B.to(ref_dtype), op).to(dtype)
    out = flagsparse_spmm_coo(data, row, col, B, (M, K), op=op)
    rtol, atol = _tol(dtype)
    assert torch.allclose(out.to(ref_dtype), ref.to(ref_dtype), rtol=rtol, atol=atol)


@pytest.mark.spmm_coo
def test_spmm_coo_return_meta_times_transpose_path():
    device = torch.device("cuda")
    data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    row = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    col = torch.tensor([1, 2, 0], dtype=torch.int32, device=device)
    B = torch.randn(3, 4, dtype=torch.float32, device=device)
    out, elapsed_ms, meta = flagsparse_spmm_coo(
        data,
        row,
        col,
        B,
        (3, 3),
        op="trans",
        return_time=True,
        return_meta=True,
    )
    assert out.shape == (3, 4)
    assert meta["op"] == "trans"
    assert meta["symbolic_ms"] >= 0.0
    assert meta["compute_ms"] >= 0.0
    assert elapsed_ms == pytest.approx(meta["op_total_ms"])
    assert meta["op_total_ms"] == pytest.approx(meta["symbolic_ms"] + meta["compute_ms"])


@pytest.mark.spmm_coo
def test_spmm_coo_non_meta_has_zero_symbolic_time():
    device = torch.device("cuda")
    data = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
    row = torch.tensor([0, 1], dtype=torch.int32, device=device)
    col = torch.tensor([0, 1], dtype=torch.int32, device=device)
    B = torch.randn(2, 3, dtype=torch.float32, device=device)
    out, meta = flagsparse_spmm_coo(
        data,
        row,
        col,
        B,
        (2, 2),
        return_meta=True,
    )
    assert out.shape == (2, 3)
    assert meta["op"] == "non"
    assert meta["symbolic_ms"] == 0.0


@pytest.mark.spmm_coo
def test_spmm_coo_dense_layout_col_return_meta():
    device = torch.device("cuda")
    data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    row = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
    col = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
    B = torch.randn(2, 4, dtype=torch.float32, device=device)
    out, meta = flagsparse_spmm_coo(
        data,
        row,
        col,
        B,
        (2, 2),
        dense_layout="col",
        return_meta=True,
    )
    ref = torch.sparse.mm(
        torch.sparse_coo_tensor(
            torch.stack([row.to(torch.int64), col.to(torch.int64)]),
            data,
            size=(2, 2),
            device=device,
        ).coalesce(),
        B,
    )
    rtol, atol = _tol(torch.float32)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol)
    assert out.stride(0) == 1
    assert meta["dense_layout"] == "col"
    assert meta["output_layout"] == "col"


@pytest.mark.spmm_coo
def test_spmm_coo_rejects_invalid_ops_and_shapes():
    device = torch.device("cuda")
    data = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
    row = torch.tensor([0, 1], dtype=torch.int32, device=device)
    col = torch.tensor([0, 1], dtype=torch.int32, device=device)
    B_non = torch.randn(2, 3, dtype=torch.float32, device=device)
    B_bad_trans = torch.randn(4, 3, dtype=torch.float32, device=device)

    with pytest.raises(ValueError):
        flagsparse_spmm_coo(data, row, col, B_non, (2, 4), op="bad")
    with pytest.raises(ValueError):
        flagsparse_spmm_coo(data, row, col, B_non, (2, 4), op="non", transpose=True)
    with pytest.raises(ValueError):
        flagsparse_spmm_coo(data, row, col, B_bad_trans, (2, 4), op="trans", transpose=False)
    with pytest.raises(ValueError):
        flagsparse_spmm_coo(data, row, col, B_bad_trans, (2, 4), op="trans")
    with pytest.raises(ValueError):
        flagsparse_spmm_coo(
            data,
            row,
            col,
            B_non,
            (2, 4),
            op="trans",
            out=torch.empty((2, 3), dtype=torch.float32, device=device),
        )


def test_spmm_coo_sparse_ref_backend_selects_hipsparse_when_direct_supported(monkeypatch):
    monkeypatch.setattr(ast_ops, "_is_rocm_runtime", lambda: True)
    monkeypatch.setattr(
        ast_ops,
        "_hipsparse_spmm_coo_skip_reason",
        lambda value_dtype, index_dtype: None,
    )
    backend, reason = ast_ops._spmm_coo_sparse_ref_backend(
        torch.float32,
        torch.int32,
    )
    assert backend == "hipsparse"
    assert reason is None


def test_spmm_coo_sparse_ref_backend_reports_direct_reason_when_unsupported(monkeypatch):
    monkeypatch.setattr(ast_ops, "_is_rocm_runtime", lambda: True)
    monkeypatch.setattr(
        ast_ops,
        "_hipsparse_spmm_coo_skip_reason",
        lambda value_dtype, index_dtype: "direct hipSPARSE COO SpMM unsupported",
    )
    backend, reason = ast_ops._spmm_coo_sparse_ref_backend(
        torch.float64,
        torch.int64,
    )
    assert backend is None
    assert reason == "direct hipSPARSE COO SpMM unsupported"
