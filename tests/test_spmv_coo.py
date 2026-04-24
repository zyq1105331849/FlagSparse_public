"""SpMV COO tests: dtype/index/op grids, synthetic + optional .mtx, CSV export."""

import argparse
import csv
import glob
import importlib
import math
import os

import flagsparse as fs
import flagsparse.sparse_operations.spmv_coo as spmv_coo_mod
import torch

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
except Exception:
    cp = None
    cpx_sparse = None


VALUE_DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]
INDEX_DTYPES = [torch.int32, torch.int64]
OPS = ["non", "trans", "conj"]
TEST_SIZES = [(512, 512), (1024, 1024), (2048, 2048)]
WARMUP = 10
ITERS = 50

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
INDEX_DTYPE_MAP = {
    "int32": torch.int32,
    "int64": torch.int64,
}
OP_NAMES = ("non", "trans", "conj")
COO_ATOMIC_BLOCK = 256
COO_ATOMIC_WARPS = 4
COO_SEG_BLOCK_INNER = 128
ast_common = importlib.import_module("flagsparse.sparse_operations._common")


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _parse_csv_tokens(value, mapping, name):
    tokens = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"{name} must not be empty")
    unknown = [token for token in tokens if token not in mapping]
    if unknown:
        allowed = ", ".join(mapping)
        raise ValueError(f"unsupported {name}: {', '.join(unknown)}; allowed: {allowed}")
    return [mapping[token] for token in tokens]


def _parse_ops(value):
    tokens = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("--ops must not be empty")
    unknown = [token for token in tokens if token not in OP_NAMES]
    if unknown:
        raise ValueError(
            f"unsupported op: {', '.join(unknown)}; allowed: {', '.join(OP_NAMES)}"
        )
    return tokens


def _random_values(shape, dtype, device):
    if dtype in (torch.float32, torch.float64):
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


def _fmt_ms(v):
    return "N/A" if v is None else f"{v:.4f}"


def _fmt_err(v):
    return "N/A" if v is None else f"{v:.2e}"


def _spd(num, den):
    if num is None or den is None or den <= 0:
        return "N/A"
    return f"{num / den:.2f}x"


def _speedup_ratio(other_ms, triton_ms):
    if other_ms is None or triton_ms is None or triton_ms <= 0:
        return None
    return other_ms / triton_ms


def _status_str(ok, available):
    if not available:
        return "N/A"
    return "PASS" if ok else "FAIL"


def _allclose_error_ratio(actual, reference, atol, rtol):
    if actual.numel() == 0:
        return 0.0
    diff = torch.abs(actual - reference).to(torch.float64)
    tol = (atol + rtol * torch.abs(reference)).to(torch.float64)
    return float(torch.max(diff / tol).item())


def _reference_dtype(dtype):
    if dtype == torch.float32:
        return torch.float64
    if dtype == torch.complex64:
        return torch.complex128
    return dtype


def _tol_for_dtype(dtype):
    if dtype in (torch.float32, torch.complex64):
        return 1e-4, 1e-2
    return 1e-12, 1e-10


def _op_transposes(op):
    return op in ("trans", "conj")


def _out_size_for_op(shape, op):
    n_rows, n_cols = shape
    return n_cols if _op_transposes(op) else n_rows


def _x_size_for_op(shape, op):
    n_rows, n_cols = shape
    return n_rows if _op_transposes(op) else n_cols


def _apply_coo_op(data, row, col, shape, op):
    if op == "non":
        return data, row, col, shape
    n_rows, n_cols = shape
    data_op = data
    if op == "conj":
        data_op = data.conj()
        if hasattr(data_op, "resolve_conj"):
            data_op = data_op.resolve_conj()
    elif op != "trans":
        raise ValueError(f"unsupported op: {op}")
    return data_op, col, row, (n_cols, n_rows)


def _apply_torch_sparse_op(matrix, x_2d, op):
    if op == "non":
        return torch.sparse.mm(matrix, x_2d).squeeze(1)
    if op == "trans":
        return torch.sparse.mm(matrix.transpose(0, 1), x_2d).squeeze(1)
    if op == "conj":
        return torch.sparse.mm(matrix.conj().transpose(0, 1), x_2d).squeeze(1)
    raise ValueError(f"unsupported op: {op}")


def _cupy_sparse_op_matrix(matrix, op):
    if op == "non":
        return matrix
    if op == "trans":
        return matrix.T
    if op == "conj":
        matrix_conj = matrix.conj() if hasattr(matrix, "conj") else matrix.conjugate()
        return matrix_conj.T
    raise ValueError(f"unsupported op: {op}")


def _apply_cupy_sparse_op(matrix, x, op):
    return _cupy_sparse_op_matrix(matrix, op) @ x


def _pytorch_coo_reference(data, row, col, x, shape, out_dtype, op="non"):
    data, row, col, shape = _apply_coo_op(data, row, col, shape, op)
    ref_dtype = _reference_dtype(out_dtype)
    data_ref = data.to(ref_dtype)
    x_ref = x.to(ref_dtype)
    coo_ref = torch.sparse_coo_tensor(
        torch.stack([row.to(torch.int64), col.to(torch.int64)]),
        data_ref,
        shape,
        device=data.device,
    ).coalesce()
    y_ref = torch.sparse.mm(coo_ref, x_ref.unsqueeze(1)).squeeze(1)
    return y_ref.to(out_dtype) if ref_dtype != out_dtype else y_ref


def _cupy_coo_reference(data, row, col, x, shape, out_dtype, op="non"):
    data, row, col, shape = _apply_coo_op(data, row, col, shape, op)
    ref_dtype = _reference_dtype(out_dtype)
    data_ref = data.to(ref_dtype)
    x_ref = x.to(ref_dtype)
    data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data_ref))
    row_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(row.to(torch.int64)))
    col_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(col.to(torch.int64)))
    x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x_ref))
    matrix = cpx_sparse.coo_matrix((data_cp, (row_cp, col_cp)), shape=shape)
    y_ref = _apply_cupy_sparse_op(matrix, x_cp, op)
    y_ref_t = torch.utils.dlpack.from_dlpack(y_ref.toDlpack())
    return y_ref_t.to(out_dtype) if ref_dtype != out_dtype else y_ref_t


def _dense_to_coo(A):
    rows, cols = A.nonzero(as_tuple=True)
    data = A[rows, cols]
    return data, rows, cols


def _sparse_ref_backend_label(backend):
    mapping = {
        None: "N/A",
        "hipsparse": "hipSPARSE",
        "cupy_cusparse": "cuSPARSE",
        "torch": "PyTorch",
    }
    return mapping.get(backend, str(backend))


COO_SEP = "-" * 208
COO_HEADER = (
    f"{'Matrix':<28} {'Op':>5} {'Out':>7} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10}  "
    f"{'Base(ms)':>9} {'Opt(ms)':>9} {'PT(ms)':>9} {'CU(ms)':>9}  "
    f"{'Opt/Base':>8} {'Opt/PT':>8} {'Opt/CU':>8}  "
    f"{'Err(Base)':>10} {'Err(Opt)':>10} {'Status':>6} {'Backend':>12}"
)


def _timed_flagsparse_coo(
    prepared,
    data,
    row,
    col,
    x,
    shape,
    sort_by_row,
    op,
    warmup,
    iters,
):
    op_name = op.lower()
    spmv_op = lambda: _run_flagsparse_coo_timed_op(
        prepared,
        data,
        row,
        col,
        x,
        shape,
        sort_by_row,
        op_name,
    )
    y = spmv_op()
    torch.cuda.synchronize()
    for _ in range(warmup):
        spmv_op()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(True)
    e1 = torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        y = spmv_op()
    e1.record()
    torch.cuda.synchronize()
    return y, e0.elapsed_time(e1) / iters


def _run_flagsparse_coo_timed_op(
    prepared,
    data,
    row,
    col,
    x,
    shape,
    sort_by_row,
    op,
):
    if op == "non":
        return fs.flagsparse_spmv_coo(
            x=x,
            prepared=prepared,
            return_time=False,
            block_inner=COO_SEG_BLOCK_INNER,
            block_size=COO_ATOMIC_BLOCK,
            num_warps=COO_ATOMIC_WARPS,
        )
    data_op, row_op, col_op, shape_op = _apply_coo_op(data, row, col, shape, op)
    timed_prepared = spmv_coo_mod.prepare_spmv_coo(
        data_op,
        row_op,
        col_op,
        shape_op,
        sort_by_row=sort_by_row,
        op="non",
    )
    return fs.flagsparse_spmv_coo(
        x=x,
        prepared=timed_prepared,
        return_time=False,
        block_inner=COO_SEG_BLOCK_INNER,
        block_size=COO_ATOMIC_BLOCK,
        num_warps=COO_ATOMIC_WARPS,
    )


def _time_pytorch_coo(data, row, col, x, shape, op, warmup, iters):
    ref_dtype = _reference_dtype(data.dtype)
    data_ref = data.to(ref_dtype)
    x_ref_2d = x.to(ref_dtype).unsqueeze(1)
    coo = torch.sparse_coo_tensor(
        torch.stack([row.to(torch.int64), col.to(torch.int64)]),
        data_ref,
        shape,
        device=data.device,
    ).coalesce()
    if data.numel() == 0:
        return 0.0
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = _apply_torch_sparse_op(coo, x_ref_2d, op)
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(True)
    e1 = torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        _ = _apply_torch_sparse_op(coo, x_ref_2d, op)
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def _mtx_value_for_dtype(raw_value, dtype):
    if dtype in (torch.complex64, torch.complex128):
        if isinstance(raw_value, complex):
            return raw_value
        return complex(float(raw_value), 0.5 * float(raw_value))
    if isinstance(raw_value, complex):
        return float(raw_value.real)
    return float(raw_value)


def _load_mtx_to_coo_torch(file_path, dtype=torch.float32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    mm_field = "real"
    mm_symmetry = "general"
    data_lines = []
    header_info = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("%%MatrixMarket"):
            tokens = stripped.split()
            if len(tokens) >= 5:
                mm_field = tokens[3].lower()
                mm_symmetry = tokens[4].lower()
            continue
        if stripped.startswith("%"):
            continue
        if not header_info and stripped:
            parts = stripped.split()
            n_rows = int(parts[0])
            n_cols = int(parts[1])
            nnz = int(parts[2]) if len(parts) > 2 else 0
            header_info = (n_rows, n_cols, nnz)
            continue
        if stripped:
            data_lines.append(stripped)
    if header_info is None:
        raise ValueError(f"Cannot parse .mtx header: {file_path}")
    n_rows, n_cols, nnz = header_info

    is_pattern = mm_field == "pattern"
    is_complex_field = mm_field == "complex"
    is_symmetric = mm_symmetry == "symmetric"
    is_hermitian = mm_symmetry == "hermitian"
    is_skew = mm_symmetry == "skew-symmetric"

    rows_host = []
    cols_host = []
    vals_host = []
    for line in data_lines[:nnz]:
        parts = line.split()
        if len(parts) < 2:
            continue
        r = int(parts[0]) - 1
        c = int(parts[1]) - 1
        if is_pattern:
            raw = 1.0
        elif is_complex_field:
            real = float(parts[2]) if len(parts) >= 3 else 0.0
            imag = float(parts[3]) if len(parts) >= 4 else 0.0
            raw = complex(real, imag)
        else:
            raw = float(parts[2]) if len(parts) >= 3 else 0.0
        v = _mtx_value_for_dtype(raw, dtype)
        if 0 <= r < n_rows and 0 <= c < n_cols:
            rows_host.append(r)
            cols_host.append(c)
            vals_host.append(v)
            if r != c:
                if is_symmetric and 0 <= c < n_rows and 0 <= r < n_cols:
                    rows_host.append(c)
                    cols_host.append(r)
                    vals_host.append(v)
                elif is_hermitian and 0 <= c < n_rows and 0 <= r < n_cols:
                    rows_host.append(c)
                    cols_host.append(r)
                    vals_host.append(v.conjugate() if isinstance(v, complex) else v)
                elif is_skew and 0 <= c < n_rows and 0 <= r < n_cols:
                    rows_host.append(c)
                    cols_host.append(r)
                    vals_host.append(-v)
    rows = torch.tensor(rows_host, dtype=torch.int64, device=device)
    cols = torch.tensor(cols_host, dtype=torch.int64, device=device)
    vals = torch.tensor(vals_host, dtype=dtype, device=device)
    return vals, rows, cols, (n_rows, n_cols)


def _run_one_coo_case(
    data,
    row,
    col,
    shape,
    dtype,
    index_dtype,
    op,
    matrix_name,
    warmup,
    iters,
):
    row = row.to(index_dtype).contiguous()
    col = col.to(index_dtype).contiguous()
    x = _random_values((_x_size_for_op(shape, op),), dtype, data.device)
    atol, rtol = _tol_for_dtype(dtype)
    prepared_seg = fs.prepare_spmv_coo(
        data, row, col, shape, sort_by_row=True, op=op
    )
    prepared_at = fs.prepare_spmv_coo(
        data, row, col, shape, sort_by_row=False, op=op
    )
    y_base, base_ms = _timed_flagsparse_coo(
        prepared_seg,
        data,
        row,
        col,
        x,
        shape,
        True,
        op,
        warmup,
        iters,
    )
    y_opt, opt_ms = _timed_flagsparse_coo(
        prepared_at,
        data,
        row,
        col,
        x,
        shape,
        False,
        op,
        warmup,
        iters,
    )
    y_ref = _pytorch_coo_reference(data, row, col, x, shape, dtype, op=op)
    err_base = _allclose_error_ratio(y_base, y_ref, atol, rtol)
    err_opt = _allclose_error_ratio(y_opt, y_ref, atol, rtol)
    pt_ms = _time_pytorch_coo(data, row, col, x, shape, op, warmup, iters)
    y_pt = _pytorch_coo_reference(data, row, col, x, shape, dtype, op=op)
    err_pt = _allclose_error_ratio(y_opt, y_pt, atol, rtol)
    triton_ok_pt = (not math.isnan(err_pt)) and err_pt <= 1.0

    sparse_ref = ast_common._benchmark_spmv_coo_sparse_ref(
        data,
        row,
        col,
        x,
        shape,
        warmup=warmup,
        iters=iters,
        op=op,
    )
    sparse_ref_backend = sparse_ref["backend"]
    sparse_ref_reason = sparse_ref["reason"]
    cu_ms = sparse_ref["ms"]
    err_cu = None
    triton_ok_cu = False
    if sparse_ref_backend is not None:
        err_cu = _allclose_error_ratio(y_opt, sparse_ref["values"], atol, rtol)
        triton_ok_cu = (not math.isnan(err_cu)) and err_cu <= 1.0

    status = (
        "PASS"
        if (
            (not math.isnan(err_base))
            and (not math.isnan(err_opt))
            and err_base <= 1.0
            and err_opt <= 1.0
        )
        else "FAIL"
    )
    n_rows, n_cols = shape
    return {
        "matrix": matrix_name,
        "value_dtype": _dtype_name(dtype),
        "index_dtype": _dtype_name(index_dtype),
        "op": op,
        "out_size": _out_size_for_op(shape, op),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": int(data.numel()),
        "base_ms": base_ms,
        "opt_ms": opt_ms,
        "triton_ms": opt_ms,
        "cusparse_ms": cu_ms,
        "pytorch_ms": pt_ms,
        "csc_ms": None,
        "triton_speedup_vs_cusparse": _speedup_ratio(cu_ms, opt_ms),
        "triton_speedup_vs_pytorch": _speedup_ratio(pt_ms, opt_ms),
        "pt_status": _status_str(triton_ok_pt, err_pt is not None),
        "cu_status": _status_str(triton_ok_cu, err_cu is not None),
        "status": status,
        "err_base": err_base,
        "err_opt": err_opt,
        "err_pt": err_pt,
        "err_cu": err_cu,
        "sparse_ref_backend": sparse_ref_backend,
        "sparse_ref_reason": sparse_ref_reason,
    }


def _print_coo_result(row_out):
    name = str(row_out["matrix"])[:27]
    if len(str(row_out["matrix"])) > 27:
        name += "..."
    backend_label = _sparse_ref_backend_label(row_out.get("sparse_ref_backend"))
    print(
        f"{name:<28} {row_out['op']:>5} {row_out['out_size']:>7} "
        f"{row_out['n_rows']:>7} {row_out['n_cols']:>7} {row_out['nnz']:>10}  "
        f"{_fmt_ms(row_out['base_ms']):>9} {_fmt_ms(row_out['opt_ms']):>9} "
        f"{_fmt_ms(row_out['pytorch_ms']):>9} {_fmt_ms(row_out['cusparse_ms']):>9}  "
        f"{_spd(row_out['base_ms'], row_out['opt_ms']):>8} "
        f"{_spd(row_out['pytorch_ms'], row_out['opt_ms']):>8} "
        f"{_spd(row_out['cusparse_ms'], row_out['opt_ms']):>8}  "
        f"{_fmt_err(row_out['err_base']):>10} {_fmt_err(row_out['err_opt']):>10} "
        f"{row_out['status']:>6} {backend_label:>12}"
    )
    if row_out.get("sparse_ref_reason") and row_out.get("cusparse_ms") is None:
        print(f"  sparse-ref: {row_out['sparse_ref_reason']}")


def _error_row(path, dtype, index_dtype, op, reason):
    return {
        "matrix": os.path.basename(path),
        "value_dtype": _dtype_name(dtype),
        "index_dtype": _dtype_name(index_dtype),
        "op": op,
        "out_size": "ERR",
        "n_rows": "ERR",
        "n_cols": "ERR",
        "nnz": "ERR",
        "base_ms": None,
        "opt_ms": None,
        "triton_ms": None,
        "cusparse_ms": None,
        "pytorch_ms": None,
        "csc_ms": None,
        "triton_speedup_vs_cusparse": None,
        "triton_speedup_vs_pytorch": None,
        "pt_status": "N/A",
        "cu_status": "N/A",
        "status": "ERROR",
        "err_base": None,
        "err_opt": None,
        "err_pt": None,
        "err_cu": None,
        "sparse_ref_backend": None,
        "sparse_ref_reason": reason,
    }


def run_synthetic(value_dtypes=None, index_dtypes=None, ops=None):
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a GPU-enabled system.")
        return
    device = torch.device("cuda")
    print("=" * 208)
    print("FLAGSPARSE SpMV COO BENCHMARK (synthetic dense -> COO)")
    print("=" * 208)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {WARMUP} | Iters: {ITERS}")
    print("Correctness reference stays local PyTorch COO; sparse backend is active cuSPARSE/hipSPARSE.")
    print()

    value_dtypes = VALUE_DTYPES if value_dtypes is None else value_dtypes
    index_dtypes = INDEX_DTYPES if index_dtypes is None else index_dtypes
    ops = OPS if ops is None else ops

    for dtype in value_dtypes:
        for index_dtype in index_dtypes:
            for op in ops:
                print(COO_SEP)
                print(
                    f"dtype: {_dtype_name(dtype)}  index_dtype: {_dtype_name(index_dtype)}  op: {op}"
                )
                print(COO_SEP)
                print(
                    "FlagSparse: prepare_spmv_coo + Triton COO SpMV (no CSR). "
                    "Base(ms) = row-run (seg) kernel; Opt(ms) = NNZ atomic kernel."
                )
                print(COO_SEP)
                print(COO_HEADER)
                print(COO_SEP)
                for m, n in TEST_SIZES:
                    A = _random_values((m, n), dtype, device)
                    A *= (torch.rand(m, n, device=device) < 0.1).to(dtype=dtype)
                    data, row, col = _dense_to_coo(A)
                    result = _run_one_coo_case(
                        data=data,
                        row=row,
                        col=col,
                        shape=(m, n),
                        dtype=dtype,
                        index_dtype=index_dtype,
                        op=op,
                        matrix_name=f"{m}x{n}",
                        warmup=WARMUP,
                        iters=ITERS,
                    )
                    _print_coo_result(result)
                print(COO_SEP)
                print()


def run_all_dtypes_coo_csv(
    mtx_paths,
    csv_path,
    value_dtypes=None,
    index_dtypes=None,
    ops=None,
):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    rows_out = []
    value_dtypes = VALUE_DTYPES if value_dtypes is None else value_dtypes
    index_dtypes = INDEX_DTYPES if index_dtypes is None else index_dtypes
    ops = OPS if ops is None else ops

    print("=" * 208)
    print("Input: MatrixMarket -> COO. Correctness reference stays local PyTorch COO.")
    print("Sparse backend column uses active cuSPARSE/hipSPARSE timing where available.")
    print("=" * 208)
    for dtype in value_dtypes:
        for index_dtype in index_dtypes:
            for op in ops:
                print(COO_SEP)
                print(
                    f"Value dtype: {_dtype_name(dtype)} | Index dtype: {_dtype_name(index_dtype)} | op: {op}"
                )
                print(COO_SEP)
                print(COO_HEADER)
                print(COO_SEP)
                for path in mtx_paths:
                    try:
                        data, row, col, shape = _load_mtx_to_coo_torch(
                            path, dtype=dtype, device=device
                        )
                        result = _run_one_coo_case(
                            data=data,
                            row=row,
                            col=col,
                            shape=shape,
                            dtype=dtype,
                            index_dtype=index_dtype,
                            op=op,
                            matrix_name=os.path.basename(path),
                            warmup=WARMUP,
                            iters=ITERS,
                        )
                        rows_out.append(result)
                        _print_coo_result(result)
                    except Exception as exc:
                        row_out = _error_row(path, dtype, index_dtype, op, str(exc))
                        rows_out.append(row_out)
                        _print_coo_result(row_out)
                        print(f"  ERROR: {exc}")
                print(COO_SEP)

    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "op",
        "out_size",
        "n_rows",
        "n_cols",
        "nnz",
        "base_ms",
        "opt_ms",
        "triton_ms",
        "cusparse_ms",
        "pytorch_ms",
        "csc_ms",
        "triton_speedup_vs_cusparse",
        "triton_speedup_vs_pytorch",
        "pt_status",
        "cu_status",
        "status",
        "err_base",
        "err_opt",
        "err_pt",
        "err_cu",
        "sparse_ref_backend",
        "sparse_ref_reason",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_out in rows_out:
            writer.writerow(row_out)
    print(f"Wrote {len(rows_out)} rows to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SpMV COO test: synthetic dense->COO and optional .mtx, export CSV."
    )
    parser.add_argument(
        "mtx",
        nargs="*",
        help=".mtx file path(s), or directory(ies) to glob for *.mtx",
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Run synthetic dense->COO tests"
    )
    parser.add_argument(
        "--csv-coo",
        type=str,
        default=None,
        metavar="FILE",
        help="Run all dtypes on given .mtx and export COO SpMV results to CSV",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        default="float32,float64,complex64,complex128",
        help="Comma-separated value dtypes: float32,float64,complex64,complex128",
    )
    parser.add_argument(
        "--index-dtypes",
        type=str,
        default="int32,int64",
        help="Comma-separated index dtypes: int32,int64",
    )
    parser.add_argument(
        "--ops",
        type=str,
        default="non,trans,conj",
        help="Comma-separated matrix ops: non,trans,conj",
    )
    args = parser.parse_args()
    value_dtypes = _parse_csv_tokens(args.dtypes, DTYPE_MAP, "--dtypes")
    index_dtypes = _parse_csv_tokens(
        args.index_dtypes, INDEX_DTYPE_MAP, "--index-dtypes"
    )
    ops = _parse_ops(args.ops)

    if args.synthetic:
        run_synthetic(value_dtypes=value_dtypes, index_dtypes=index_dtypes, ops=ops)
        return

    paths = []
    for p in args.mtx:
        if os.path.isfile(p) and p.endswith(".mtx"):
            paths.append(p)
        elif os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.mtx"))))
    if args.csv_coo:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-coo")
            return
        run_all_dtypes_coo_csv(
            paths,
            args.csv_coo,
            value_dtypes=value_dtypes,
            index_dtypes=index_dtypes,
            ops=ops,
        )
        return

    print("Use --synthetic or --csv-coo to run COO SpMV tests.")


if __name__ == "__main__":
    main()
