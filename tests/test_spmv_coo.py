"""SpMV COO tests: synthetic + optional .mtx, compare FlagSparse vs active sparse references."""

import argparse
import csv
import glob
import importlib
import math
import os

import flagsparse as fs
import torch


VALUE_DTYPES = [torch.float32, torch.float64]
INDEX_DTYPE = torch.int32
TEST_SIZES = [(512, 512), (1024, 1024), (2048, 2048)]
WARMUP = 10
ITERS = 50
COO_ATOMIC_BLOCK = 256
COO_ATOMIC_WARPS = 4
COO_SEG_BLOCK_INNER = 128
ast_common = importlib.import_module("flagsparse.sparse_operations._common")


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _fmt_ms(v):
    return "N/A" if v is None else f"{v:.4f}"


def _fmt_speedup(other_ms, triton_ms):
    if other_ms is None or triton_ms is None or triton_ms <= 0:
        return "N/A"
    return f"{other_ms / triton_ms:.2f}x"


def _fmt_err(v):
    return "N/A" if v is None else f"{v:.2e}"


def _allclose_error_ratio(actual, reference, atol, rtol):
    if actual.numel() == 0:
        return 0.0
    diff = torch.abs(actual - reference).to(torch.float64)
    tol = (atol + rtol * torch.abs(reference)).to(torch.float64)
    return float(torch.max(diff / tol).item())


def _dense_to_coo(A):
    rows, cols = A.nonzero(as_tuple=True)
    data = A[rows, cols]
    return data, rows, cols


def _sparse_ref_backend_label(backend):
    mapping = {
        None: "N/A",
        "cupy_cusparse": "cuSPARSE",
        "torch": "PyTorch",
    }
    return mapping.get(backend, str(backend))


COO_SEP = "-" * 188
COO_HEADER = (
    f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10}  "
    f"{'Base(ms)':>9} {'Opt(ms)':>9} {'PT(ms)':>9} {'CU(ms)':>9}  "
    f"{'Opt/Base':>8} {'Opt/PT':>8} {'Opt/CU':>8}  "
    f"{'Err(Base)':>10} {'Err(Opt)':>10} {'Status':>6} {'Backend':>12}"
)


def _timed_flagsparse_coo(prepared, x, warmup, iters):
    op = lambda: fs.flagsparse_spmv_coo(
        x=x,
        prepared=prepared,
        return_time=False,
        block_inner=COO_SEG_BLOCK_INNER,
        block_size=COO_ATOMIC_BLOCK,
        num_warps=COO_ATOMIC_WARPS,
    )
    return ast_common._benchmark_cuda_op(op, warmup=warmup, iters=iters)


def _timed_pytorch_coo_reference(data, row, col, x, shape, out_dtype, warmup, iters):
    op = lambda: ast_common._spmv_coo_ref_pytorch(
        data,
        row,
        col,
        x,
        shape,
        out_dtype=out_dtype,
        op="non",
        reference_compute_dtype=True,
    )
    return ast_common._benchmark_cuda_op(op, warmup=warmup, iters=iters)


def _tol_for_dtype(dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-2
    return 1e-12, 1e-10


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
    is_symmetric = mm_symmetry in ("symmetric", "hermitian")
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
        v = 1.0 if is_pattern else (float(parts[2]) if len(parts) >= 3 else 0.0)
        if 0 <= r < n_rows and 0 <= c < n_cols:
            rows_host.append(r)
            cols_host.append(c)
            vals_host.append(v)
            if r != c:
                if is_symmetric and 0 <= c < n_rows and 0 <= r < n_cols:
                    rows_host.append(c)
                    cols_host.append(r)
                    vals_host.append(v)
                elif is_skew and 0 <= c < n_rows and 0 <= r < n_cols:
                    rows_host.append(c)
                    cols_host.append(r)
                    vals_host.append(-v)
    rows = torch.tensor(rows_host, dtype=torch.int64, device=device)
    cols = torch.tensor(cols_host, dtype=torch.int64, device=device)
    vals = torch.tensor(vals_host, dtype=dtype, device=device)
    return vals, rows, cols, (n_rows, n_cols)


def _run_one_case(data, row, col, shape, x, dtype, name):
    atol, rtol = _tol_for_dtype(dtype)
    prepared_seg = fs.prepare_spmv_coo(data, row, col, shape, sort_by_row=True)
    prepared_at = fs.prepare_spmv_coo(data, row, col, shape, sort_by_row=False)
    y_base, base_ms = _timed_flagsparse_coo(prepared_seg, x, WARMUP, ITERS)
    y_opt, opt_ms = _timed_flagsparse_coo(prepared_at, x, WARMUP, ITERS)
    y_ref, ref_meta = ast_common._spmv_coo_reference(
        data,
        row,
        col,
        x,
        shape,
        out_dtype=dtype,
        op="non",
        reference_compute_dtype=True,
        return_metadata=True,
    )
    _pt_values, pt_ms = _timed_pytorch_coo_reference(data, row, col, x, shape, dtype, WARMUP, ITERS)

    err_base = _allclose_error_ratio(y_base, y_ref, atol, rtol)
    err_opt = _allclose_error_ratio(y_opt, y_ref, atol, rtol)
    err_pt = _allclose_error_ratio(y_opt, y_ref, atol, rtol)
    triton_ok_pt = (not math.isnan(err_pt)) and err_pt <= 1.0

    sparse_ref = ast_common._benchmark_spmv_coo_sparse_ref(
        data,
        row,
        col,
        x,
        shape,
        warmup=WARMUP,
        iters=ITERS,
        op="non",
    )
    sparse_ref_backend = sparse_ref["backend"]
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
    return {
        "matrix": name,
        "n_rows": int(shape[0]),
        "n_cols": int(shape[1]),
        "nnz": int(data.numel()),
        "base_ms": base_ms,
        "opt_ms": opt_ms,
        "triton_ms": opt_ms,
        "cusparse_ms": cu_ms,
        "pytorch_ms": pt_ms,
        "csc_ms": None,
        "pt_status": "PASS" if triton_ok_pt else "FAIL",
        "cu_status": "PASS" if triton_ok_cu else "N/A",
        "status": status,
        "err_base": err_base,
        "err_opt": err_opt,
        "err_pt": err_pt,
        "err_cu": err_cu,
        "sparse_ref_backend": ref_meta["backend"] if sparse_ref_backend is None else sparse_ref_backend,
        "sparse_ref_reason": sparse_ref["reason"] if sparse_ref_backend is None else None,
    }


def _print_row(row):
    name = row["matrix"][:27]
    backend_label = _sparse_ref_backend_label(row.get("sparse_ref_backend"))
    print(
        f"{name:<28} {row['n_rows']:>7} {row['n_cols']:>7} {row['nnz']:>10}  "
        f"{_fmt_ms(row['base_ms']):>9} {_fmt_ms(row['opt_ms']):>9} {_fmt_ms(row['pytorch_ms']):>9} {_fmt_ms(row['cusparse_ms']):>9}  "
        f"{_fmt_speedup(row['base_ms'], row['opt_ms']):>8} {_fmt_speedup(row['pytorch_ms'], row['opt_ms']):>8} {_fmt_speedup(row['cusparse_ms'], row['opt_ms']):>8}  "
        f"{_fmt_err(row['err_base']):>10} {_fmt_err(row['err_opt']):>10} {row['status']:>6} {backend_label:>12}"
    )


def run_synthetic():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a GPU-enabled system.")
        return
    device = torch.device("cuda")
    print("=" * 188)
    print("FLAGSPARSE SpMV COO BENCHMARK (synthetic dense -> COO)")
    print("=" * 188)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {WARMUP} | Iters: {ITERS}")
    print("References: PyTorch sparse.mm + active sparse reference backend.")
    print()

    for dtype in VALUE_DTYPES:
        print(COO_SEP)
        print(f"dtype: {_dtype_name(dtype)}  index_dtype: int32")
        print(COO_SEP)
        print(COO_HEADER)
        print(COO_SEP)
        for m, n in TEST_SIZES:
            A = torch.randn(m, n, dtype=dtype, device=device)
            A *= torch.rand_like(A) < 0.1
            data, row, col = _dense_to_coo(A)
            x = torch.randn(n, dtype=dtype, device=device)
            row_out = _run_one_case(data, row.to(INDEX_DTYPE), col.to(INDEX_DTYPE), (m, n), x, dtype, f"{m}x{n}")
            _print_row(row_out)
        print(COO_SEP)
        print()


def run_all_dtypes_coo_csv(mtx_paths, csv_path):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    rows_out = []
    for dtype in VALUE_DTYPES:
        print("=" * 188)
        print(f"Value dtype: {_dtype_name(dtype)}  |  Index dtype: int32")
        print("Input: MatrixMarket -> COO. FlagSparse stays COO; references come from shared sparse-ref helpers.")
        print(COO_SEP)
        print(COO_HEADER)
        print(COO_SEP)
        for path in mtx_paths:
            try:
                data, row, col, shape = _load_mtx_to_coo_torch(path, dtype=dtype, device=device)
                x = torch.randn(shape[1], dtype=dtype, device=device)
                row_out = _run_one_case(
                    data,
                    row.to(INDEX_DTYPE),
                    col.to(INDEX_DTYPE),
                    shape,
                    x,
                    dtype,
                    os.path.basename(path),
                )
                rows_out.append(row_out)
                _print_row(row_out)
            except Exception as exc:
                print(f"{os.path.basename(path):<28} {'ERR':>7} {'ERR':>7} {'ERR':>10}  "
                      f"{_fmt_ms(None):>9} {_fmt_ms(None):>9} {_fmt_ms(None):>9} {_fmt_ms(None):>9}  "
                      f"{'N/A':>8} {'N/A':>8} {'N/A':>8}  "
                      f"{_fmt_err(None):>10} {_fmt_err(None):>10} {'ERROR':>6} {'N/A':>12}")
                print(f"  ERROR: {exc}")
                rows_out.append(
                    {
                        "matrix": os.path.basename(path),
                        "value_dtype": _dtype_name(dtype),
                        "index_dtype": "torch.int32",
                        "n_rows": "ERR",
                        "n_cols": "ERR",
                        "nnz": "ERR",
                        "base_ms": None,
                        "opt_ms": None,
                        "triton_ms": None,
                        "cusparse_ms": None,
                        "pytorch_ms": None,
                        "csc_ms": None,
                        "pt_status": "N/A",
                        "cu_status": "N/A",
                        "status": "ERROR",
                        "err_base": None,
                        "err_opt": None,
                        "err_pt": None,
                        "err_cu": None,
                        "sparse_ref_backend": None,
                        "sparse_ref_reason": str(exc),
                    }
                )
        print(COO_SEP)

    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "n_rows",
        "n_cols",
        "nnz",
        "base_ms",
        "opt_ms",
        "triton_ms",
        "cusparse_ms",
        "pytorch_ms",
        "csc_ms",
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
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_out:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
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
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic()
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
        run_all_dtypes_coo_csv(paths, args.csv_coo)
        return

    print("Use --synthetic or --csv-coo to run COO SpMV tests.")


if __name__ == "__main__":
    main()
