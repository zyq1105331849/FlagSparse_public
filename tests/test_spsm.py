"""SpSM tests: synthetic and optional .mtx CSV export (CSR/COO)."""

import argparse
import csv
import glob
import os

import torch

import flagsparse as fs

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
    from cupyx.scipy.sparse.linalg import spsolve_triangular as cpx_spsolve_triangular
except Exception:
    cp = None
    cpx_sparse = None
    cpx_spsolve_triangular = None


FORMATS = ("csr", "coo")
VALUE_DTYPES = (torch.float32, torch.float64)
INDEX_DTYPES = [torch.int32]
CSV_VALUE_DTYPES = [torch.float32, torch.float64]
CSV_INDEX_DTYPES = [torch.int32]
WARMUP = 10
ITERS = 50


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _tol(dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-3
    return 1e-12, 1e-10


def _fmt_ms(v):
    return "N/A" if v is None else f"{v:.4f}"


def _fmt_speedup(other_ms, fs_ms):
    if other_ms is None or fs_ms is None or fs_ms <= 0:
        return "N/A"
    return f"{other_ms / fs_ms:.2f}x"


def _fmt_err(v):
    return "N/A" if v is None else f"{v:.2e}"


def _build_triangular_case(n=512, n_rhs=32, value_dtype=torch.float32):
    device = torch.device("cuda")
    A = torch.tril(torch.randn((n, n), dtype=value_dtype, device=device) * 0.02)
    diag = torch.rand((n,), dtype=value_dtype, device=device) + 2.0
    A = A + torch.diag(diag)
    coo = A.to_sparse().coalesce()
    row = coo.indices()[0].to(torch.int64)
    col = coo.indices()[1].to(torch.int64)
    data = coo.values().to(value_dtype)
    _, order = torch.sort(row * n + col)
    row = row[order]
    col = col[order]
    data = data[order]
    nnz_per_row = torch.bincount(row, minlength=n)
    indptr = torch.zeros(n + 1, dtype=torch.int64, device=device)
    indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    B = torch.randn((n, n_rhs), dtype=value_dtype, device=device).contiguous()
    return data, row, col, indptr, B, (n, n)


def _csr_to_coo(data, indices, indptr, shape):
    n_rows = int(shape[0])
    row = torch.repeat_interleave(
        torch.arange(n_rows, device=data.device, dtype=torch.int64),
        indptr[1:] - indptr[:-1],
    )
    col = indices.to(torch.int64)
    return data, row, col


def _load_mtx_to_csr_torch(file_path, dtype=torch.float32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data_lines = []
    header_info = None
    mm_field = "real"
    mm_symmetry = "general"
    for line in lines:
        line = line.strip()
        if line.startswith("%%MatrixMarket"):
            parts = line.split()
            if len(parts) >= 5:
                mm_field = parts[3].lower()
                mm_symmetry = parts[4].lower()
            continue
        if line.startswith("%"):
            continue
        if not header_info and line:
            parts = line.split()
            n_rows = int(parts[0])
            n_cols = int(parts[1])
            nnz = int(parts[2]) if len(parts) > 2 else 0
            header_info = (n_rows, n_cols, nnz)
            continue
        if line:
            data_lines.append(line)
    if header_info is None:
        raise ValueError(f"Cannot parse .mtx header: {file_path}")
    n_rows, n_cols, nnz = header_info
    if n_rows != n_cols:
        raise ValueError("SpSM requires square matrices")

    row_maps = [dict() for _ in range(n_rows)]

    def _accum(r, c, v):
        row = row_maps[r]
        row[c] = row.get(c, 0.0) + v

    for line in data_lines[:nnz]:
        parts = line.split()
        if len(parts) < 2:
            continue
        r = int(parts[0]) - 1
        c = int(parts[1]) - 1
        if len(parts) >= 3:
            v = float(parts[2])
        elif mm_field == "pattern":
            v = 1.0
        else:
            continue
        _accum(r, c, v)
        if mm_symmetry in ("symmetric", "hermitian") and r != c:
            _accum(c, r, v)
        elif mm_symmetry == "skew-symmetric" and r != c:
            _accum(c, r, -v)

    # Force lower-triangular + strong diagonal so triangular solve is well-defined.
    for r in range(n_rows):
        row = row_maps[r]
        lower_row = {}
        off_abs_sum = 0.0
        for c, v in row.items():
            if c < r:
                lower_row[c] = lower_row.get(c, 0.0) + v
                off_abs_sum += abs(v)
        lower_row[r] = off_abs_sum + 2.0
        row_maps[r] = lower_row

    cols_s = []
    vals_s = []
    indptr_list = [0]
    for r in range(n_rows):
        row = row_maps[r]
        for c in sorted(row.keys()):
            cols_s.append(c)
            vals_s.append(row[c])
        indptr_list.append(len(cols_s))

    data = torch.tensor(vals_s, dtype=dtype, device=device)
    indices = torch.tensor(cols_s, dtype=torch.int64, device=device)
    indptr = torch.tensor(indptr_list, dtype=torch.int64, device=device)
    return data, indices, indptr, (n_rows, n_cols)


def _cupy_spsm_ref(data, row, col, indptr, B, shape, fmt="csr"):
    if cp is None or cpx_sparse is None or cpx_spsolve_triangular is None:
        return None, None, "cupy/cusparse unavailable"
    try:
        data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data.contiguous()))
        B_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(B.contiguous()))
        if fmt == "coo":
            row_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(row.to(torch.int64).contiguous()))
            col_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(col.to(torch.int64).contiguous()))
            A_cp = cpx_sparse.coo_matrix((data_cp, (row_cp, col_cp)), shape=shape)
        else:
            idx_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(col.to(torch.int64).contiguous()))
            ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr.contiguous()))
            A_cp = cpx_sparse.csr_matrix((data_cp, idx_cp, ptr_cp), shape=shape)
        for _ in range(max(0, int(WARMUP))):
            _ = cpx_spsolve_triangular(A_cp, B_cp, lower=True, unit_diagonal=False)
        cp.cuda.runtime.deviceSynchronize()
        c0 = cp.cuda.Event()
        c1 = cp.cuda.Event()
        c0.record()
        for _ in range(max(1, int(ITERS))):
            X_cp = cpx_spsolve_triangular(A_cp, B_cp, lower=True, unit_diagonal=False)
        c1.record()
        c1.synchronize()
        ms = cp.cuda.get_elapsed_time(c0, c1) / max(1, int(ITERS))
        X_t = torch.utils.dlpack.from_dlpack(X_cp.toDlpack()).to(B.dtype)
        return X_t, ms, None
    except Exception as exc:
        return None, None, str(exc)


def _run_one_spsm_case(data, indices, indptr, shape, value_dtype, index_dtype, n_rhs, fmt):
    n_rows, _ = shape
    B = torch.randn((n_rows, n_rhs), dtype=value_dtype, device=data.device).contiguous()
    atol, rtol = _tol(value_dtype)

    if fmt == "csr":
        X_fs, fs_ms = fs.flagsparse_spsm_csr(
            data=data,
            indices=indices.to(index_dtype),
            indptr=indptr,
            B=B,
            shape=shape,
            alpha=1.0,
            lower=True,
            unit_diagonal=False,
            opA="NON_TRANS",
            opB="NON_TRANS",
            major="row",
            return_time=True,
        )
        data_c, row_c, col_c = _csr_to_coo(data, indices, indptr, shape)
    else:
        data_c, row_c, col_c = _csr_to_coo(data, indices, indptr, shape)
        X_fs, fs_ms = fs.flagsparse_spsm_coo(
            data=data_c,
            row=row_c.to(index_dtype),
            col=col_c.to(index_dtype),
            B=B,
            shape=shape,
            alpha=1.0,
            lower=True,
            unit_diagonal=False,
            opA="NON_TRANS",
            opB="NON_TRANS",
            major="row",
            return_time=True,
        )

    X_cu, cu_ms, cu_reason = _cupy_spsm_ref(
        data, row_c, col_c, indptr, B, shape, fmt=fmt
    )
    ok_cu = None
    err_cu = None
    if X_cu is not None:
        ok_cu = torch.allclose(X_fs, X_cu, atol=atol, rtol=rtol)
        err_cu = float(torch.max(torch.abs(X_fs - X_cu)).item()) if X_fs.numel() else 0.0

    if ok_cu is None:
        status = "SKIP"
    elif ok_cu:
        status = "PASS"
    else:
        status = "FAIL"

    note_parts = []
    if cu_reason:
        note_parts.append(cu_reason)

    return {
        "fmt": fmt,
        "n_rows": int(shape[0]),
        "n_cols": int(shape[1]),
        "nnz": int(data.numel()),
        "rhs": int(n_rhs),
        "flagsparse_ms": fs_ms,
        "cusparse_ms": cu_ms,
        "fs_vs_cu": _fmt_speedup(cu_ms, fs_ms),
        "status": status,
        "err_cu": err_cu,
        "note": " | ".join(note_parts),
    }


def run_spsm_synthetic_all(n=512, n_rhs=32):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    total = 0
    failed = 0
    print("=" * 120)
    print("FLAGSPARSE SpSM (NON_TRANS/NON_TRANS, row-major) synthetic test")
    print("=" * 120)
    print(
        f"{'Fmt':>5} {'dtype':>9} {'index':>7} {'N':>6} {'RHS':>6} {'NNZ':>10} "
        f"{'FS(ms)':>10} {'CU(ms)':>10} {'FS/CU':>8} "
        f"{'Status':>8} {'Err(CU)':>12}"
    )
    print("-" * 120)
    for fmt in FORMATS:
        for value_dtype in VALUE_DTYPES:
            for index_dtype in INDEX_DTYPES:
                data, row, col, indptr, B, shape = _build_triangular_case(
                    n=n, n_rhs=n_rhs, value_dtype=value_dtype
                )
                atol, rtol = _tol(value_dtype)
                if fmt == "csr":
                    X_fs, fs_ms = fs.flagsparse_spsm_csr(
                        data=data,
                        indices=col.to(index_dtype),
                        indptr=indptr,
                        B=B,
                        shape=shape,
                        alpha=1.0,
                        lower=True,
                        unit_diagonal=False,
                        opA="NON_TRANS",
                        opB="NON_TRANS",
                        major="row",
                        return_time=True,
                    )
                else:
                    X_fs, fs_ms = fs.flagsparse_spsm_coo(
                        data=data,
                        row=row.to(index_dtype),
                        col=col.to(index_dtype),
                        B=B,
                        shape=shape,
                        alpha=1.0,
                        lower=True,
                        unit_diagonal=False,
                        opA="NON_TRANS",
                        opB="NON_TRANS",
                        major="row",
                        return_time=True,
                    )

                X_cu, cu_ms, cu_reason = _cupy_spsm_ref(
                    data, row, col, indptr, B, shape, fmt=fmt
                )
                ok_cu = None
                err_cu = None
                if X_cu is not None:
                    ok_cu = torch.allclose(X_fs, X_cu, atol=atol, rtol=rtol)
                    err_cu = float(torch.max(torch.abs(X_fs - X_cu)).item()) if X_fs.numel() else 0.0

                if ok_cu is None:
                    status = "SKIP"
                elif ok_cu:
                    status = "PASS"
                else:
                    status = "FAIL"
                total += 1
                if status != "PASS":
                    failed += 1

                print(
                    f"{fmt:>5} {_dtype_name(value_dtype):>9} {_dtype_name(index_dtype):>7} "
                    f"{shape[0]:>6} {B.shape[1]:>6} {int(data.numel()):>10} "
                    f"{_fmt_ms(fs_ms):>10} {_fmt_ms(cu_ms):>10} {_fmt_speedup(cu_ms, fs_ms):>8} "
                    f"{status:>8} {_fmt_err(err_cu):>12}"
                )
                if cu_reason is not None:
                    print(f"  NOTE: {cu_reason}")
    print("-" * 120)
    print(f"Total cases: {total}  Failed: {failed}")
    print("=" * 120)


def run_all_dtypes_spsm_csv(mtx_paths, csv_path, use_coo=False, n_rhs=32):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    rows_out = []
    fmt = "coo" if use_coo else "csr"

    print("=" * 132)
    print(f"FLAGSPARSE SpSM .mtx batch ({fmt.upper()}), NON_TRANS/NON_TRANS, row-major")
    print("=" * 132)
    print(
        f"{'Matrix':<28} {'dtype':>9} {'index':>7} {'N':>7} {'RHS':>6} {'NNZ':>10} "
        f"{'FS(ms)':>10} {'CU(ms)':>10} {'FS/CU':>8} "
        f"{'Status':>8} {'Err(CU)':>12}"
    )
    print("-" * 132)

    for value_dtype in CSV_VALUE_DTYPES:
        for index_dtype in CSV_INDEX_DTYPES:
            for path in mtx_paths:
                base = {
                    "matrix": os.path.basename(path),
                    "value_dtype": _dtype_name(value_dtype),
                    "index_dtype": _dtype_name(index_dtype),
                }
                try:
                    data, indices, indptr, shape = _load_mtx_to_csr_torch(
                        path, dtype=value_dtype, device=device
                    )
                    one = _run_one_spsm_case(
                        data,
                        indices,
                        indptr,
                        shape,
                        value_dtype,
                        index_dtype,
                        n_rhs,
                        fmt,
                    )
                    row = {
                        **base,
                        **one,
                    }
                    rows_out.append(row)
                    short = base["matrix"][:27] + ("…" if len(base["matrix"]) > 27 else "")
                    print(
                        f"{short:<28} {base['value_dtype']:>9} {base['index_dtype']:>7} "
                        f"{row['n_rows']:>7} {row['rhs']:>6} {row['nnz']:>10} "
                        f"{_fmt_ms(row['flagsparse_ms']):>10} {_fmt_ms(row['cusparse_ms']):>10} "
                        f"{row['fs_vs_cu']:>8} {row['status']:>8} "
                        f"{_fmt_err(row['err_cu']):>12}"
                    )
                    if row.get("note"):
                        print(f"  NOTE: {row['note']}")
                except Exception as exc:
                    row = {
                        **base,
                        "fmt": fmt,
                        "n_rows": "ERR",
                        "n_cols": "ERR",
                        "nnz": "ERR",
                        "rhs": int(n_rhs),
                        "flagsparse_ms": None,
                        "cusparse_ms": None,
                        "fs_vs_cu": "N/A",
                        "status": "ERROR",
                        "err_cu": None,
                        "note": str(exc),
                    }
                    rows_out.append(row)
                    short = base["matrix"][:27] + ("…" if len(base["matrix"]) > 27 else "")
                    print(
                        f"{short:<28} {base['value_dtype']:>9} {base['index_dtype']:>7} "
                        f"{'ERR':>7} {int(n_rhs):>6} {'ERR':>10} "
                        f"{_fmt_ms(None):>10} {_fmt_ms(None):>10} "
                        f"{'N/A':>8} {'ERROR':>8} "
                        f"{_fmt_err(None):>12}"
                    )
                    print(f"  ERROR: {exc}")

    print("-" * 132)
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "fmt",
        "n_rows",
        "n_cols",
        "nnz",
        "rhs",
        "flagsparse_ms",
        "cusparse_ms",
        "fs_vs_cu",
        "status",
        "err_cu",
        "note",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"Wrote {len(rows_out)} rows to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SpSM test: synthetic and optional .mtx CSV export (CSR/COO)."
    )
    parser.add_argument(
        "mtx",
        nargs="*",
        help=".mtx file path(s), or directory(ies) to glob for *.mtx",
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Run synthetic triangular tests"
    )
    parser.add_argument("--n", type=int, default=512, help="matrix size (synthetic)")
    parser.add_argument("--rhs", type=int, default=32, help="number of RHS columns")
    parser.add_argument(
        "--csv-csr",
        type=str,
        default=None,
        metavar="FILE",
        help="Run .mtx batch in CSR mode and export CSV",
    )
    parser.add_argument(
        "--csv-coo",
        type=str,
        default=None,
        metavar="FILE",
        help="Run .mtx batch in COO mode and export CSV",
    )
    args = parser.parse_args()

    if args.synthetic:
        run_spsm_synthetic_all(n=args.n, n_rhs=args.rhs)
        return

    paths = []
    for p in args.mtx:
        if os.path.isfile(p) and p.endswith(".mtx"):
            paths.append(p)
        elif os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.mtx"))))

    if args.csv_csr:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-csr")
            return
        run_all_dtypes_spsm_csv(paths, args.csv_csr, use_coo=False, n_rhs=args.rhs)
        return

    if args.csv_coo:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-coo")
            return
        run_all_dtypes_spsm_csv(paths, args.csv_coo, use_coo=True, n_rhs=args.rhs)
        return

    print("Use --synthetic, --csv-csr, or --csv-coo to run SpSM tests.")


if __name__ == "__main__":
    main()