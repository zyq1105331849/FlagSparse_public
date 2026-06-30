"""SpSM tests: synthetic triangular systems and optional .mtx batch CSV."""

import argparse
import csv
import glob
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from queue import Empty

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as fs
import flagsparse.sparse_operations.spsm as fs_spsm_impl

try:
    import cupy as cp
    import cupyx.cusparse as cpx_cusparse
    import cupyx.scipy.sparse as cpx_sparse
except Exception:
    cp = None
    cpx_cusparse = None
    cpx_sparse = None


FORMATS = ("csr", "coo")
VALUE_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
INDEX_DTYPES = [torch.int32]
CSV_VALUE_DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]
CSV_INDEX_DTYPES = [torch.int32]
WARMUP = 0
ITERS = 1
SPSM_CASE_TIMEOUT_SECONDS = 180
SPSM_OP_MODES = ["NON", "NON_TRANS"]


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _tol(dtype):
    if dtype in (torch.float32, torch.complex64):
        return 1e-4, 1e-3
    return 1e-12, 1e-10


def _reference_check_threshold(dtype):
    if dtype in (torch.float32, torch.complex64):
        return 1e-6
    return 1e-12


def _reference_max_relative_error(answer, result, dtype):
    if answer is None or result is None:
        return None
    if answer.numel() != result.numel():
        return float("inf")
    if answer.numel() == 0:
        return 0.0
    if dtype in (torch.complex64, torch.complex128):
        answer_cmp = torch.abs(answer)
        result_cmp = torch.abs(result)
        diff = torch.abs(answer_cmp - result_cmp)
    else:
        diff = torch.abs(answer - result)
        result_cmp = torch.abs(result)
    if not bool(torch.isfinite(diff).all().item()) or not bool(torch.isfinite(result_cmp).all().item()):
        return float("inf")
    max_error = torch.max(diff)
    max_result = torch.max(result_cmp)
    if float(max_result.item()) == 0.0:
        return 0.0 if float(max_error.item()) == 0.0 else float("inf")
    return float((max_error / max_result).item())


def _is_fatal_cuda_error(exc):
    msg = str(exc).lower()
    return (
        "illegal memory access" in msg
        or "device-side assert" in msg
        or "unspecified launch failure" in msg
    )


def _fmt_ms(v):
    return "N/A" if v is None else f"{v:.4f}"


def _fmt_ratio(v):
    return "N/A" if v is None else f"{v:.2f}"


def _fmt_err(v):
    return "N/A" if v is None else f"{v:.2e}"


def _safe_ratio(other_ms, triton_ms):
    if other_ms is None or triton_ms is None or triton_ms <= 0:
        return None
    return other_ms / triton_ms


def _spsm_benchmark_schedule(nnz, n_rhs, value_dtype, fmt="csr"):
    del nnz, n_rhs, value_dtype, fmt
    return int(WARMUP), int(ITERS)


def _allinone_filtered_avg_ms(times):
    if not times:
        return None
    times = [float(t) for t in times]
    if len(times) == 1:
        return times[0]
    ordered = sorted(times)
    n = len(ordered)
    if n % 2 == 0:
        median = (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    else:
        median = ordered[n // 2]
    lo = median * 0.9
    hi = median * 1.1
    kept = [t for t in ordered if lo <= t <= hi]
    return sum(kept) / len(kept) if kept else median


def _csv_export_row_spsm(row):
    return {
        "matrix": row.get("matrix"),
        "value_dtype": row.get("value_dtype"),
        "index_dtype": row.get("index_dtype"),
        "format": row.get("format"),
        "n_rows": row.get("n_rows"),
        "n_cols": row.get("n_cols"),
        "nnz": row.get("nnz"),
        "n_rhs": row.get("n_rhs"),
        "flagsparse_ms": row.get("flagsparse_ms"),
        "hipsparse_ms": row.get("hipsparse_ms"),
        "pytorch_ms": row.get("pytorch_ms"),
        "flagsparse_speedup_vs_hipsparse": row.get("flagsparse_speedup_vs_hipsparse"),
        "flagsparse_speedup_vs_pytorch": row.get("flagsparse_speedup_vs_pytorch"),
        "pt_status": row.get("pt_status"),
        "hs_status": row.get("hs_status"),
        "status": row.get("status"),
        "err_ref": row.get("err_ref"),
        "err_res": row.get("err_res"),
        "err_pt": row.get("err_pt"),
        "err_hs": row.get("err_hs"),
        "hipsparse_reason": row.get("hipsparse_reason"),
        "pytorch_reason": row.get("pytorch_reason"),
        "error": row.get("error"),
    }


def _empty_csv_case_row(base, fmt, n_rhs, status, error):
    return {
        **base,
        "format": fmt,
        "n_rows": "ERR",
        "n_cols": "ERR",
        "nnz": "ERR",
        "n_rhs": int(n_rhs),
        "flagsparse_ms": None,
        "hipsparse_ms": None,
        "pytorch_ms": None,
        "flagsparse_speedup_vs_hipsparse": None,
        "flagsparse_speedup_vs_pytorch": None,
        "pt_status": "N/A",
        "hs_status": "N/A",
        "status": status,
        "err_ref": None,
        "err_res": None,
        "err_pt": None,
        "err_hs": None,
        "hipsparse_reason": None,
        "pytorch_reason": None,
        "error": error,
    }


def _partial_csv_case_row(
    fmt,
    n_rows,
    n_cols,
    nnz,
    n_rhs,
    *,
    flagsparse_ms=None,
    hipsparse_ms=None,
    pytorch_ms=None,
    flagsparse_speedup_vs_hipsparse=None,
    flagsparse_speedup_vs_pytorch=None,
    pt_status="N/A",
    hs_status="N/A",
    status="PARTIAL",
    err_ref=None,
    err_res=None,
    err_pt=None,
    err_hs=None,
    hipsparse_reason=None,
    pytorch_reason=None,
    error=None,
):
    return {
        "format": fmt,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "nnz": int(nnz),
        "n_rhs": int(n_rhs),
        "flagsparse_ms": flagsparse_ms,
        "hipsparse_ms": hipsparse_ms,
        "pytorch_ms": pytorch_ms,
        "flagsparse_speedup_vs_hipsparse": flagsparse_speedup_vs_hipsparse,
        "flagsparse_speedup_vs_pytorch": flagsparse_speedup_vs_pytorch,
        "pt_status": pt_status,
        "hs_status": hs_status,
        "status": status,
        "err_ref": err_ref,
        "err_res": err_res,
        "err_pt": err_pt,
        "err_hs": err_hs,
        "hipsparse_reason": hipsparse_reason,
        "pytorch_reason": pytorch_reason,
        "error": error,
    }


def _short_matrix_name(matrix):
    return matrix[:27] + ("…" if len(matrix) > 27 else "")


def _print_csv_case_row(row):
    short = _short_matrix_name(row["matrix"])
    print(
        f"{short:<28} {row['value_dtype']:>9} {row['index_dtype']:>7} "
        f"{row['n_rows']:>7} {row['n_rhs']:>6} {row['nnz']:>10} "
        f"{_fmt_ms(row['flagsparse_ms']):>10} {_fmt_ms(row['hipsparse_ms']):>10} {_fmt_ms(row['pytorch_ms']):>10} "
        f"{_fmt_ratio(row['flagsparse_speedup_vs_hipsparse']):>10} {_fmt_ratio(row['flagsparse_speedup_vs_pytorch']):>10} "
        f"{row['status']:>10} {_fmt_err(row['err_ref']):>12} {_fmt_err(row['err_res']):>12} "
        f"{_fmt_err(row['err_pt']):>12} {_fmt_err(row['err_hs']):>12}"
    )


def _parse_csv_tokens(raw):
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _parse_ops_filter(raw):
    tokens = [tok.strip().upper() for tok in _parse_csv_tokens(raw)]
    if not tokens:
        return ["NON"]
    invalid = [tok for tok in tokens if tok not in SPSM_OP_MODES]
    if invalid:
        raise ValueError(
            f"unsupported spsm ops: {invalid}; current SpSM test only supports NON/NON_TRANS"
        )
    normalized = []
    for tok in tokens:
        normalized.append("NON" if tok == "NON_TRANS" else tok)
    return normalized


def _build_triangular_case(n=512, n_rhs=1024, value_dtype=torch.float32):
    device = torch.device("cuda")
    A = torch.tril(torch.randn((n, n), dtype=value_dtype, device=device) * 0.02)
    diag_base_dtype = torch.float32 if value_dtype == torch.complex64 else torch.float64
    diag = (torch.rand((n,), dtype=diag_base_dtype, device=device) + 2.0).to(value_dtype)
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


def _csr_to_coo(indices, indptr, n_rows):
    row = torch.repeat_interleave(
        torch.arange(n_rows, device=indptr.device, dtype=torch.int64),
        indptr[1:] - indptr[:-1],
    )
    return row, indices.to(torch.int64)


def _extract_effective_lower_csr(data, indices, indptr, shape):
    n_rows = int(shape[0])
    row, col = _csr_to_coo(indices, indptr, n_rows)
    tri_mask = col <= row
    data_tri = data[tri_mask]
    row_tri = row[tri_mask]
    col_tri = col[tri_mask]
    if row_tri.numel() == 0:
        empty_indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=indptr.device)
        empty_indices = torch.empty((0,), dtype=torch.int64, device=indices.device)
        empty_data = data.new_empty((0,))
        return empty_data, empty_indices, empty_indptr
    order = torch.argsort(row_tri * n_rows + col_tri)
    row_tri = row_tri[order]
    col_tri = col_tri[order]
    data_tri = data_tri[order]
    counts = torch.bincount(row_tri, minlength=n_rows)
    indptr_tri = torch.zeros(n_rows + 1, dtype=torch.int64, device=indptr.device)
    indptr_tri[1:] = torch.cumsum(counts, dim=0)
    return data_tri, col_tri.to(torch.int64), indptr_tri


def _stabilize_lower_triangular_csr(data, indices, indptr, shape):
    """Make Matrix Market lower-triangular extracts safe for NON_UNIT SpSM."""
    n_rows = int(shape[0])
    row, col = _csr_to_coo(indices, indptr, n_rows)
    diag_mask = row == col
    offdiag_mask = ~diag_mask

    real_dtype = (
        torch.float32
        if data.dtype in (torch.float32, torch.complex64)
        else torch.float64
    )
    offdiag_abs_sum = torch.zeros(
        n_rows, dtype=real_dtype, device=data.device
    )
    if bool(torch.any(offdiag_mask).item()):
        offdiag_abs_sum.index_add_(
            0,
            row[offdiag_mask],
            torch.abs(data[offdiag_mask]).to(real_dtype),
        )
    stable_diag = (offdiag_abs_sum + 1.0).to(data.dtype)

    data_stable = data.clone()
    diag_present = torch.zeros(n_rows, dtype=torch.bool, device=data.device)
    if bool(torch.any(diag_mask).item()):
        diag_rows = row[diag_mask]
        data_stable[diag_mask] = stable_diag[diag_rows]
        diag_present[diag_rows] = True

    missing_diag = torch.nonzero(
        ~diag_present, as_tuple=False
    ).reshape(-1).to(torch.int64)
    if missing_diag.numel() > 0:
        row = torch.cat((row, missing_diag))
        col = torch.cat((col, missing_diag))
        data_stable = torch.cat((data_stable, stable_diag[missing_diag]))

    order = torch.argsort(row * max(1, n_rows) + col)
    row = row[order]
    col = col[order]
    data_stable = data_stable[order]
    counts = torch.bincount(row, minlength=n_rows)
    indptr_stable = torch.zeros(
        n_rows + 1, dtype=torch.int64, device=data.device
    )
    indptr_stable[1:] = torch.cumsum(counts, dim=0)
    return data_stable, col.to(torch.int64), indptr_stable


def _benchmark_pytorch_reference(data, indices, indptr, shape, B):
    try:
        sparse_spsolve = getattr(torch.sparse, "spsolve", None)
        if sparse_spsolve is None:
            raise NotImplementedError("torch.sparse.spsolve is unavailable")
        data_eff, indices_eff, indptr_eff = _extract_effective_lower_csr(
            data, indices, indptr, shape
        )
        A_csr = torch.sparse_csr_tensor(
            indptr_eff,
            indices_eff,
            data_eff,
            size=shape,
            device=data.device,
        )
        if not A_csr.is_cuda:
            raise RuntimeError("torch.sparse.spsolve CUDA path is unavailable")
        cols = []
        for bj in torch.unbind(B, dim=1):
            cols.append(sparse_spsolve(A_csr, bj))
        X_ref = torch.stack(cols, dim=1) if cols else B.new_empty(B.shape)
        torch.cuda.synchronize()
        return X_ref.to(B.dtype), None, "gpu_sparse", None
    except Exception as exc:
        if "out of memory" in str(exc).lower() and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, None, "unavailable", f"PyTorch sparse solve unavailable ({exc})"


def _benchmark_sparse_reference(data, row, col, indptr, B, shape, fmt, warmup, iters):
    if fs_spsm_impl._is_rocm_runtime():
        result = fs_spsm_impl._benchmark_spsm_csr_sparse_ref(
            data,
            col,
            indptr,
            B,
            shape,
            lower=True,
            unit_diagonal=False,
            warmup=warmup,
            iters=iters,
        )
        return (
            result["values"],
            result["ms"],
            result["reason"],
        )
    if cp is None or cpx_sparse is None or cpx_cusparse is None:
        return None, None, "cusparse unavailable"
    try:
        data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data.contiguous()))
        B_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(B.contiguous()))
        if fmt == "coo":
            row_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(row.contiguous()))
            col_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(col.contiguous()))
            A_cp = cpx_sparse.coo_matrix((data_cp, (row_cp, col_cp)), shape=shape)
        else:
            idx_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(col.contiguous()))
            ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr.contiguous()))
            A_cp = cpx_sparse.csr_matrix((data_cp, idx_cp, ptr_cp), shape=shape)
        A_cp.sum_duplicates()
        for _ in range(warmup):
            _ = cpx_cusparse.spsm(A_cp, B_cp, lower=True, unit_diag=False, transa=False)
        cp.cuda.runtime.deviceSynchronize()
        times = []
        for _ in range(iters):
            c0 = cp.cuda.Event()
            c1 = cp.cuda.Event()
            c0.record()
            X_cp = cpx_cusparse.spsm(A_cp, B_cp, lower=True, unit_diag=False, transa=False)
            c1.record()
            c1.synchronize()
            times.append(cp.cuda.get_elapsed_time(c0, c1))
        ms = _allinone_filtered_avg_ms(times)
        X_t = torch.utils.dlpack.from_dlpack(X_cp.toDlpack()).to(B.dtype)
        return X_t, ms, None
    except Exception as exc:
        return None, None, str(exc)


def _apply_csr_to_dense_rhs(data, indices, indptr, X, shape):
    n_rows = int(shape[0])
    row, col = _csr_to_coo(indices, indptr, n_rows)
    out = torch.zeros((n_rows, X.shape[1]), dtype=X.dtype, device=X.device)
    out.index_add_(0, row, data[:, None] * X[col])
    return out


def _solution_residual_metrics(data, indices, indptr, shape, X, B, value_dtype):
    atol, rtol = _tol(value_dtype)
    data_eff, indices_eff, indptr_eff = _extract_effective_lower_csr(
        data, indices, indptr, shape
    )
    B_recon = _apply_csr_to_dense_rhs(data_eff, indices_eff, indptr_eff, X, shape)
    err = float(torch.max(torch.abs(B_recon - B)).item()) if B.numel() else 0.0
    ok = torch.allclose(B_recon, B, atol=atol, rtol=rtol)
    return err, ok


def _benchmark_flagsparse_full_round(
    reset_call,
    analyze_call,
    solve_call,
    warmup,
    iters,
    *,
    progress=None,
    max_total_seconds=None,
    stage_prefix="flagsparse",
):
    def _progress(stage):
        if progress is not None:
            progress(stage)

    total_rounds = max(1, int(warmup) + int(iters))

    def _check_estimate(round_ms, source):
        if max_total_seconds is None:
            return
        estimated_seconds = (float(round_ms) * total_rounds) / 1000.0
        if estimated_seconds > float(max_total_seconds):
            raise TimeoutError(
                f"{source} took {round_ms:.4f} ms; estimated "
                f"{estimated_seconds:.1f}s for warmup={warmup}, timed_iters={iters}, "
                f"exceeds per-matrix timeout {float(max_total_seconds):.1f}s"
            )

    X = None
    for warmup_idx in range(warmup):
        reset_call()
        torch.cuda.synchronize()
        start = time.perf_counter()
        _progress(f"{stage_prefix}_warmup_{warmup_idx + 1}_analysis")
        analyze_call()
        _progress(f"{stage_prefix}_warmup_{warmup_idx + 1}_solve")
        X = solve_call()
        torch.cuda.synchronize()
        round_ms = (time.perf_counter() - start) * 1000.0
        if warmup_idx > 0 or warmup <= 1:
            _check_estimate(
                round_ms,
                f"{stage_prefix} warmup round {warmup_idx + 1}",
            )
    times = []
    for iter_idx in range(iters):
        reset_call()
        torch.cuda.synchronize()
        start = time.perf_counter()
        _progress(f"{stage_prefix}_timed_{iter_idx + 1}_analysis")
        analyze_call()
        _progress(f"{stage_prefix}_timed_{iter_idx + 1}_solve")
        X = solve_call()
        torch.cuda.synchronize()
        round_ms = (time.perf_counter() - start) * 1000.0
        if iter_idx == 0 and warmup <= 0:
            _check_estimate(round_ms, f"{stage_prefix} timed round 1")
        times.append(round_ms)
    return X, _allinone_filtered_avg_ms(times)


def _benchmark_flagsparse_spsm_csr_total(
    data,
    indices,
    indptr,
    B,
    shape,
    *,
    progress=None,
    max_total_seconds=None,
):
    warmup, iters = _spsm_benchmark_schedule(
        data.numel(), B.shape[1], data.dtype, fmt="csr"
    )
    analyze_call = lambda: fs_spsm_impl._analyze_spsm_csr(
        data, indices, indptr, B, shape,
        lower=True, unit_diagonal=False, clear_cache=False, return_time=False,
    )
    solve_call = lambda: fs.flagsparse_spsm_csr(
            data,
            indices,
            indptr,
            B,
            shape,
            lower=True,
            unit_diagonal=False,
            opA="NON_TRANS",
            opB="NON_TRANS",
            major="row",
        )
    return _benchmark_flagsparse_full_round(
        fs_spsm_impl._clear_spsm_preprocess_cache,
        analyze_call,
        solve_call,
        warmup,
        iters,
        progress=progress,
        max_total_seconds=max_total_seconds,
        stage_prefix="flagsparse_csr",
    )


def _benchmark_flagsparse_spsm_coo_total(
    data,
    row,
    col,
    B,
    shape,
    *,
    progress=None,
    max_total_seconds=None,
):
    warmup, iters = _spsm_benchmark_schedule(
        data.numel(), B.shape[1], data.dtype, fmt="coo"
    )
    analyze_call = lambda: fs_spsm_impl._analyze_spsm_coo(
        data, row, col, B, shape,
        lower=True, unit_diagonal=False, clear_cache=False, return_time=False,
    )
    solve_call = lambda: fs.flagsparse_spsm_coo(
            data,
            row,
            col,
            B,
            shape,
            lower=True,
            unit_diagonal=False,
            opA="NON_TRANS",
            opB="NON_TRANS",
            major="row",
        )
    return _benchmark_flagsparse_full_round(
        fs_spsm_impl._clear_spsm_preprocess_cache,
        analyze_call,
        solve_call,
        warmup,
        iters,
        progress=progress,
        max_total_seconds=max_total_seconds,
        stage_prefix="flagsparse_coo",
    )


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
        if c in row:
            row[c] += v
        else:
            row[c] = v

    for line in data_lines[:nnz]:
        parts = line.split()
        if len(parts) < 2:
            continue
        r = int(parts[0]) - 1
        c = int(parts[1]) - 1
        if mm_field == "complex":
            if len(parts) < 4:
                raise ValueError("MatrixMarket complex entry requires real and imag parts")
            v = complex(float(parts[2]), float(parts[3]))
        elif len(parts) >= 3:
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


def _run_one_spsm_case(
    data,
    indices,
    indptr,
    shape,
    value_dtype,
    index_dtype,
    n_rhs,
    fmt,
    progress=None,
    partial=None,
    max_case_seconds=None,
):
    def _progress(stage):
        if progress is not None:
            progress(stage)

    def _partial(row):
        if partial is not None:
            partial(row)

    _progress("prepare_rhs_and_triangular_csr")
    n_rows = int(shape[0])
    B = torch.randn((n_rows, n_rhs), dtype=value_dtype, device=data.device).contiguous()
    data_eff, indices_eff, indptr_eff = _extract_effective_lower_csr(
        data, indices, indptr, shape
    )
    data_eff, indices_eff, indptr_eff = _stabilize_lower_triangular_csr(
        data_eff, indices_eff, indptr_eff, shape
    )
    row, col = _csr_to_coo(indices_eff, indptr_eff, n_rows)
    warmup, iters = _spsm_benchmark_schedule(
        data_eff.numel(), n_rhs, value_dtype, fmt=fmt
    )

    if fmt == "csr":
        _progress("flagsparse_csr_analysis_plus_solve")
        X_fs, flagsparse_ms = _benchmark_flagsparse_spsm_csr_total(
            data_eff,
            indices_eff.to(index_dtype),
            indptr_eff.to(index_dtype),
            B,
            shape,
            progress=_progress,
            max_total_seconds=max_case_seconds,
        )
    else:
        _progress("flagsparse_coo_analysis_plus_solve")
        X_fs, flagsparse_ms = _benchmark_flagsparse_spsm_coo_total(
            data_eff,
            row.to(index_dtype),
            col.to(index_dtype),
            B,
            shape,
            progress=_progress,
            max_total_seconds=max_case_seconds,
        )
    _partial(
        _partial_csv_case_row(
            fmt,
            n_rows,
            int(shape[1]),
            int(data_eff.numel()),
            n_rhs,
            flagsparse_ms=flagsparse_ms,
            status="FS_DONE",
        )
    )
    _progress("hipsparse_csrsm2_analysis_plus_solve")
    (
        X_hs,
        hipsparse_ms,
        hipsparse_reason,
    ) = _benchmark_sparse_reference(
        data_eff,
        row.to(index_dtype),
        col.to(index_dtype),
        indptr_eff.to(index_dtype),
        B,
        shape,
        fmt,
        warmup,
        iters,
    )
    _partial(
        _partial_csv_case_row(
            fmt,
            n_rows,
            int(shape[1]),
            int(data_eff.numel()),
            n_rhs,
            flagsparse_ms=flagsparse_ms,
            hipsparse_ms=hipsparse_ms,
            flagsparse_speedup_vs_hipsparse=_safe_ratio(
                hipsparse_ms, flagsparse_ms
            ),
            hs_status="DONE" if X_hs is not None else "N/A",
            status="HS_DONE",
            hipsparse_reason=hipsparse_reason,
        )
    )
    _progress("pytorch_reference_single_spsolve")
    X_pt, pytorch_ms, _pt_backend, pytorch_reason = _benchmark_pytorch_reference(
        data_eff, indices_eff, indptr_eff, shape, B
    )
    _progress("validate_results")

    err_hs = None
    ok_hs = None
    rel_hs = None
    if X_hs is not None:
        err_hs = float(torch.max(torch.abs(X_fs - X_hs)).item()) if X_fs.numel() else 0.0
        rel_hs = _reference_max_relative_error(X_hs, X_fs, value_dtype)
        ok_hs = rel_hs <= _reference_check_threshold(value_dtype)

    err_pt = None
    ok_pt = None
    rel_pt = None
    if X_pt is not None:
        err_pt = float(torch.max(torch.abs(X_fs - X_pt)).item()) if X_fs.numel() else 0.0
        rel_pt = _reference_max_relative_error(X_pt, X_fs, value_dtype)
        ok_pt = rel_pt <= _reference_check_threshold(value_dtype)

    err_res, _ = _solution_residual_metrics(
        data_eff, indices_eff, indptr_eff, shape, X_fs, B, value_dtype
    )
    ref_errors = [v for v in (err_pt, err_hs) if v is not None]
    err_ref = min(ref_errors) if ref_errors else None

    if ok_hs is not None:
        status = "PASS" if ok_hs else "FAIL"
    elif ok_pt is not None:
        status = "PASS" if ok_pt else "FAIL"
    elif X_pt is None and X_hs is None:
        status = "REF_FAIL"
    else:
        status = "FAIL"

    return {
        "format": fmt,
        "n_rows": n_rows,
        "n_cols": int(shape[1]),
        "nnz": int(data_eff.numel()),
        "n_rhs": int(n_rhs),
        "flagsparse_ms": flagsparse_ms,
        "hipsparse_ms": hipsparse_ms,
        "pytorch_ms": pytorch_ms,
        "flagsparse_speedup_vs_hipsparse": _safe_ratio(hipsparse_ms, flagsparse_ms),
        "flagsparse_speedup_vs_pytorch": _safe_ratio(pytorch_ms, flagsparse_ms),
        "pt_status": "PASS" if ok_pt else ("FAIL" if X_pt is not None else "N/A"),
        "hs_status": "PASS" if ok_hs else ("FAIL" if X_hs is not None else "N/A"),
        "status": status,
        "err_ref": err_ref,
        "err_res": err_res,
        "err_pt": err_pt,
        "err_hs": err_hs,
        "hipsparse_reason": hipsparse_reason,
        "pytorch_reason": pytorch_reason,
        "error": None,
    }


def _run_spsm_csv_case_worker(
    result_queue,
    path,
    base,
    value_dtype,
    index_dtype,
    n_rhs,
    fmt,
    warmup,
    iters,
    timeout_seconds,
):
    global WARMUP, ITERS
    WARMUP = max(0, int(warmup))
    ITERS = max(1, int(iters))
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm device is not available.")
        device = torch.device("cuda")
        result_queue.put(("progress", "load_mtx"))
        data, indices, indptr, shape = _load_mtx_to_csr_torch(
            path,
            dtype=value_dtype,
            device=device,
        )
        result_queue.put(("progress", "run_case"))
        row = _run_one_spsm_case(
            data,
            indices,
            indptr,
            shape,
            value_dtype,
            index_dtype,
            n_rhs,
            fmt,
            progress=lambda stage: result_queue.put(("progress", stage)),
            partial=lambda row: result_queue.put(("partial", {**base, **row})),
            max_case_seconds=timeout_seconds,
        )
        result_queue.put(("ok", {**base, **row}))
    except BaseException as exc:
        result_queue.put(("error", f"{exc.__class__.__name__}: {exc}"))


def _run_spsm_csv_case_with_timeout(
    path,
    base,
    value_dtype,
    index_dtype,
    n_rhs,
    fmt,
    timeout_seconds,
):
    timeout_seconds = max(1, int(timeout_seconds))
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_run_spsm_csv_case_worker,
        args=(
            result_queue,
            path,
            base,
            value_dtype,
            index_dtype,
            n_rhs,
            fmt,
            WARMUP,
            ITERS,
            timeout_seconds,
        ),
    )
    proc.start()
    deadline = time.monotonic() + timeout_seconds
    last_phase = "startup"
    latest_partial = None
    while proc.is_alive() and time.monotonic() < deadline:
        proc.join(0.2)
        while True:
            try:
                status, payload = result_queue.get_nowait()
            except Empty:
                break
            if status == "progress":
                last_phase = str(payload)
                continue
            if status == "partial":
                latest_partial = dict(payload)
                continue
            if status == "ok":
                proc.join()
                return payload
            err_msg = str(payload)
            status_out = (
                "SKIP"
                if (
                    "SpSM requires square matrices" in err_msg
                    or "exceeds per-matrix timeout" in err_msg
                    or "TimeoutError" in err_msg
                )
                else "ERROR"
            )
            proc.join()
            return _empty_csv_case_row(base, fmt, n_rhs, status_out, err_msg)

    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join()
        timeout_error = f"timed out after {timeout_seconds} seconds during {last_phase}"
        if latest_partial is not None:
            row = dict(latest_partial)
            row["status"] = "SKIP"
            row["error"] = timeout_error
            return row
        return _empty_csv_case_row(
            base,
            fmt,
            n_rhs,
            "SKIP",
            timeout_error,
        )

    while True:
        try:
            status, payload = result_queue.get_nowait()
        except Empty:
            break
        if status == "progress":
            last_phase = str(payload)
            continue
        if status == "partial":
            latest_partial = dict(payload)
            continue
        if status == "ok":
            return payload
        err_msg = str(payload)
        status_out = (
            "SKIP"
            if (
                "SpSM requires square matrices" in err_msg
                or "exceeds per-matrix timeout" in err_msg
                or "TimeoutError" in err_msg
            )
            else "ERROR"
        )
        return _empty_csv_case_row(base, fmt, n_rhs, status_out, err_msg)

    if latest_partial is not None:
        row = dict(latest_partial)
        row["status"] = "ERROR"
        row["error"] = (
            f"worker exited with code {proc.exitcode} after partial result; "
            f"last phase: {last_phase}"
        )
        return row

    return _empty_csv_case_row(
        base,
        fmt,
        n_rhs,
        "ERROR",
        f"worker exited with code {proc.exitcode} without returning a result; last phase: {last_phase}",
    )


def run_spsm_synthetic_all(n=512, n_rhs=1024):
    if not torch.cuda.is_available():
        print("CUDA/ROCm device is not available.")
        return
    total = 0
    failed = 0
    print("=" * 160)
    print("FLAGSPARSE SpSM synthetic test")
    print("=" * 160)
    if fs_spsm_impl._is_rocm_runtime():
        print(
            "Baselines: hipSPARSE csrsm2 matrix solve + PyTorch official sparse solve "
            "(each timed hipSPARSE round performs fresh analysis/preparation + solve)."
        )
    else:
        print(
            "Baselines: cuSPARSE sparse triangular solve + PyTorch official sparse solve "
            "(PyTorch aggregates one torch.sparse.spsolve call per RHS column; "
            "cuSPARSE solves the full matrix RHS in one interface call)."
        )
    print(
        f"{'Fmt':>5} {'dtype':>9} {'index':>7} {'N':>6} {'RHS':>6} {'NNZ':>10} "
        f"{'FS(ms)':>10} {'HS(ms)':>10} {'PT(ms)':>10} "
        f"{'FS/HS':>10} {'FS/PT':>10} "
        f"{'Status':>10} {'Err(Ref)':>12} {'Err(Res)':>12} {'Err(PT)':>12} {'Err(HS)':>12}"
    )
    print("-" * 160)

    for fmt in FORMATS:
        for value_dtype in VALUE_DTYPES:
            for index_dtype in INDEX_DTYPES:
                data, row, col, indptr, _B, shape = _build_triangular_case(
                    n=n,
                    n_rhs=n_rhs,
                    value_dtype=value_dtype,
                )
                one = _run_one_spsm_case(
                    data,
                    col,
                    indptr,
                    shape,
                    value_dtype,
                    index_dtype,
                    n_rhs,
                    fmt,
                )
                total += 1
                if one["status"] != "PASS":
                    failed += 1
                print(
                    f"{fmt:>5} {_dtype_name(value_dtype):>9} {_dtype_name(index_dtype):>7} "
                    f"{shape[0]:>6} {n_rhs:>6} {one['nnz']:>10} "
                    f"{_fmt_ms(one['flagsparse_ms']):>10} {_fmt_ms(one['hipsparse_ms']):>10} {_fmt_ms(one['pytorch_ms']):>10} "
                    f"{_fmt_ratio(one['flagsparse_speedup_vs_hipsparse']):>10} {_fmt_ratio(one['flagsparse_speedup_vs_pytorch']):>10} "
                    f"{one['status']:>10} {_fmt_err(one['err_ref']):>12} {_fmt_err(one['err_res']):>12} "
                    f"{_fmt_err(one['err_pt']):>12} {_fmt_err(one['err_hs']):>12}"
                )
                if one["status"] in ("FAIL", "REF_FAIL"):
                    if one["hipsparse_reason"]:
                        print(f"  NOTE: sparse library baseline unavailable: {one['hipsparse_reason']}")
                    if one["pytorch_reason"]:
                        print(f"  NOTE: {one['pytorch_reason']}")
    print("-" * 160)
    print(f"Total cases: {total}  Failed: {failed}")
    print("=" * 160)


def run_all_dtypes_spsm_csv(
    mtx_paths,
    csv_path,
    use_coo=False,
    n_rhs=1024,
    case_timeout_seconds=SPSM_CASE_TIMEOUT_SECONDS,
    warmup=WARMUP,
    iters=ITERS,
):
    if not torch.cuda.is_available():
        print("CUDA/ROCm device is not available.")
        return
    global WARMUP, ITERS
    WARMUP = max(0, int(warmup))
    ITERS = max(1, int(iters))
    rows_out = []
    fmt = "coo" if use_coo else "csr"

    print("=" * 176)
    if fs_spsm_impl._is_rocm_runtime():
        baseline_text = (
            "hipSPARSE csrsm2 matrix solve + PyTorch official sparse solve "
            "(each timed hipSPARSE round performs fresh analysis/preparation + solve)"
        )
    else:
        baseline_text = (
            "cuSPARSE sparse triangular solve + PyTorch official sparse solve "
            "(PyTorch calls spsolve per RHS; cuSPARSE solves the full dense RHS)"
        )
    print(f"FLAGSPARSE SpSM .mtx batch ({fmt.upper()}) | baselines: {baseline_text}")
    print("=" * 176)
    print(
        f"Benchmark schedule: warmup={WARMUP}, timed_iters={ITERS} "
        "(each timed round performs analysis/preparation + solve; override with --warmup/--iters)"
    )
    print(f"Per-matrix timeout: {int(case_timeout_seconds)} seconds")
    print(
        "FS(ms) and HS(ms) each include one fresh analysis/preparation plus one solve. "
        "FS/HS = HS(ms) / FS(ms); FS/PT = PT(ms) / FS(ms)."
    )
    print(
        f"{'Matrix':<28} {'dtype':>9} {'index':>7} {'N':>7} {'RHS':>6} {'NNZ':>10} "
        f"{'FS(ms)':>10} {'HS(ms)':>10} {'PT(ms)':>10} "
        f"{'FS/HS':>10} {'FS/PT':>10} "
        f"{'Status':>10} {'Err(Ref)':>12} {'Err(Res)':>12} {'Err(PT)':>12} {'Err(HS)':>12}"
    )
    print("-" * 176)

    for value_dtype in CSV_VALUE_DTYPES:
        for index_dtype in CSV_INDEX_DTYPES:
            for path in mtx_paths:
                base = {
                    "matrix": os.path.basename(path),
                    "value_dtype": _dtype_name(value_dtype),
                    "index_dtype": _dtype_name(index_dtype),
                }
                try:
                    print(
                        f"RUNNING: {base['matrix']} | dtype={base['value_dtype']} | "
                        f"index={base['index_dtype']} | fmt={fmt}",
                        flush=True,
                    )
                    row = _run_spsm_csv_case_with_timeout(
                        path,
                        base,
                        value_dtype,
                        index_dtype,
                        n_rhs,
                        fmt,
                        case_timeout_seconds,
                    )
                    rows_out.append(row)
                    _print_csv_case_row(row)
                    if row["status"] in ("FAIL", "REF_FAIL"):
                        if row["hipsparse_reason"]:
                            print(
                                "  NOTE: sparse library baseline unavailable: "
                                f"{row['hipsparse_reason']}"
                            )
                        if row["pytorch_reason"]:
                            print(f"  NOTE: {row['pytorch_reason']}")
                    if row["status"] == "SKIP" and row.get("error"):
                        print(f"  SKIP: {row['error']}")
                    elif row["status"] == "ERROR" and row.get("error"):
                        print(f"  ERROR: {row['error']}")
                except Exception as exc:
                    err_msg = str(exc)
                    if _is_fatal_cuda_error(exc):
                        print(
                            f"  FATAL CUDA ERROR: {exc}\n"
                            "  CUDA context is no longer reliable. Restart the Python/Singularity session "
                            "and rerun the single failing matrix with CUDA_LAUNCH_BLOCKING=1."
                        )
                        raise
                    status = "SKIP" if "SpSM requires square matrices" in err_msg else "ERROR"
                    row = _empty_csv_case_row(base, fmt, n_rhs, status, err_msg)
                    rows_out.append(row)
                    _print_csv_case_row(row)
                    print(f"  {status}: {exc}")

    print("-" * 176)
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "format",
        "n_rows",
        "n_cols",
        "nnz",
        "n_rhs",
        "flagsparse_ms",
        "hipsparse_ms",
        "pytorch_ms",
        "flagsparse_speedup_vs_hipsparse",
        "flagsparse_speedup_vs_pytorch",
        "pt_status",
        "hs_status",
        "status",
        "err_ref",
        "err_res",
        "err_pt",
        "err_hs",
        "hipsparse_reason",
        "pytorch_reason",
        "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_out:
            w.writerow({k: ("" if v is None else v) for k, v in _csv_export_row_spsm(row).items()})
    print(f"Wrote {len(rows_out)} rows to {csv_path}")


def main():
    global WARMUP, ITERS
    parser = argparse.ArgumentParser(
        description="SpSM test: synthetic triangular systems and optional .mtx batch CSV."
    )
    parser.add_argument(
        "mtx",
        nargs="*",
        help=".mtx file path(s), or directory(ies) to glob for *.mtx",
    )
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic triangular tests")
    parser.add_argument("--n", type=int, default=512, help="matrix size (synthetic)")
    parser.add_argument(
        "--rhs",
        type=int,
        default=1024,
        help="number of RHS columns (default: 1024, matching all-in-one SpSM)",
    )
    parser.add_argument("--csv-csr", type=str, default=None, metavar="FILE")
    parser.add_argument("--csv-coo", type=str, default=None, metavar="FILE")
    parser.add_argument(
        "--ops",
        type=str,
        default="NON",
        help="comma-separated op(A) modes; currently only NON/NON_TRANS is supported",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP,
        help="Benchmark warmup full rounds; each round performs analysis/preparation + solve",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=ITERS,
        help="Benchmark timed full rounds; each round performs analysis/preparation + solve",
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=int,
        default=SPSM_CASE_TIMEOUT_SECONDS,
        help="Skip one .mtx case if it does not finish within this many seconds (default: 180)",
    )
    args = parser.parse_args()
    WARMUP = max(0, int(args.warmup))
    ITERS = max(1, int(args.iters))

    ops = _parse_ops_filter(args.ops)
    if any(op != "NON" for op in ops):
        raise ValueError("SpSM test currently supports only --ops NON/NON_TRANS")

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
        run_all_dtypes_spsm_csv(
            paths,
            args.csv_csr,
            use_coo=False,
            n_rhs=args.rhs,
            case_timeout_seconds=args.case_timeout_seconds,
            warmup=WARMUP,
            iters=ITERS,
        )
        return

    if args.csv_coo:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-coo")
            return
        run_all_dtypes_spsm_csv(
            paths,
            args.csv_coo,
            use_coo=True,
            n_rhs=args.rhs,
            case_timeout_seconds=args.case_timeout_seconds,
            warmup=WARMUP,
            iters=ITERS,
        )
        return

    print("Use --synthetic, --csv-csr, or --csv-coo to run SpSM tests.")


if __name__ == "__main__":
    main()
