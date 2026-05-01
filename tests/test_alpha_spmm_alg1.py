"""
Experimental AlphaSparse ALG1 port benchmark: base vs alpha_alg1 vs opt_alg2 vs references.
"""

import argparse
import csv
import glob
import math
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as fs

from test_spmm_opt import _seeded_dense_matrix, load_mtx_to_csr_torch


VALUE_DTYPES = [torch.float32, torch.float64]
INDEX_DTYPES = [torch.int32]
WARMUP = 10
ITERS = 50
DEFAULT_DENSE_COLS = 32
DEFAULT_SEED = 0
SUMMARY_FIELDS = [
    "matrix",
    "value_dtype",
    "dense_cols",
    "seed",
    "n_rows",
    "n_cols",
    "nnz",
    "avg_nnz_per_row",
    "max_row_nnz",
    "base_ms",
    "alpha_alg1_ms",
    "opt_alg2_ms",
    "torch_ms",
    "cusparse_ms",
    "base_vs_torch_err",
    "alpha_alg1_vs_torch_err",
    "opt_alg2_vs_torch_err",
    "base_vs_cusparse_err",
    "alpha_alg1_vs_cusparse_err",
    "opt_alg2_vs_cusparse_err",
    "base_status_vs_torch",
    "alpha_alg1_status_vs_torch",
    "opt_alg2_status_vs_torch",
    "base_status_vs_cusparse",
    "alpha_alg1_status_vs_cusparse",
    "opt_alg2_status_vs_cusparse",
    "matrix_status",
]
LAUNCH_FIELDS = [
    "matrix",
    "device_name",
    "sm_count",
    "dtype",
    "dense_cols",
    "warp_size",
    "factor",
    "block_size",
    "block_rows",
    "block_cols",
    "num_warps",
    "num_stages",
    "grid_m",
    "grid_n",
]


def _resolve_input_paths(input_paths):
    paths = []
    for path in input_paths:
        if os.path.isfile(path) and path.lower().endswith(".mtx"):
            paths.append(os.path.abspath(path))
        elif os.path.isdir(path):
            paths.extend(sorted(glob.glob(os.path.join(path, "*.mtx"))))
    return paths


def _reference_tolerance(dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-2
    return 1e-12, 1e-10


def _status_from_error(error_value):
    if error_value is None:
        return "SKIP"
    return "PASS" if float(error_value) <= 1.0 else "FAIL"


def _error_profile(candidate, reference, dtype):
    if reference is None:
        return {"global_err": None, "status": "SKIP"}
    atol, rtol = _reference_tolerance(dtype)
    if candidate.numel() == 0:
        return {"global_err": 0.0, "status": "PASS"}
    diff = torch.abs(candidate - reference).to(torch.float64)
    denom = (atol + rtol * torch.abs(reference)).to(torch.float64)
    ratio = diff / denom
    global_err = float(torch.max(ratio).item()) if ratio.numel() > 0 else 0.0
    return {"global_err": global_err, "status": _status_from_error(global_err)}


def _benchmark(op, warmup, iters):
    out = op()
    torch.cuda.synchronize()
    for _ in range(max(0, int(warmup))):
        out = op()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, int(iters))):
        out = op()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end) / max(1, int(iters))


def _timed_spmm_base(data, indices, indptr, B, shape, warmup, iters):
    out = fs.flagsparse_spmm_csr(data, indices, indptr, B, shape)
    _, elapsed = _benchmark(
        lambda: fs.flagsparse_spmm_csr(data, indices, indptr, B, shape),
        warmup,
        iters,
    )
    return out, elapsed


def _timed_alpha_spmm_alg1(data, indices, indptr, B, shape, warmup, iters):
    prepared = fs.prepare_alpha_spmm_alg1(data, indices, indptr, shape)
    out = fs.flagsparse_alpha_spmm_alg1(B=B, prepared=prepared)
    _, elapsed = _benchmark(
        lambda: fs.flagsparse_alpha_spmm_alg1(B=B, prepared=prepared),
        warmup,
        iters,
    )
    return out, elapsed, prepared


def _timed_spmm_opt_alg2(data, indices, indptr, B, shape, warmup, iters):
    prepared = fs.prepare_spmm_csr_opt_alg2(data, indices, indptr, shape)
    out = fs.flagsparse_spmm_csr_opt_alg2(B=B, prepared=prepared)
    _, elapsed = _benchmark(
        lambda: fs.flagsparse_spmm_csr_opt_alg2(B=B, prepared=prepared),
        warmup,
        iters,
    )
    return out, elapsed, prepared


def _timed_torch_reference(data, indices, indptr, B, shape, dtype, warmup, iters):
    device = data.device
    ref_dtype = torch.float64 if dtype == torch.float32 else dtype
    sparse = torch.sparse_csr_tensor(
        indptr.to(torch.int64),
        indices.to(torch.int64),
        data.to(ref_dtype),
        size=shape,
        device=device,
    )
    ref = torch.sparse.mm(sparse, B.to(ref_dtype)).to(dtype)
    _, elapsed = _benchmark(
        lambda: torch.sparse.mm(
            torch.sparse_csr_tensor(
                indptr.to(torch.int64),
                indices.to(torch.int64),
                data,
                size=shape,
                device=device,
            ),
            B,
        ),
        warmup,
        iters,
    )
    return ref, elapsed


def _timed_sparse_backend(data, indices, indptr, B, shape, warmup, iters, enabled):
    backend_name = "hipsparse_ref" if getattr(torch.version, "hip", None) else "cusparse_ref"
    if not enabled:
        return None, None, backend_name, "disabled"
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpx
    except Exception as exc:
        return None, None, backend_name, str(exc)

    try:
        data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
        ind_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices.to(torch.int64)))
        ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr))
        B_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(B))
        sparse = cpx.csr_matrix((data_cp, ind_cp, ptr_cp), shape=shape)
        out_cp, elapsed = _benchmark(lambda: sparse @ B_cp, warmup, iters)
        out = torch.utils.dlpack.from_dlpack(out_cp.toDlpack())
        return out, elapsed, backend_name, None
    except Exception as exc:
        return None, None, backend_name, str(exc)


def _build_csr_from_row_lengths(row_lengths, n_cols, dtype, index_dtype, device, pattern="random"):
    data_parts = []
    col_parts = []
    indptr = [0]
    for row, row_nnz in enumerate(row_lengths):
        row_nnz = int(max(0, min(int(row_nnz), n_cols)))
        if row_nnz == 0:
            indptr.append(indptr[-1])
            continue
        if pattern == "banded":
            center = row % max(n_cols, 1)
            offsets = torch.arange(row_nnz, device=device, dtype=torch.int64)
            cols = (center + offsets) % max(n_cols, 1)
        else:
            cols = torch.randperm(n_cols, device=device, dtype=torch.int64)[:row_nnz]
        cols, _ = torch.sort(cols)
        vals = torch.randn(row_nnz, dtype=dtype, device=device)
        data_parts.append(vals)
        col_parts.append(cols.to(index_dtype))
        indptr.append(indptr[-1] + row_nnz)

    if data_parts:
        data = torch.cat(data_parts)
        indices = torch.cat(col_parts)
    else:
        data = torch.empty(0, dtype=dtype, device=device)
        indices = torch.empty(0, dtype=index_dtype, device=device)
    indptr_tensor = torch.tensor(indptr, dtype=torch.int64, device=device)
    return data, indices, indptr_tensor


def _synthetic_row_lengths(case_name, n_rows, n_cols):
    if case_name == "short_uniform":
        return [min(8, n_cols)] * n_rows
    if case_name == "medium_uniform":
        return [min(64, n_cols)] * n_rows
    if case_name == "long_uniform":
        return [min(256, n_cols)] * n_rows
    if case_name == "heavy_tail":
        rows = [min(8, n_cols)] * n_rows
        long_rows = max(1, n_rows // 32)
        for row in range(long_rows):
            rows[row] = min(max(256, n_cols // 2), n_cols)
        return rows
    if case_name == "powerlaw_irregular":
        rows = []
        max_len = max(2, min(n_cols, 512))
        for row in range(n_rows):
            length = int(max(1, round(max_len / math.sqrt(row + 1))))
            rows.append(min(length, n_cols))
        return rows
    raise ValueError(f"Unknown synthetic case: {case_name}")


def build_synthetic_case(case_name, dtype, index_dtype, device):
    if case_name == "short_uniform":
        shape = (4096, 4096)
        pattern = "random"
    elif case_name == "medium_uniform":
        shape = (2048, 4096)
        pattern = "random"
    elif case_name == "long_uniform":
        shape = (1024, 4096)
        pattern = "random"
    elif case_name == "heavy_tail":
        shape = (4096, 8192)
        pattern = "random"
    elif case_name == "powerlaw_irregular":
        shape = (8192, 8192)
        pattern = "random"
    else:
        raise ValueError(f"Unknown synthetic case: {case_name}")

    n_rows, n_cols = shape
    row_lengths = _synthetic_row_lengths(case_name, n_rows, n_cols)
    data, indices, indptr = _build_csr_from_row_lengths(
        row_lengths,
        n_cols,
        dtype,
        index_dtype,
        device,
        pattern=pattern,
    )
    return data, indices, indptr, shape


def run_one_case(
    case_name,
    data,
    indices,
    indptr,
    shape,
    dtype,
    index_dtype,
    dense_cols,
    warmup,
    iters,
    seed,
    with_cusparse,
    return_details=False,
):
    device = data.device
    n_rows, n_cols = shape
    B = _seeded_dense_matrix((n_cols, dense_cols), dtype, device, seed)
    base_out, base_ms = _timed_spmm_base(data, indices, indptr, B, shape, warmup, iters)
    alpha_out, alpha_ms, prepared_alpha = _timed_alpha_spmm_alg1(
        data, indices, indptr, B, shape, warmup, iters
    )
    opt_alg2_out, opt_alg2_ms, prepared_alg2 = _timed_spmm_opt_alg2(
        data, indices, indptr, B, shape, warmup, iters
    )
    torch_out, torch_ms = _timed_torch_reference(data, indices, indptr, B, shape, dtype, warmup, iters)
    sparse_out, sparse_ms, sparse_name, sparse_reason = _timed_sparse_backend(
        data, indices, indptr, B, shape, warmup, iters, with_cusparse
    )

    profiles = {
        "base_vs_torch": _error_profile(base_out, torch_out, dtype),
        "alpha_alg1_vs_torch": _error_profile(alpha_out, torch_out, dtype),
        "opt_alg2_vs_torch": _error_profile(opt_alg2_out, torch_out, dtype),
        "base_vs_cusparse": _error_profile(base_out, sparse_out, dtype),
        "alpha_alg1_vs_cusparse": _error_profile(alpha_out, sparse_out, dtype),
        "opt_alg2_vs_cusparse": _error_profile(opt_alg2_out, sparse_out, dtype),
    }
    summary = {
        "matrix": case_name,
        "value_dtype": str(dtype).replace("torch.", ""),
        "dense_cols": dense_cols,
        "seed": seed,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": int(data.numel()),
        "avg_nnz_per_row": (float(data.numel()) / float(max(1, n_rows))),
        "max_row_nnz": int((indptr[1:] - indptr[:-1]).max().item()) if n_rows > 0 else 0,
        "base_ms": base_ms,
        "alpha_alg1_ms": alpha_ms,
        "opt_alg2_ms": opt_alg2_ms,
        "torch_ms": torch_ms,
        "cusparse_ms": sparse_ms,
        "base_vs_torch_err": profiles["base_vs_torch"]["global_err"],
        "alpha_alg1_vs_torch_err": profiles["alpha_alg1_vs_torch"]["global_err"],
        "opt_alg2_vs_torch_err": profiles["opt_alg2_vs_torch"]["global_err"],
        "base_vs_cusparse_err": profiles["base_vs_cusparse"]["global_err"],
        "alpha_alg1_vs_cusparse_err": profiles["alpha_alg1_vs_cusparse"]["global_err"],
        "opt_alg2_vs_cusparse_err": profiles["opt_alg2_vs_cusparse"]["global_err"],
        "base_status_vs_torch": profiles["base_vs_torch"]["status"],
        "alpha_alg1_status_vs_torch": profiles["alpha_alg1_vs_torch"]["status"],
        "opt_alg2_status_vs_torch": profiles["opt_alg2_vs_torch"]["status"],
        "base_status_vs_cusparse": profiles["base_vs_cusparse"]["status"],
        "alpha_alg1_status_vs_cusparse": profiles["alpha_alg1_vs_cusparse"]["status"],
        "opt_alg2_status_vs_cusparse": profiles["opt_alg2_vs_cusparse"]["status"],
        "matrix_status": (
            "PASS"
            if (
                profiles["alpha_alg1_vs_torch"]["status"] == "PASS"
                or profiles["alpha_alg1_vs_cusparse"]["status"] == "PASS"
            )
            else "FAIL"
        ),
    }
    if not return_details:
        return summary
    return {
        "summary": summary,
        "prepared_alpha": prepared_alpha,
        "prepared_alg2": prepared_alg2,
        "profiles": profiles,
        "sparse_backend_name": sparse_name,
        "sparse_backend_reason": sparse_reason,
    }


def _print_header():
    print("-" * 198)
    print(
        f"{'Matrix':<24} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} {'DenseN':>8} "
        f"{'Base(ms)':>10} {'Alpha(ms)':>10} {'Alg2(ms)':>10} {'PT(ms)':>9} {'CU(ms)':>9} "
        f"{'A/PT':>8} {'A/CU':>8} {'Err(A/PT)':>12} {'Err(A/CU)':>12} {'Status':>8}"
    )
    print("-" * 198)


def _print_summary_row(summary):
    def _fmt_ms(value):
        return "N/A" if value is None else f"{value:.4f}"

    def _fmt_speedup(other, current):
        if other is None or current is None or current <= 0:
            return "N/A"
        return f"{other / current:.2f}x"

    def _fmt_err(value):
        return "N/A" if value is None else f"{value:.2e}"

    print(
        f"{summary['matrix'][:23]:<24} {summary['n_rows']:>7} {summary['n_cols']:>7} {summary['nnz']:>10} "
        f"{summary['dense_cols']:>8} {_fmt_ms(summary['base_ms']):>10} {_fmt_ms(summary['alpha_alg1_ms']):>10} "
        f"{_fmt_ms(summary['opt_alg2_ms']):>10} {_fmt_ms(summary['torch_ms']):>9} {_fmt_ms(summary['cusparse_ms']):>9} "
        f"{_fmt_speedup(summary['torch_ms'], summary['alpha_alg1_ms']):>8} "
        f"{_fmt_speedup(summary['cusparse_ms'], summary['alpha_alg1_ms']):>8} "
        f"{_fmt_err(summary['alpha_alg1_vs_torch_err']):>12} {_fmt_err(summary['alpha_alg1_vs_cusparse_err']):>12} "
        f"{summary['matrix_status']:>8}"
    )


def _write_csv(path, rows):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})


def _write_launch_csv(path, rows):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LAUNCH_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})


def _build_launch_row(matrix_name, dtype_name, dense_cols, prepared, B):
    _, meta = fs.flagsparse_alpha_spmm_alg1(B=B, prepared=prepared, return_meta=True)
    return {
        "matrix": matrix_name,
        "device_name": meta["device_name"],
        "sm_count": meta["sm_count"],
        "dtype": dtype_name,
        "dense_cols": dense_cols,
        "warp_size": meta["warp_size"],
        "factor": meta["factor"],
        "block_size": meta["block_size"],
        "block_rows": meta["block_rows"],
        "block_cols": meta["block_cols"],
        "num_warps": meta["num_warps"],
        "num_stages": meta["num_stages"],
        "grid_m": meta["grid_m"],
        "grid_n": meta["grid_n"],
    }


def main():
    parser = argparse.ArgumentParser(description="Experimental AlphaSparse ALG1 Triton SpMM benchmark.")
    parser.add_argument("input_path", nargs="*", help=".mtx file or directory")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Write summary CSV; also writes launch CSV next to it as <stem>_launch.csv",
    )
    parser.add_argument("--dense-cols", type=int, default=DEFAULT_DENSE_COLS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--with-cusparse", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    rows = []
    launch_rows = []
    _print_header()
    if args.synthetic:
        synthetic_cases = [
            "short_uniform",
            "medium_uniform",
            "long_uniform",
            "heavy_tail",
            "powerlaw_irregular",
        ]
        for dense_cols in (4, 5, 12, 24, 48, 96):
            for case_name in synthetic_cases:
                data, indices, indptr, shape = build_synthetic_case(
                    case_name, torch.float32, torch.int32, device
                )
                result = run_one_case(
                    f"{case_name}_n{dense_cols}",
                    data,
                    indices,
                    indptr,
                    shape,
                    torch.float32,
                    torch.int32,
                    dense_cols,
                    args.warmup,
                    args.iters,
                    args.seed,
                    args.with_cusparse,
                    return_details=bool(args.csv),
                )
                if args.csv:
                    summary = result["summary"]
                    B = _seeded_dense_matrix((shape[1], dense_cols), torch.float32, device, args.seed)
                    launch_rows.append(
                        _build_launch_row(
                            summary["matrix"],
                            "float32",
                            dense_cols,
                            result["prepared_alpha"],
                            B,
                        )
                    )
                else:
                    summary = result
                rows.append(summary)
                _print_summary_row(summary)
    else:
        paths = _resolve_input_paths(args.input_path)
        for value_dtype in VALUE_DTYPES:
            for index_dtype in INDEX_DTYPES:
                for path in paths:
                    data, indices, indptr, shape = load_mtx_to_csr_torch(
                        path, dtype=value_dtype, device=device
                    )
                    indices = indices.to(index_dtype)
                    result = run_one_case(
                        os.path.basename(path),
                        data,
                        indices,
                        indptr,
                        shape,
                        value_dtype,
                        index_dtype,
                        args.dense_cols,
                        args.warmup,
                    args.iters,
                    args.seed,
                    args.with_cusparse,
                    return_details=bool(args.csv),
                )
                    if args.csv:
                        summary = result["summary"]
                        B = _seeded_dense_matrix((shape[1], args.dense_cols), value_dtype, device, args.seed)
                        launch_rows.append(
                            _build_launch_row(
                                summary["matrix"],
                                str(value_dtype).replace("torch.", ""),
                                args.dense_cols,
                                result["prepared_alpha"],
                                B,
                            )
                        )
                    else:
                        summary = result
                    rows.append(summary)
                    _print_summary_row(summary)
    print("-" * 198)
    if args.csv:
        csv_path = os.path.abspath(args.csv)
        stem, ext = os.path.splitext(csv_path)
        launch_csv_path = f"{stem}_launch{ext or '.csv'}"
        _write_csv(csv_path, rows)
        _write_launch_csv(launch_csv_path, launch_rows)
        print(f"Wrote {len(rows)} rows to {csv_path}")
        print(f"Wrote {len(launch_rows)} launch rows to {launch_csv_path}")


if __name__ == "__main__":
    main()
