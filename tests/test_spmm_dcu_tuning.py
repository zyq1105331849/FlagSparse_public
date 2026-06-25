"""DCU/HIP SpMM tuning sweep.

This file is a CLI-oriented test/benchmark helper.  It is intentionally kept in
``tests/`` so DCU runs can sweep many launch strategies without changing the
default library behavior first.

Example:
    python tests/test_spmm_dcu_tuning.py --input tests/data --format all --alg all
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import flagsparse as fs
import flagsparse.sparse_operations.spmm_coo as spmm_coo_ops
from flagsparse.sparse_operations._spmm_dcu_tuning import (
    SPMM_DCU_TUNING_STRATEGIES,
    get_spmm_backend_info,
    resolve_spmm_dcu_launch_strategy,
)

from test_spmm import (
    _build_dense_matrix as _build_csr_dense_matrix,
    _build_pytorch_reference as _build_csr_reference,
    load_mtx_to_csr_torch,
)
from test_spmm_coo import (
    _build_dense_matrix as _build_coo_dense_matrix,
    _build_pytorch_reference as _build_coo_reference,
    load_mtx_to_coo_torch,
)


FORMAT_NAMES = ("csr", "coo")
CSR_ALGS = (
    "csr_base",
    "spmm_csr_opt_alg1",
    "spmm_csr_opt_alg2",
    "tle",
    "tle_opt",
    "tle_opt2",
)
COO_ALGS = ("coo_rowrun", "coo_atomic")
ALL_ALGS = CSR_ALGS + COO_ALGS
DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
INDEX_DTYPE_MAP = {
    "int32": torch.int32,
    "int64": torch.int64,
}

RAW_FIELDS = [
    "matrix",
    "format",
    "alg",
    "route",
    "strategy",
    "value_dtype",
    "index_dtype",
    "dense_cols",
    "n_rows",
    "n_cols",
    "nnz",
    "mean_nnz_per_row",
    "max_row_nnz",
    "backend",
    "device_name",
    "device_warp_size",
    "block_n",
    "block_nnz",
    "num_warps",
    "num_stages",
    "warmup",
    "iters",
    "time_ms",
    "gpu_ms",
    "process_ms",
    "ref_time_ms",
    "speedup_vs_ref",
    "status",
    "skip_reason",
    "max_error",
]

SUMMARY_FIELDS = [
    "matrix",
    "format",
    "dense_cols",
    "value_dtype",
    "index_dtype",
    "best_alg",
    "best_route",
    "best_strategy",
    "best_time_ms",
    "baseline_time_ms",
    "speedup_vs_default",
    "status",
]


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _parse_names(value, allowed, option_name):
    token = str(value).strip().lower()
    if token == "all":
        return list(allowed)
    names = [item.strip().lower() for item in token.split(",") if item.strip()]
    invalid = [name for name in names if name not in allowed]
    if not names or invalid:
        raise ValueError(
            f"unsupported {option_name}: {', '.join(invalid or names)}; "
            f"allowed: all,{','.join(allowed)}"
        )
    return names


def _parse_dtypes(value):
    return [_resolve_mapping(name, DTYPE_MAP, "--value-dtypes") for name in _parse_names(value, tuple(DTYPE_MAP), "--value-dtypes")]


def _parse_index_dtypes(value):
    return [_resolve_mapping(name, INDEX_DTYPE_MAP, "--index-dtypes") for name in _parse_names(value, tuple(INDEX_DTYPE_MAP), "--index-dtypes")]


def _resolve_mapping(name, mapping, option_name):
    if name not in mapping:
        raise ValueError(f"unsupported {option_name}: {name}; allowed: {','.join(mapping)}")
    return mapping[name]


def _parse_int_csv(value, option_name):
    values = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed <= 0:
            raise ValueError(f"{option_name} values must be positive")
        values.append(parsed)
    if not values:
        raise ValueError(f"{option_name} must not be empty")
    return values


def _resolve_paths(inputs):
    paths = []
    for item in inputs:
        item = os.path.abspath(item)
        if os.path.isdir(item):
            paths.extend(sorted(glob.glob(os.path.join(item, "*.mtx"))))
        elif os.path.isfile(item) and item.lower().endswith(".mtx"):
            paths.append(item)
    deduped = []
    for path in paths:
        if path not in deduped:
            deduped.append(path)
    return deduped


def _cuda_event_benchmark(op, warmup, iters):
    out = None
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


def _scaled_error(candidate, reference, dtype):
    if candidate is None or reference is None:
        return None
    if candidate.numel() == 0:
        return 0.0
    if dtype in (torch.float32, torch.complex64):
        atol, rtol = 1.3e-6, 1e-3
    elif dtype in (torch.float64, torch.complex128):
        atol, rtol = 1e-7, 1e-5
    elif dtype == torch.float16:
        atol, rtol = 1e-3, 2e-3
    elif dtype == torch.bfloat16:
        atol, rtol = 0.016, 1e-1
    else:
        atol, rtol = 1e-6, 1e-5
    diff = torch.abs(candidate - reference).to(torch.float64)
    denom = (atol + rtol * torch.abs(reference)).to(torch.float64)
    return float(torch.max(diff / denom).item())


def _speedup(ref_ms, time_ms):
    if ref_ms is None or time_ms is None or time_ms <= 0:
        return None
    return float(ref_ms) / float(time_ms)


def _matrix_stats_from_csr(indptr, shape):
    n_rows = int(shape[0])
    nnz = int(indptr[-1].item()) if indptr.numel() else 0
    row_lengths = (indptr[1:].to(torch.int64) - indptr[:-1].to(torch.int64)) if n_rows else torch.tensor([], device=indptr.device)
    max_row_nnz = int(torch.max(row_lengths).item()) if row_lengths.numel() else 0
    mean_nnz = float(nnz) / float(max(1, n_rows))
    return nnz, mean_nnz, max_row_nnz


def _matrix_stats_from_coo(row, shape):
    n_rows = int(shape[0])
    nnz = int(row.numel())
    if nnz and n_rows:
        counts = torch.bincount(row.to(torch.int64), minlength=n_rows)
        max_row_nnz = int(torch.max(counts).item())
    else:
        max_row_nnz = 0
    mean_nnz = float(nnz) / float(max(1, n_rows))
    return nnz, mean_nnz, max_row_nnz


def _read_mtx_shape_header(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                n_rows = int(parts[0])
                n_cols = int(parts[1])
                nnz = int(parts[2]) if len(parts) > 2 else 0
                return (n_rows, n_cols), (nnz, float(nnz) / float(max(1, n_rows)), 0)
    return (0, 0), (0, 0.0, 0)


def _empty_row(path, fmt, alg, route, strategy, value_dtype, index_dtype, dense_cols, shape, stats, launch, warmup, iters, status, reason):
    n_rows, n_cols = shape
    nnz, mean_nnz, max_row_nnz = stats
    return {
        "matrix": os.path.basename(path),
        "format": fmt,
        "alg": alg,
        "route": route,
        "strategy": strategy,
        "value_dtype": _dtype_name(value_dtype),
        "index_dtype": _dtype_name(index_dtype),
        "dense_cols": dense_cols,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": nnz,
        "mean_nnz_per_row": mean_nnz,
        "max_row_nnz": max_row_nnz,
        "backend": launch.backend,
        "device_name": launch.device_name,
        "device_warp_size": launch.device_warp_size,
        "block_n": launch.block_n,
        "block_nnz": launch.block_nnz,
        "num_warps": launch.num_warps,
        "num_stages": launch.num_stages,
        "warmup": warmup,
        "iters": iters,
        "time_ms": None,
        "gpu_ms": None,
        "process_ms": None,
        "ref_time_ms": None,
        "speedup_vs_ref": None,
        "status": status,
        "skip_reason": reason,
        "max_error": None,
    }


def _tle_availability(alg):
    if alg == "tle":
        return fs.is_alpha_spmm_alg1_tle_available(), fs.alpha_spmm_alg1_tle_unavailable_reason()
    if alg == "tle_opt":
        return fs.is_alpha_spmm_alg1_tle_opt_available(), fs.alpha_spmm_alg1_tle_opt_unavailable_reason()
    if alg == "tle_opt2":
        return fs.is_alpha_spmm_alg1_tle_opt2_available(), fs.alpha_spmm_alg1_tle_opt2_unavailable_reason()
    return True, ""


def _run_csr_alg(data, indices, indptr, B, shape, alg, launch):
    if alg == "csr_base":
        return fs.flagsparse_spmm_csr(
            data,
            indices,
            indptr,
            B,
            shape,
            block_n=launch.block_n,
            block_nnz=launch.block_nnz,
            num_warps=launch.num_warps,
            num_stages=launch.num_stages,
        )
    if alg == "spmm_csr_opt_alg1":
        return fs.flagsparse_spmm_csr_opt_alg1(data=data, indices=indices, indptr=indptr, B=B, shape=shape)
    if alg == "spmm_csr_opt_alg2":
        return fs.flagsparse_spmm_csr_opt_alg2(data=data, indices=indices, indptr=indptr, B=B, shape=shape)
    if alg == "tle":
        return fs.flagsparse_alpha_spmm_alg1_tle(data=data, indices=indices, indptr=indptr, B=B, shape=shape)
    if alg == "tle_opt":
        return fs.flagsparse_alpha_spmm_alg1_tle_opt(data=data, indices=indices, indptr=indptr, B=B, shape=shape)
    if alg == "tle_opt2":
        return fs.flagsparse_alpha_spmm_alg1_tle_opt2(data=data, indices=indices, indptr=indptr, B=B, shape=shape)
    raise ValueError(f"unsupported CSR alg: {alg}")


def _run_one_csr(path, value_dtype, index_dtype, dense_cols, alg, strategy, warmup, iters, run_ref):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        data, indices, indptr, shape = load_mtx_to_csr_torch(path, dtype=value_dtype, device=device)
    except Exception as exc:
        shape, stats = _read_mtx_shape_header(path)
        launch = resolve_spmm_dcu_launch_strategy(
            strategy,
            n_dense_cols=dense_cols,
            max_row_nnz=stats[2],
            nnz=stats[0],
            fmt="csr",
            dtype=value_dtype,
            device=None,
        )
        return _empty_row(path, "csr", alg, "", strategy, value_dtype, index_dtype, dense_cols, shape, stats, launch, warmup, iters, "SKIP", f"load failed: {exc}")
    indices = indices.to(index_dtype)
    indptr = indptr.to(index_dtype)
    stats = _matrix_stats_from_csr(indptr, shape)
    launch = resolve_spmm_dcu_launch_strategy(
        strategy,
        n_dense_cols=dense_cols,
        max_row_nnz=stats[2],
        nnz=stats[0],
        fmt="csr",
        dtype=value_dtype,
        device=data.device if data.is_cuda else None,
    )
    route = ""
    if not torch.cuda.is_available():
        return _empty_row(path, "csr", alg, route, strategy, value_dtype, index_dtype, dense_cols, shape, stats, launch, warmup, iters, "SKIP", "CUDA/HIP device is not available")
    if alg.startswith("tle"):
        ok, reason = _tle_availability(alg)
        if not ok:
            return _empty_row(path, "csr", alg, route, strategy, value_dtype, index_dtype, dense_cols, shape, stats, launch, warmup, iters, "SKIP", reason or "TLE algorithm is unavailable")
    B = _build_csr_dense_matrix(shape[1], dense_cols, value_dtype, data.device)
    ref = None
    ref_ms = None
    if run_ref:
        ref, ref_op, _ = _build_csr_reference(data, indices, indptr, shape, B)
        _, ref_ms = _cuda_event_benchmark(ref_op, warmup, iters)
    process_ms = 0.0
    try:
        t0 = time.perf_counter()
        out, gpu_ms = _cuda_event_benchmark(lambda: _run_csr_alg(data, indices, indptr, B, shape, alg, launch), warmup, iters)
        process_ms = (time.perf_counter() - t0) * 1000.0 - gpu_ms
        max_error = _scaled_error(out, ref, value_dtype) if ref is not None else None
        status = "PASS" if max_error is None or max_error <= 1.0 else "FAIL"
        reason = (
            "strategy not injected into this CSR algorithm yet"
            if alg != "csr_base" and strategy != "default"
            else ""
        )
    except Exception as exc:
        out = None
        gpu_ms = None
        max_error = None
        status = "ERROR"
        reason = str(exc)
    row = _empty_row(path, "csr", alg, route, strategy, value_dtype, index_dtype, dense_cols, shape, stats, launch, warmup, iters, status, reason)
    row.update({
        "time_ms": gpu_ms,
        "gpu_ms": gpu_ms,
        "process_ms": process_ms if gpu_ms is not None else None,
        "ref_time_ms": ref_ms,
        "speedup_vs_ref": _speedup(ref_ms, gpu_ms),
        "max_error": max_error,
    })
    return row


def _run_one_coo(path, value_dtype, index_dtype, dense_cols, alg, strategy, warmup, iters, run_ref):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        data, row_idx, col_idx, shape = load_mtx_to_coo_torch(path, dtype=value_dtype, device=device)
    except Exception as exc:
        shape, stats = _read_mtx_shape_header(path)
        launch = resolve_spmm_dcu_launch_strategy(
            strategy,
            n_dense_cols=dense_cols,
            max_row_nnz=stats[2],
            nnz=stats[0],
            fmt="coo",
            dtype=value_dtype,
            device=None,
        )
        route = "rowrun" if alg == "coo_rowrun" else "atomic"
        return _empty_row(path, "coo", alg, route, strategy, value_dtype, index_dtype, dense_cols, shape, stats, launch, warmup, iters, "SKIP", f"load failed: {exc}")
    row_idx = row_idx.to(index_dtype)
    col_idx = col_idx.to(index_dtype)
    stats = _matrix_stats_from_coo(row_idx, shape)
    launch = resolve_spmm_dcu_launch_strategy(
        strategy,
        n_dense_cols=dense_cols,
        max_row_nnz=stats[2],
        nnz=stats[0],
        fmt="coo",
        dtype=value_dtype,
        device=data.device if data.is_cuda else None,
    )
    route = "rowrun" if alg == "coo_rowrun" else "atomic"
    if not torch.cuda.is_available():
        return _empty_row(path, "coo", alg, route, strategy, value_dtype, index_dtype, dense_cols, shape, stats, launch, warmup, iters, "SKIP", "CUDA/HIP device is not available")

    B = _build_coo_dense_matrix(shape[1], dense_cols, value_dtype, data.device)
    ref = None
    ref_ms = None
    if run_ref:
        ref, ref_op, _, _ = _build_coo_reference(data, row_idx, col_idx, shape, B)
        _, ref_ms = _cuda_event_benchmark(ref_op, warmup, iters)
    process_ms = 0.0
    try:
        run = lambda: spmm_coo_ops._run_spmm_coo_route(
            data,
            row_idx,
            col_idx,
            B,
            shape,
            block_n=launch.block_n,
            block_nnz=launch.block_nnz,
            route=route,
        )
        t0 = time.perf_counter()
        out, gpu_ms = _cuda_event_benchmark(run, warmup, iters)
        process_ms = (time.perf_counter() - t0) * 1000.0 - gpu_ms
        max_error = _scaled_error(out, ref, value_dtype) if ref is not None else None
        status = "PASS" if max_error is None or max_error <= 1.0 else "FAIL"
        reason = ""
    except Exception as exc:
        gpu_ms = None
        max_error = None
        status = "ERROR"
        reason = str(exc)
    row = _empty_row(path, "coo", alg, route, strategy, value_dtype, index_dtype, dense_cols, shape, stats, launch, warmup, iters, status, reason)
    row.update({
        "time_ms": gpu_ms,
        "gpu_ms": gpu_ms,
        "process_ms": process_ms if gpu_ms is not None else None,
        "ref_time_ms": ref_ms,
        "speedup_vs_ref": _speedup(ref_ms, gpu_ms),
        "max_error": max_error,
    })
    return row


def _write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})


def _best_summary(rows):
    grouped = {}
    for row in rows:
        key = (
            row["matrix"],
            row["format"],
            row["dense_cols"],
            row["value_dtype"],
            row["index_dtype"],
        )
        grouped.setdefault(key, []).append(row)
    summary = []
    for key, group in grouped.items():
        matrix, fmt, dense_cols, value_dtype, index_dtype = key
        pass_rows = [row for row in group if row.get("status") == "PASS" and row.get("time_ms") not in (None, "")]
        baseline_candidates = [
            row for row in pass_rows
            if row.get("strategy") == "default"
            and ((fmt == "csr" and row.get("alg") == "csr_base") or (fmt == "coo" and row.get("alg") == "coo_rowrun"))
        ]
        baseline = min(baseline_candidates, key=lambda row: float(row["time_ms"])) if baseline_candidates else None
        best = min(pass_rows, key=lambda row: float(row["time_ms"])) if pass_rows else None
        baseline_ms = float(baseline["time_ms"]) if baseline else None
        best_ms = float(best["time_ms"]) if best else None
        summary.append({
            "matrix": matrix,
            "format": fmt,
            "dense_cols": dense_cols,
            "value_dtype": value_dtype,
            "index_dtype": index_dtype,
            "best_alg": "" if best is None else best.get("alg", ""),
            "best_route": "" if best is None else best.get("route", ""),
            "best_strategy": "" if best is None else best.get("strategy", ""),
            "best_time_ms": best_ms,
            "baseline_time_ms": baseline_ms,
            "speedup_vs_default": _speedup(baseline_ms, best_ms),
            "status": "PASS" if best is not None else "NO_PASS",
        })
    return summary


def _expand_algs(formats, alg_names):
    if alg_names == ["all"]:
        names = []
        if "csr" in formats:
            names.extend(CSR_ALGS)
        if "coo" in formats:
            names.extend(COO_ALGS)
        return names
    expanded = []
    for name in alg_names:
        if name in CSR_ALGS or name in COO_ALGS:
            expanded.append(name)
        else:
            raise ValueError(f"unsupported --alg: {name}; allowed: all,{','.join(ALL_ALGS)}")
    return expanded


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sweep DCU/HIP-aware SpMM tuning strategies.")
    parser.add_argument("--format", default="all", choices=("csr", "coo", "all"))
    parser.add_argument("--alg", default="all", help=f"all or comma list from: {','.join(ALL_ALGS)}")
    parser.add_argument("--strategy", default="all", help=f"all or comma list from: {','.join(SPMM_DCU_TUNING_STRATEGIES)}")
    parser.add_argument("--dense-cols", default="8,16,32,64,128")
    parser.add_argument("--value-dtypes", default="float32,float64")
    parser.add_argument("--index-dtypes", default="int32,int64")
    parser.add_argument("--input", nargs="*", default=[str(_PROJECT_ROOT / "tests" / "data")])
    parser.add_argument("--out-dir", default=str(_PROJECT_ROOT / "tests" / "results" / "spmm_dcu_tuning"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--no-ref", action="store_true", help="Skip PyTorch reference timing and correctness checks.")
    parser.add_argument(
        "--ref-mode",
        default="baseline",
        choices=("none", "baseline", "all"),
        help=(
            "Reference policy: none skips PyTorch reference; baseline checks only "
            "default csr_base/coo_rowrun rows; all checks every row. Default: baseline."
        ),
    )
    args = parser.parse_args(argv)

    formats = list(FORMAT_NAMES) if args.format == "all" else [args.format]
    alg_tokens = [token.strip().lower() for token in str(args.alg).split(",") if token.strip()]
    algs = _expand_algs(formats, alg_tokens or ["all"])
    strategies = (
        list(SPMM_DCU_TUNING_STRATEGIES)
        if str(args.strategy).strip().lower() == "all"
        else _parse_names(args.strategy, SPMM_DCU_TUNING_STRATEGIES, "--strategy")
    )
    dense_cols_values = _parse_int_csv(args.dense_cols, "--dense-cols")
    value_dtypes = _parse_dtypes(args.value_dtypes)
    index_dtypes = _parse_index_dtypes(args.index_dtypes)
    paths = _resolve_paths(args.input)
    if not paths:
        raise FileNotFoundError(f"no .mtx files found from --input: {args.input}")
    ref_mode = "none" if args.no_ref else args.ref_mode

    backend = get_spmm_backend_info()
    planned_rows = 0
    for alg in algs:
        if alg in CSR_ALGS and "csr" in formats:
            planned_rows += len(paths) * len(value_dtypes) * len(index_dtypes) * len(dense_cols_values) * len(strategies)
        if alg in COO_ALGS and "coo" in formats:
            planned_rows += len(paths) * len(value_dtypes) * len(index_dtypes) * len(dense_cols_values) * len(strategies)
    print(
        f"SpMM DCU tuning sweep: backend={backend['backend']} "
        f"device={backend['device_name'] or 'N/A'} warp={backend['device_warp_size']}",
        flush=True,
    )
    print(
        f"Plan: matrices={len(paths)} formats={','.join(formats)} algs={len(algs)} "
        f"strategies={len(strategies)} dense_cols={dense_cols_values} "
        f"value_dtypes={[ _dtype_name(v) for v in value_dtypes ]} "
        f"index_dtypes={[ _dtype_name(v) for v in index_dtypes ]} "
        f"rows={planned_rows} ref_mode={ref_mode} warmup={args.warmup} iters={args.iters}",
        flush=True,
    )
    csr_rows = []
    coo_rows = []
    row_index = 0
    for path in paths:
        for value_dtype in value_dtypes:
            for index_dtype in index_dtypes:
                for dense_cols in dense_cols_values:
                    for strategy in strategies:
                        for alg in algs:
                            if alg in CSR_ALGS and "csr" in formats:
                                row_index += 1
                                run_ref = ref_mode == "all" or (
                                    ref_mode == "baseline"
                                    and alg == "csr_base"
                                    and strategy == "default"
                                )
                                print(
                                    f"[{row_index}/{planned_rows}] START CSR "
                                    f"matrix={os.path.basename(path)} alg={alg} strategy={strategy} "
                                    f"dtype={_dtype_name(value_dtype)} index={_dtype_name(index_dtype)} "
                                    f"N={dense_cols} ref={'yes' if run_ref else 'no'}",
                                    flush=True,
                                )
                                row = _run_one_csr(
                                    path,
                                    value_dtype,
                                    index_dtype,
                                    dense_cols,
                                    alg,
                                    strategy,
                                    args.warmup,
                                    args.iters,
                                    run_ref,
                                )
                                csr_rows.append(row)
                                print(
                                    f"[{row_index}/{planned_rows}] DONE  CSR {row['matrix']} "
                                    f"{alg} {strategy} N={dense_cols}: {row['status']} "
                                    f"{row['time_ms'] or ''} {row['skip_reason'] or ''}",
                                    flush=True,
                                )
                            if alg in COO_ALGS and "coo" in formats:
                                row_index += 1
                                run_ref = ref_mode == "all" or (
                                    ref_mode == "baseline"
                                    and alg == "coo_rowrun"
                                    and strategy == "default"
                                )
                                print(
                                    f"[{row_index}/{planned_rows}] START COO "
                                    f"matrix={os.path.basename(path)} alg={alg} strategy={strategy} "
                                    f"dtype={_dtype_name(value_dtype)} index={_dtype_name(index_dtype)} "
                                    f"N={dense_cols} ref={'yes' if run_ref else 'no'}",
                                    flush=True,
                                )
                                row = _run_one_coo(
                                    path,
                                    value_dtype,
                                    index_dtype,
                                    dense_cols,
                                    alg,
                                    strategy,
                                    args.warmup,
                                    args.iters,
                                    run_ref,
                                )
                                coo_rows.append(row)
                                print(
                                    f"[{row_index}/{planned_rows}] DONE  COO {row['matrix']} "
                                    f"{alg} {strategy} N={dense_cols}: {row['status']} "
                                    f"{row['time_ms'] or ''} {row['skip_reason'] or ''}",
                                    flush=True,
                                )

    out_dir = os.path.abspath(args.out_dir)
    csr_path = os.path.join(out_dir, "spmm_dcu_tuning_csr_raw.csv")
    coo_path = os.path.join(out_dir, "spmm_dcu_tuning_coo_raw.csv")
    summary_path = os.path.join(out_dir, "spmm_dcu_tuning_best_summary.csv")
    _write_csv(csr_path, csr_rows, RAW_FIELDS)
    _write_csv(coo_path, coo_rows, RAW_FIELDS)
    _write_csv(summary_path, _best_summary(csr_rows + coo_rows), SUMMARY_FIELDS)
    print(f"Wrote CSR raw: {csr_path}", flush=True)
    print(f"Wrote COO raw: {coo_path}", flush=True)
    print(f"Wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
