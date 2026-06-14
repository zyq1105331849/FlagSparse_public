"""
Diagnose CSR SpMM route error across all registered CSR algorithms.

The primary reference is a CPU CSR row loop in the tested dtype.  Separate
float64-accumulation references are kept only as comparison points, not as the
default f32 pass/fail reference.

Usage:
    python tests/diagnose_spmm_csr_error.py tests/data/wave.mtx --out-dir diag_smoke --no-cusparse
    python tests/diagnose_spmm_csr_error.py path/to/mtx_dir --dtype float32,float64 --op non --out-dir diag_all
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as fs

from test_spmm import _build_dense_matrix, _build_pytorch_reference, load_mtx_to_csr_torch
from test_spmm_csr import _materialize_dense_layout_for_test, _time_cusparse


DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}
DEFAULT_DTYPES = ("float32", "float64")
DEFAULT_OPS = ("non",)
ALL_OPS = ("non", "trans", "conj")

SUMMARY_FIELDS = [
    "matrix",
    "dtype",
    "op",
    "layout",
    "alg",
    "n_rows",
    "n_cols",
    "nnz",
    "avg_nnz_per_row",
    "max_row_nnz",
    "dense_cols",
    "b_stride",
    "c_stride",
    "ms",
    "gpu_ms",
    "process_cpu_ms",
    "process_gpu_ms",
    "compute_ms",
    "cpu_ref_kind",
    "cpu_ref_status",
    "cpu_ref_reason",
    "cpu_f64_ref_err_vs_cpu_ref",
    "cpu_f64_ref_reason",
    "torch_f64_cast_ref_err_vs_cpu_ref",
    "torch_f64_cast_ref_reason",
    "torch_native_ref_err_vs_cpu_ref",
    "torch_native_ref_reason",
    "cusparse_ref_err_vs_cpu_ref",
    "err_vs_cpu_ref",
    "err_vs_cpu_f64_ref",
    "err_vs_torch_f64_cast_ref",
    "err_vs_torch_native_ref",
    "err_vs_cusparse_ref",
    "err_vs_csr_base",
    "max_abs_vs_cpu_ref",
    "mean_abs_vs_cpu_ref",
    "row_p50_vs_cpu_ref",
    "row_p90_vs_cpu_ref",
    "row_p99_vs_cpu_ref",
    "row_max_vs_cpu_ref",
    "rows_gt_0_5_vs_cpu_ref",
    "rows_gt_1_vs_cpu_ref",
    "rows_gt_2_vs_cpu_ref",
    "rows_gt_10_vs_cpu_ref",
    "worst_row",
    "worst_col",
    "worst_row_nnz",
    "worst_bucket",
    "status",
    "reason",
    "cusparse_ms",
    "cusparse_reason",
]

WORST_ROW_FIELDS = [
    "matrix",
    "dtype",
    "op",
    "alg",
    "row",
    "row_nnz",
    "bucket",
    "row_max_ratio_vs_cpu_ref",
    "row_mean_ratio_vs_cpu_ref",
    "worst_col",
    "candidate_value",
    "cpu_ref_value",
    "cpu_f64_ref_value",
    "torch_f64_cast_ref_value",
    "torch_native_ref_value",
    "cusparse_ref_value",
    "abs_diff_vs_cpu_ref",
]

WORST_VALUE_FIELDS = [
    "matrix",
    "dtype",
    "op",
    "alg",
    "rank",
    "row",
    "col",
    "row_nnz",
    "bucket",
    "scaled_ratio_vs_cpu_ref",
    "abs_diff_vs_cpu_ref",
    "candidate_value",
    "cpu_ref_value",
    "cpu_f64_ref_value",
    "torch_f64_cast_ref_value",
    "torch_native_ref_value",
    "cusparse_ref_value",
]

WORST_CONTRIB_FIELDS = [
    "matrix",
    "dtype",
    "op",
    "alg",
    "rank",
    "row",
    "col",
    "row_nnz",
    "bucket",
    "scaled_ratio_vs_cpu_ref",
    "candidate_value",
    "cpu_ref_value",
    "cpu_f64_ref_value",
    "torch_f64_cast_ref_value",
    "torch_native_ref_value",
    "cusparse_ref_value",
    "tolerance_atol",
    "tolerance_rtol",
    "tolerance_denom",
    "contribution_count",
    "seq_f32_sum",
    "reverse_f32_sum",
    "pairwise_f32_sum",
    "abs_asc_f32_sum",
    "abs_desc_f32_sum",
    "f64_sum",
    "f64_seq_sum",
    "f32_cast_of_f64_sum",
    "seq_f32_scaled_ratio_vs_cpu_ref",
    "reverse_f32_scaled_ratio_vs_cpu_ref",
    "pairwise_f32_scaled_ratio_vs_cpu_ref",
    "abs_asc_f32_scaled_ratio_vs_cpu_ref",
    "abs_desc_f32_scaled_ratio_vs_cpu_ref",
    "candidate_minus_cpu_ref",
    "candidate_minus_seq_f32",
    "candidate_minus_reverse_f32",
    "candidate_minus_pairwise_f32",
    "candidate_minus_abs_asc_f32",
    "candidate_minus_abs_desc_f32",
    "sum_abs_products",
    "abs_sum",
    "cancellation_ratio",
    "positive_sum",
    "negative_sum",
    "positive_count",
    "negative_count",
    "zero_count",
    "max_abs_product",
    "min_nonzero_abs_product",
    "product_abs_dynamic_range",
    "mean_abs_product",
    "max_abs_a",
    "max_abs_b",
    "top_abs_products",
]

BUCKET_FIELDS = [
    "matrix",
    "dtype",
    "op",
    "alg",
    "bucket",
    "row_count",
    "row_nnz_min",
    "row_nnz_mean",
    "row_nnz_max",
]

LAUNCH_FIELDS = [
    "matrix",
    "dtype",
    "op",
    "alg",
    "launch_config_scope",
    "launch_config_count",
    "bucket_count",
    "long_row_count",
    "long_part_count",
    "num_warps",
    "num_stages",
    "block_n",
    "block_nnz",
    "block_rows",
    "block_cols",
    "grid_m",
    "grid_n",
    "warp_size",
    "factor",
    "launch_version",
    "dense_layout",
    "b_stride",
    "c_stride",
    "output_layout",
]


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _fmt(value, digits=4):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _stride_string(tensor):
    if tensor is None:
        return ""
    return "x".join(str(int(v)) for v in tensor.stride())


def _reference_tolerance(dtype):
    # Keep this diagnostic aligned with tests/test_spmm_csr.py.  The return
    # order is (atol, rtol), matching the scaled-error denominator below.
    if dtype in (torch.float32, torch.complex64):
        return 1.3e-6, 1e-3
    if dtype in (torch.float64, torch.complex128):
        return 1e-7, 1e-5
    if dtype == torch.float16:
        return 1e-3, 2e-3
    if dtype == torch.bfloat16:
        return 0.016, 1e-1
    return 1e-6, 1e-5


def _status_from_error(error_value):
    if error_value is None:
        return "SKIP"
    return "PASS" if float(error_value) <= 1.0 else "FAIL"


def _resolve_input_paths(input_paths):
    paths = []
    for path in input_paths:
        if os.path.isfile(path) and path.lower().endswith(".mtx"):
            paths.append(os.path.abspath(path))
        elif os.path.isdir(path):
            paths.extend(sorted(glob.glob(os.path.join(path, "*.mtx"))))
    return paths


def _parse_csv_names(value, allowed, option_name):
    token = str(value).strip().lower()
    if token == "all":
        return list(allowed)
    names = [part.strip().lower() for part in token.split(",") if part.strip()]
    if not names:
        raise ValueError(f"{option_name} must not be empty")
    invalid = [name for name in names if name not in allowed]
    if invalid:
        raise ValueError(
            f"unsupported {option_name}: {', '.join(invalid)}; allowed: all,{','.join(allowed)}"
        )
    return names


def _parse_algs(value):
    token = str(value).strip().lower()
    if token == "all":
        return ["all"]
    allowed = set(fs.SPMM_CSR_ALGORITHMS)
    names = [part.strip().lower() for part in token.split(",") if part.strip()]
    if not names:
        raise ValueError("--alg must not be empty")
    invalid = [name for name in names if name not in allowed]
    if invalid:
        raise ValueError(
            f"unsupported --alg: {', '.join(invalid)}; allowed: all,{','.join(sorted(allowed))}"
        )
    return names


def _expand_algs(alg_names, op, dtype):
    expanded = []
    for alg in alg_names:
        if alg == "all":
            expanded.extend(fs.list_spmm_csr_algorithms(op=op, dtype=dtype))
        else:
            expanded.append(alg)
    deduped = []
    for alg in expanded:
        if alg not in deduped:
            deduped.append(alg)
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


def _to_cpu_compare(tensor):
    if tensor is None:
        return None
    return tensor.detach().cpu()


def _scalar(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.item()
    return value


def _error_profile(candidate, reference, dtype):
    if candidate is None or reference is None:
        return {
            "global_err": None,
            "status": "SKIP",
            "max_abs": None,
            "mean_abs": None,
            "row_max_ratio": None,
            "row_mean_ratio": None,
            "row_worst_col": None,
            "row_p50": None,
            "row_p90": None,
            "row_p99": None,
            "row_max": None,
            "rows_gt_0_5": None,
            "rows_gt_1": None,
            "rows_gt_2": None,
            "rows_gt_10": None,
            "worst_row": None,
            "worst_col": None,
        }
    candidate = candidate.to(torch.float64)
    reference = reference.to(torch.float64)
    if candidate.shape != reference.shape:
        raise ValueError(f"shape mismatch: {candidate.shape} vs {reference.shape}")
    if candidate.numel() == 0:
        row_count = int(candidate.shape[0]) if candidate.ndim > 0 else 0
        return {
            "global_err": 0.0,
            "status": "PASS",
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "row_max_ratio": torch.zeros(row_count, dtype=torch.float64),
            "row_mean_ratio": torch.zeros(row_count, dtype=torch.float64),
            "row_worst_col": torch.zeros(row_count, dtype=torch.int64),
            "row_p50": 0.0,
            "row_p90": 0.0,
            "row_p99": 0.0,
            "row_max": 0.0,
            "rows_gt_0_5": 0,
            "rows_gt_1": 0,
            "rows_gt_2": 0,
            "rows_gt_10": 0,
            "worst_row": 0,
            "worst_col": 0,
        }
    atol, rtol = _reference_tolerance(dtype)
    diff = torch.abs(candidate - reference)
    denom = atol + rtol * torch.abs(reference)
    ratio = diff / denom
    row_max_ratio, row_worst_col = torch.max(ratio, dim=1)
    row_mean_ratio = torch.mean(ratio, dim=1)
    global_err = float(torch.max(row_max_ratio).item()) if row_max_ratio.numel() else 0.0
    worst_row = int(torch.argmax(row_max_ratio).item()) if row_max_ratio.numel() else 0
    worst_col = int(row_worst_col[worst_row].item()) if row_worst_col.numel() else 0
    return {
        "global_err": global_err,
        "status": _status_from_error(global_err),
        "max_abs": float(torch.max(diff).item()) if diff.numel() else 0.0,
        "mean_abs": float(torch.mean(diff).item()) if diff.numel() else 0.0,
        "row_max_ratio": row_max_ratio,
        "row_mean_ratio": row_mean_ratio,
        "row_worst_col": row_worst_col,
        "row_p50": float(torch.quantile(row_max_ratio, 0.50).item()) if row_max_ratio.numel() else 0.0,
        "row_p90": float(torch.quantile(row_max_ratio, 0.90).item()) if row_max_ratio.numel() else 0.0,
        "row_p99": float(torch.quantile(row_max_ratio, 0.99).item()) if row_max_ratio.numel() else 0.0,
        "row_max": global_err,
        "rows_gt_0_5": int(torch.sum(row_max_ratio > 0.5).item()),
        "rows_gt_1": int(torch.sum(row_max_ratio > 1.0).item()),
        "rows_gt_2": int(torch.sum(row_max_ratio > 2.0).item()),
        "rows_gt_10": int(torch.sum(row_max_ratio > 10.0).item()),
        "worst_row": worst_row,
        "worst_col": worst_col,
    }


def _cpu_csr_spmm_reference(data, indices, indptr, shape, B, dtype, op, accumulate_dtype):
    data_acc = data.detach().cpu().to(accumulate_dtype)
    indices64 = indices.detach().cpu().to(torch.int64)
    indptr64 = indptr.detach().cpu().to(torch.int64)
    B_acc = B.detach().cpu().to(accumulate_dtype)
    n_rows, n_cols = int(shape[0]), int(shape[1])
    dense_cols = int(B_acc.shape[1])
    if op == "non":
        out = torch.zeros((n_rows, dense_cols), dtype=accumulate_dtype)
        for row in range(n_rows):
            start = int(indptr64[row].item())
            end = int(indptr64[row + 1].item())
            if start == end:
                continue
            cols = indices64[start:end]
            vals = data_acc[start:end].view(-1, 1)
            out[row] = torch.sum(vals * B_acc.index_select(0, cols), dim=0)
    elif op in ("trans", "conj"):
        out = torch.zeros((n_cols, dense_cols), dtype=accumulate_dtype)
        for row in range(n_rows):
            start = int(indptr64[row].item())
            end = int(indptr64[row + 1].item())
            if start == end:
                continue
            cols = indices64[start:end]
            vals = data_acc[start:end].view(-1, 1)
            contrib = vals * B_acc[row].view(1, -1)
            out.index_add_(0, cols, contrib)
    else:
        raise ValueError(f"unsupported op: {op}")
    return out.to(dtype), None


def _torch_native_reference(data, indices, indptr, shape, B, op):
    ref, _, _ = _build_pytorch_reference(data, indices, indptr, shape, B, op=op)
    if data.dtype == torch.float32:
        ref_dtype = data.dtype
        sparse = torch.sparse_csr_tensor(
            indptr.to(torch.int64),
            indices.to(torch.int64),
            data,
            size=shape,
            device=data.device,
        )
        if op == "non":
            return torch.sparse.mm(sparse, B).to(ref_dtype)
        if op == "trans":
            return torch.sparse.mm(sparse.transpose(0, 1), B).to(ref_dtype)
        if op == "conj":
            return torch.sparse.mm(sparse.transpose(0, 1), B).to(ref_dtype)
    return ref


def _build_row_bucket_map(row_count, diagnostics):
    # Current route diagnostics expose only aggregate fields.  Keep this helper
    # tolerant so future bucket row metadata can populate row labels immediately.
    labels = [""] * int(row_count)
    rows_by_bucket = diagnostics.get("rows_by_bucket") if diagnostics else None
    if isinstance(rows_by_bucket, dict):
        for label, rows in rows_by_bucket.items():
            for row in rows:
                row = int(row)
                if 0 <= row < row_count:
                    labels[row] = str(label)
    return labels


def _diagnostic_launch_row(matrix_name, dtype_name, op, alg, diagnostics):
    row = {"matrix": matrix_name, "dtype": dtype_name, "op": op, "alg": alg}
    diagnostics = diagnostics or {}
    for field in LAUNCH_FIELDS:
        if field not in row:
            value = diagnostics.get(field)
            if isinstance(value, (tuple, list)):
                value = "x".join(str(int(v)) for v in value)
            row[field] = value
    return row


def _bucket_rows_from_diagnostics(matrix_name, dtype_name, op, alg, diagnostics, row_lengths):
    diagnostics = diagnostics or {}
    rows = []
    bucket_stats = diagnostics.get("bucket_stats")
    if isinstance(bucket_stats, list):
        for bucket in bucket_stats:
            rows.append(
                {
                    "matrix": matrix_name,
                    "dtype": dtype_name,
                    "op": op,
                    "alg": alg,
                    "bucket": bucket.get("bucket") or bucket.get("label"),
                    "row_count": bucket.get("row_count"),
                    "row_nnz_min": bucket.get("row_nnz_min"),
                    "row_nnz_mean": bucket.get("row_nnz_mean"),
                    "row_nnz_max": bucket.get("row_nnz_max"),
                }
            )
    if rows:
        return rows
    label = diagnostics.get("launch_version") or diagnostics.get("launch_config_scope") or "matrix"
    if row_lengths.numel() == 0:
        return [
            {
                "matrix": matrix_name,
                "dtype": dtype_name,
                "op": op,
                "alg": alg,
                "bucket": label,
                "row_count": 0,
                "row_nnz_min": 0,
                "row_nnz_mean": 0.0,
                "row_nnz_max": 0,
            }
        ]
    row_lengths_f = row_lengths.to(torch.float64).cpu()
    return [
        {
            "matrix": matrix_name,
            "dtype": dtype_name,
            "op": op,
            "alg": alg,
            "bucket": label,
            "row_count": int(row_lengths_f.numel()),
            "row_nnz_min": int(torch.min(row_lengths_f).item()),
            "row_nnz_mean": float(torch.mean(row_lengths_f).item()),
            "row_nnz_max": int(torch.max(row_lengths_f).item()),
        }
    ]


def _build_worst_rows(matrix_name, dtype_name, op, alg, candidate, refs, profile, row_lengths, buckets, top_rows):
    row_max = profile.get("row_max_ratio")
    if row_max is None or row_max.numel() == 0 or top_rows <= 0:
        return []
    order = torch.argsort(row_max, descending=True)[: int(top_rows)]
    row_mean = profile["row_mean_ratio"]
    worst_col = profile["row_worst_col"]
    out = []
    candidate = _to_cpu_compare(candidate)
    cpu_ref = refs.get("cpu")
    cpu_f64 = refs.get("cpu_f64")
    torch_f64_cast = refs.get("torch_f64_cast")
    torch_native = refs.get("torch_native")
    cusparse = refs.get("cusparse")
    row_lengths_cpu = row_lengths.to(torch.int64).cpu()
    for row_id in order.to(torch.int64).tolist():
        col = int(worst_col[row_id].item())
        out.append(
            {
                "matrix": matrix_name,
                "dtype": dtype_name,
                "op": op,
                "alg": alg,
                "row": row_id,
                "row_nnz": int(row_lengths_cpu[row_id].item()) if row_id < row_lengths_cpu.numel() else "",
                "bucket": buckets[row_id] if row_id < len(buckets) else "",
                "row_max_ratio_vs_cpu_ref": float(row_max[row_id].item()),
                "row_mean_ratio_vs_cpu_ref": float(row_mean[row_id].item()),
                "worst_col": col,
                "candidate_value": _scalar(candidate[row_id, col]) if candidate is not None else None,
                "cpu_ref_value": _scalar(cpu_ref[row_id, col]) if cpu_ref is not None else None,
                "cpu_f64_ref_value": _scalar(cpu_f64[row_id, col]) if cpu_f64 is not None else None,
                "torch_f64_cast_ref_value": _scalar(torch_f64_cast[row_id, col]) if torch_f64_cast is not None else None,
                "torch_native_ref_value": _scalar(torch_native[row_id, col]) if torch_native is not None else None,
                "cusparse_ref_value": _scalar(cusparse[row_id, col]) if cusparse is not None else None,
                "abs_diff_vs_cpu_ref": (
                    float(torch.abs(candidate[row_id, col].to(torch.float64) - cpu_ref[row_id, col].to(torch.float64)).item())
                    if candidate is not None and cpu_ref is not None
                    else None
                ),
            }
        )
    return out


def _build_worst_values(matrix_name, dtype_name, op, alg, candidate, refs, dtype, row_lengths, buckets, top_values):
    if top_values <= 0 or candidate is None or refs.get("cpu") is None:
        return []
    candidate_cpu = _to_cpu_compare(candidate).to(torch.float64)
    cpu_ref = refs["cpu"].to(torch.float64)
    atol, rtol = _reference_tolerance(dtype)
    diff = torch.abs(candidate_cpu - cpu_ref)
    ratio = diff / (atol + rtol * torch.abs(cpu_ref))
    flat_count = min(int(top_values), int(ratio.numel()))
    if flat_count <= 0:
        return []
    values, flat_indices = torch.topk(ratio.reshape(-1), k=flat_count)
    dense_cols = int(ratio.shape[1])
    cpu_f64 = refs.get("cpu_f64")
    torch_f64_cast = refs.get("torch_f64_cast")
    torch_native = refs.get("torch_native")
    cusparse = refs.get("cusparse")
    row_lengths_cpu = row_lengths.to(torch.int64).cpu()
    out = []
    for rank, (value, flat_index) in enumerate(zip(values.tolist(), flat_indices.tolist()), start=1):
        row = int(flat_index // dense_cols)
        col = int(flat_index % dense_cols)
        out.append(
            {
                "matrix": matrix_name,
                "dtype": dtype_name,
                "op": op,
                "alg": alg,
                "rank": rank,
                "row": row,
                "col": col,
                "row_nnz": int(row_lengths_cpu[row].item()) if row < row_lengths_cpu.numel() else "",
                "bucket": buckets[row] if row < len(buckets) else "",
                "scaled_ratio_vs_cpu_ref": float(value),
                "abs_diff_vs_cpu_ref": float(diff[row, col].item()),
                "candidate_value": _scalar(candidate_cpu[row, col]),
                "cpu_ref_value": _scalar(cpu_ref[row, col]),
                "cpu_f64_ref_value": _scalar(cpu_f64[row, col]) if cpu_f64 is not None else None,
                "torch_f64_cast_ref_value": _scalar(torch_f64_cast[row, col]) if torch_f64_cast is not None else None,
                "torch_native_ref_value": _scalar(torch_native[row, col]) if torch_native is not None else None,
                "cusparse_ref_value": _scalar(cusparse[row, col]) if cusparse is not None else None,
            }
        )
    return out


def _sum_float32_sequence(values):
    acc = torch.tensor(0.0, dtype=torch.float32)
    for value in values.reshape(-1):
        acc = acc + value.to(torch.float32)
    return float(acc.item())


def _sum_float64_sequence(values):
    acc = torch.tensor(0.0, dtype=torch.float64)
    for value in values.reshape(-1):
        acc = acc + value.to(torch.float64)
    return float(acc.item())


def _sum_float32_pairwise(values):
    work = values.reshape(-1).to(torch.float32)
    if work.numel() == 0:
        return 0.0
    while work.numel() > 1:
        if work.numel() % 2:
            work = torch.cat([work, torch.zeros(1, dtype=torch.float32)])
        work = torch.sum(work.reshape(-1, 2), dim=1, dtype=torch.float32)
    return float(work.item())


def _format_top_products(source_indices, a_values, b_values, products, top_k):
    if top_k <= 0 or products.numel() == 0:
        return ""
    count = min(int(top_k), int(products.numel()))
    _, order = torch.topk(torch.abs(products), k=count)
    terms = []
    for idx in order.to(torch.int64).tolist():
        terms.append(
            f"{int(source_indices[idx].item())}:"
            f"{float(a_values[idx].item()):.9g}:"
            f"{float(b_values[idx].item()):.9g}:"
            f"{float(products[idx].item()):.9g}"
        )
    return ";".join(terms)


def _extract_contrib_terms(data_cpu, indices_cpu, indptr_cpu, B_cpu, shape, op, row, col, row_ids_cpu=None):
    row = int(row)
    col = int(col)
    n_rows, _n_cols = int(shape[0]), int(shape[1])
    if op == "non":
        start = int(indptr_cpu[row].item())
        end = int(indptr_cpu[row + 1].item())
        source = indices_cpu[start:end].to(torch.int64)
        a_values = data_cpu[start:end].to(torch.float64)
        b_values = B_cpu.index_select(0, source).select(1, col).to(torch.float64)
        return source, a_values, b_values

    if row_ids_cpu is None:
        row_counts = indptr_cpu[1:] - indptr_cpu[:-1]
        row_ids_cpu = torch.repeat_interleave(torch.arange(n_rows, dtype=torch.int64), row_counts.to(torch.int64))
    mask = indices_cpu.to(torch.int64) == row
    source = row_ids_cpu[mask].to(torch.int64)
    a_values = data_cpu[mask].to(torch.float64)
    b_values = B_cpu.index_select(0, source).select(1, col).to(torch.float64)
    return source, a_values, b_values


def _contrib_metrics_for_value(
    value_row,
    data_cpu,
    indices_cpu,
    indptr_cpu,
    B_cpu,
    shape,
    op,
    dtype,
    row_ids_cpu,
    top_products,
):
    row = int(value_row["row"])
    col = int(value_row["col"])
    source, a_values, b_values = _extract_contrib_terms(
        data_cpu,
        indices_cpu,
        indptr_cpu,
        B_cpu,
        shape,
        op,
        row,
        col,
        row_ids_cpu=row_ids_cpu,
    )
    products64 = a_values * b_values
    products32 = a_values.to(torch.float32) * b_values.to(torch.float32)
    abs_products = torch.abs(products64)
    nonzero_abs = abs_products[abs_products > 0]
    f64_sum = float(torch.sum(products64).item()) if products64.numel() else 0.0
    sum_abs = float(torch.sum(abs_products).item()) if abs_products.numel() else 0.0
    abs_sum = abs(f64_sum)
    tiny = torch.finfo(torch.float64).tiny
    positive = products64[products64 > 0]
    negative = products64[products64 < 0]
    asc_order = torch.argsort(torch.abs(products32), descending=False) if products32.numel() else torch.empty(0, dtype=torch.int64)
    desc_order = torch.argsort(torch.abs(products32), descending=True) if products32.numel() else torch.empty(0, dtype=torch.int64)
    max_abs_product = float(torch.max(abs_products).item()) if abs_products.numel() else 0.0
    min_nonzero = float(torch.min(nonzero_abs).item()) if nonzero_abs.numel() else 0.0
    seq_f32 = _sum_float32_sequence(products32)
    reverse_f32 = _sum_float32_sequence(torch.flip(products32, dims=(0,)))
    pairwise_f32 = _sum_float32_pairwise(products32)
    abs_asc_f32 = _sum_float32_sequence(products32.index_select(0, asc_order)) if products32.numel() else 0.0
    abs_desc_f32 = _sum_float32_sequence(products32.index_select(0, desc_order)) if products32.numel() else 0.0
    cpu_ref_value = float(value_row["cpu_ref_value"]) if value_row.get("cpu_ref_value") not in ("", None) else f64_sum
    candidate_value = float(value_row["candidate_value"]) if value_row.get("candidate_value") not in ("", None) else 0.0
    atol, rtol = _reference_tolerance(dtype)
    denom = atol + rtol * abs(cpu_ref_value)

    def scaled(value):
        return abs(float(value) - cpu_ref_value) / denom if denom != 0 else 0.0

    return {
        **value_row,
        "tolerance_atol": atol,
        "tolerance_rtol": rtol,
        "tolerance_denom": denom,
        "contribution_count": int(products64.numel()),
        "seq_f32_sum": seq_f32,
        "reverse_f32_sum": reverse_f32,
        "pairwise_f32_sum": pairwise_f32,
        "abs_asc_f32_sum": abs_asc_f32,
        "abs_desc_f32_sum": abs_desc_f32,
        "f64_sum": f64_sum,
        "f64_seq_sum": _sum_float64_sequence(products64),
        "f32_cast_of_f64_sum": float(torch.tensor(f64_sum, dtype=torch.float32).item()),
        "seq_f32_scaled_ratio_vs_cpu_ref": scaled(seq_f32),
        "reverse_f32_scaled_ratio_vs_cpu_ref": scaled(reverse_f32),
        "pairwise_f32_scaled_ratio_vs_cpu_ref": scaled(pairwise_f32),
        "abs_asc_f32_scaled_ratio_vs_cpu_ref": scaled(abs_asc_f32),
        "abs_desc_f32_scaled_ratio_vs_cpu_ref": scaled(abs_desc_f32),
        "candidate_minus_cpu_ref": candidate_value - cpu_ref_value,
        "candidate_minus_seq_f32": candidate_value - seq_f32,
        "candidate_minus_reverse_f32": candidate_value - reverse_f32,
        "candidate_minus_pairwise_f32": candidate_value - pairwise_f32,
        "candidate_minus_abs_asc_f32": candidate_value - abs_asc_f32,
        "candidate_minus_abs_desc_f32": candidate_value - abs_desc_f32,
        "sum_abs_products": sum_abs,
        "abs_sum": abs_sum,
        "cancellation_ratio": sum_abs / max(abs_sum, tiny),
        "positive_sum": float(torch.sum(positive).item()) if positive.numel() else 0.0,
        "negative_sum": float(torch.sum(negative).item()) if negative.numel() else 0.0,
        "positive_count": int(positive.numel()),
        "negative_count": int(negative.numel()),
        "zero_count": int(torch.sum(products64 == 0).item()) if products64.numel() else 0,
        "max_abs_product": max_abs_product,
        "min_nonzero_abs_product": min_nonzero,
        "product_abs_dynamic_range": (max_abs_product / min_nonzero) if min_nonzero > 0 else None,
        "mean_abs_product": float(torch.mean(abs_products).item()) if abs_products.numel() else 0.0,
        "max_abs_a": float(torch.max(torch.abs(a_values)).item()) if a_values.numel() else 0.0,
        "max_abs_b": float(torch.max(torch.abs(b_values)).item()) if b_values.numel() else 0.0,
        "top_abs_products": _format_top_products(source, a_values, b_values, products64, top_products),
    }


def _build_worst_contribs(value_rows, data, indices, indptr, B, shape, op, dtype, top_contribs, top_products):
    if top_contribs <= 0 or not value_rows:
        return []
    selected = value_rows[: int(top_contribs)]
    data_cpu = data.detach().cpu()
    indices_cpu = indices.detach().cpu().to(torch.int64)
    indptr_cpu = indptr.detach().cpu().to(torch.int64)
    B_cpu = B.detach().cpu()
    row_ids_cpu = None
    if op in ("trans", "conj"):
        n_rows = int(shape[0])
        row_counts = indptr_cpu[1:] - indptr_cpu[:-1]
        row_ids_cpu = torch.repeat_interleave(torch.arange(n_rows, dtype=torch.int64), row_counts.to(torch.int64))
    return [
        _contrib_metrics_for_value(
            row,
            data_cpu,
            indices_cpu,
            indptr_cpu,
            B_cpu,
            shape,
            op,
            dtype,
            row_ids_cpu,
            top_products,
        )
        for row in selected
    ]


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})


def _run_route(prepared, B, alg, warmup, iters, layout):
    out, gpu_ms = _cuda_event_benchmark(
        lambda: fs.flagsparse_spmm_csr_run(prepared, B, alg=alg, dense_layout=layout),
        warmup,
        iters,
    )
    _, meta = fs.flagsparse_spmm_csr_run(
        prepared,
        B,
        alg=alg,
        dense_layout=layout,
        return_meta=True,
        timing=True,
        diagnostics=True,
    )
    return out, gpu_ms, meta


def _skip_summary(matrix_name, dtype_name, op, alg, shape, nnz, dense_cols, b_stride, reason, refs, dtype, row_lengths):
    n_rows, n_cols = shape
    avg = float(nnz) / float(max(1, n_rows))
    return {
        "matrix": matrix_name,
        "dtype": dtype_name,
        "op": op,
        "layout": "row",
        "alg": alg,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": int(nnz),
        "avg_nnz_per_row": avg,
        "max_row_nnz": int(row_lengths.max().item()) if row_lengths.numel() else 0,
        "dense_cols": dense_cols,
        "b_stride": b_stride,
        "status": "SKIP",
        "reason": reason,
        "cpu_ref_kind": refs.get("cpu_kind") or "",
        "cpu_ref_status": "SKIP" if refs.get("cpu") is None else "PASS",
        "cpu_ref_reason": refs.get("cpu_reason") or "",
    }


def _run_one_case(args, path, dtype, op, alg_names):
    device = torch.device("cuda")
    dtype_name = _dtype_name(dtype)
    matrix_name = os.path.basename(path)
    data, indices, indptr, shape = load_mtx_to_csr_torch(path, dtype=dtype, device=device)
    indices = indices.to(torch.int32)
    n_rows, n_cols = shape
    row_lengths = (indptr[1:] - indptr[:-1]).to(torch.int64)
    max_row_nnz = int(row_lengths.max().item()) if row_lengths.numel() else 0
    b_rows = n_rows if op in ("trans", "conj") else n_cols
    torch.manual_seed(int(args.seed))
    B = _materialize_dense_layout_for_test(
        _build_dense_matrix(b_rows, args.dense_cols, dtype, device),
        "row",
    )
    b_stride = _stride_string(B)

    refs = {
        "cpu": None,
        "cpu_kind": f"cpu_{_dtype_name(dtype)}_accum",
        "cpu_reason": "",
        "cpu_f64": None,
        "cpu_f64_reason": "",
        "torch_f64_cast": None,
        "torch_native": None,
        "cusparse": None,
    }
    if args.cpu_ref:
        try:
            refs["cpu"], refs["cpu_reason"] = _cpu_csr_spmm_reference(
                data, indices, indptr, shape, B, dtype, op, dtype
            )
            refs["cpu_f64"], refs["cpu_f64_reason"] = _cpu_csr_spmm_reference(
                data, indices, indptr, shape, B, dtype, op, torch.float64
            )
        except Exception as exc:
            refs["cpu_reason"] = str(exc)
    try:
        torch_f64_cast, _, _ = _build_pytorch_reference(data, indices, indptr, shape, B, op=op)
        refs["torch_f64_cast"] = _to_cpu_compare(torch_f64_cast)
    except Exception as exc:
        refs["torch_f64_cast_reason"] = str(exc)
    try:
        refs["torch_native"] = _to_cpu_compare(_torch_native_reference(data, indices, indptr, shape, B, op))
    except Exception as exc:
        refs["torch_native_reason"] = str(exc)

    cusparse_ms = None
    cusparse_reason = ""
    if not args.no_cusparse:
        cusparse_out, cusparse_ms, cusparse_reason = _time_cusparse(
            data, indices, indptr, shape, B, op, args.warmup, args.iters, layout="row"
        )
        refs["cusparse"] = _to_cpu_compare(cusparse_out)
    else:
        cusparse_reason = "disabled"

    cpu_ref = refs["cpu"]
    cpu_f64_vs_cpu = _error_profile(refs["cpu_f64"], cpu_ref, dtype)["global_err"] if cpu_ref is not None else None
    torch_f64_cast_vs_cpu = _error_profile(refs["torch_f64_cast"], cpu_ref, dtype)["global_err"] if cpu_ref is not None else None
    torch_native_vs_cpu = _error_profile(refs["torch_native"], cpu_ref, dtype)["global_err"] if cpu_ref is not None else None
    cusparse_vs_cpu = _error_profile(refs["cusparse"], cpu_ref, dtype)["global_err"] if cpu_ref is not None else None

    prepared = fs.prepare_spmm_csr_route(data, indices, indptr, shape, op=op, alg="auto")
    algs = _expand_algs(alg_names, op, dtype)
    base_out = None
    case_summary = []
    case_worst_rows = []
    case_worst_values = []
    case_worst_contribs = []
    case_buckets = []
    case_launch = []

    for alg in algs:
        try:
            out, gpu_ms, meta = _run_route(prepared, B, alg, args.warmup, args.iters, "row")
        except (fs.SpmmCsrAlgorithmUnavailable, ValueError, TypeError, RuntimeError) as exc:
            case_summary.append(
                _skip_summary(
                    matrix_name,
                    dtype_name,
                    op,
                    alg,
                    shape,
                    data.numel(),
                    args.dense_cols,
                    b_stride,
                    str(exc),
                    refs,
                    dtype,
                    row_lengths,
                )
            )
            continue

        if meta.get("alg") == "csr_base":
            base_out = _to_cpu_compare(out)
        out_cpu = _to_cpu_compare(out)
        diagnostics = meta.get("diagnostics", {})
        buckets = _build_row_bucket_map(int(out_cpu.shape[0]), diagnostics)
        cpu_profile = _error_profile(out_cpu, cpu_ref, dtype)
        cpu_f64_profile = _error_profile(out_cpu, refs["cpu_f64"], dtype)
        torch_f64_cast_profile = _error_profile(out_cpu, refs["torch_f64_cast"], dtype)
        torch_native_profile = _error_profile(out_cpu, refs["torch_native"], dtype)
        cusparse_profile = _error_profile(out_cpu, refs["cusparse"], dtype)
        base_profile = _error_profile(out_cpu, base_out, dtype) if base_out is not None else {"global_err": None}
        worst_row = cpu_profile["worst_row"]
        worst_bucket = buckets[worst_row] if worst_row is not None and worst_row < len(buckets) else ""
        summary_row = {
            "matrix": matrix_name,
            "dtype": dtype_name,
            "op": op,
            "layout": "row",
            "alg": meta.get("alg", alg),
            "n_rows": n_rows,
            "n_cols": n_cols,
            "nnz": int(data.numel()),
            "avg_nnz_per_row": float(data.numel()) / float(max(1, n_rows)),
            "max_row_nnz": max_row_nnz,
            "dense_cols": args.dense_cols,
            "b_stride": b_stride,
            "c_stride": _stride_string(out),
            "ms": meta.get("operator_ms"),
            "gpu_ms": gpu_ms,
            "process_cpu_ms": meta.get("process_cpu_ms"),
            "process_gpu_ms": meta.get("process_gpu_ms"),
            "compute_ms": meta.get("compute_ms"),
            "cpu_ref_kind": refs.get("cpu_kind") or "",
            "cpu_ref_status": "SKIP" if cpu_ref is None else "PASS",
            "cpu_ref_reason": refs.get("cpu_reason") or "",
            "cpu_f64_ref_err_vs_cpu_ref": cpu_f64_vs_cpu,
            "cpu_f64_ref_reason": refs.get("cpu_f64_reason") or "",
            "torch_f64_cast_ref_err_vs_cpu_ref": torch_f64_cast_vs_cpu,
            "torch_f64_cast_ref_reason": refs.get("torch_f64_cast_reason") or "",
            "torch_native_ref_err_vs_cpu_ref": torch_native_vs_cpu,
            "torch_native_ref_reason": refs.get("torch_native_reason") or "",
            "cusparse_ref_err_vs_cpu_ref": cusparse_vs_cpu,
            "err_vs_cpu_ref": cpu_profile["global_err"],
            "err_vs_cpu_f64_ref": cpu_f64_profile["global_err"],
            "err_vs_torch_f64_cast_ref": torch_f64_cast_profile["global_err"],
            "err_vs_torch_native_ref": torch_native_profile["global_err"],
            "err_vs_cusparse_ref": cusparse_profile["global_err"],
            "err_vs_csr_base": base_profile["global_err"],
            "max_abs_vs_cpu_ref": cpu_profile["max_abs"],
            "mean_abs_vs_cpu_ref": cpu_profile["mean_abs"],
            "row_p50_vs_cpu_ref": cpu_profile["row_p50"],
            "row_p90_vs_cpu_ref": cpu_profile["row_p90"],
            "row_p99_vs_cpu_ref": cpu_profile["row_p99"],
            "row_max_vs_cpu_ref": cpu_profile["row_max"],
            "rows_gt_0_5_vs_cpu_ref": cpu_profile["rows_gt_0_5"],
            "rows_gt_1_vs_cpu_ref": cpu_profile["rows_gt_1"],
            "rows_gt_2_vs_cpu_ref": cpu_profile["rows_gt_2"],
            "rows_gt_10_vs_cpu_ref": cpu_profile["rows_gt_10"],
            "worst_row": worst_row,
            "worst_col": cpu_profile["worst_col"],
            "worst_row_nnz": (
                int(row_lengths.cpu()[worst_row].item())
                if worst_row is not None and worst_row < row_lengths.numel()
                else ""
            ),
            "worst_bucket": worst_bucket,
            "status": cpu_profile["status"],
            "reason": "",
            "cusparse_ms": cusparse_ms,
            "cusparse_reason": cusparse_reason or "",
        }
        case_summary.append(summary_row)
        refs_for_rows = {
            "cpu": cpu_ref,
            "cpu_f64": refs["cpu_f64"],
            "torch_f64_cast": refs["torch_f64_cast"],
            "torch_native": refs["torch_native"],
            "cusparse": refs["cusparse"],
        }
        case_worst_rows.extend(
            _build_worst_rows(
                matrix_name,
                dtype_name,
                op,
                meta.get("alg", alg),
                out_cpu,
                refs_for_rows,
                cpu_profile,
                row_lengths,
                buckets,
                args.top_rows,
            )
        )
        value_rows = _build_worst_values(
            matrix_name,
            dtype_name,
            op,
            meta.get("alg", alg),
            out_cpu,
            refs_for_rows,
            dtype,
            row_lengths,
            buckets,
            args.top_values,
        )
        case_worst_values.extend(value_rows)
        case_worst_contribs.extend(
            _build_worst_contribs(
                value_rows,
                data,
                indices,
                indptr,
                B,
                shape,
                op,
                dtype,
                args.top_contribs,
                args.contrib_top_products,
            )
        )
        case_buckets.extend(
            _bucket_rows_from_diagnostics(
                matrix_name,
                dtype_name,
                op,
                meta.get("alg", alg),
                diagnostics,
                row_lengths,
            )
        )
        case_launch.append(_diagnostic_launch_row(matrix_name, dtype_name, op, meta.get("alg", alg), diagnostics))
        print(
            f"{matrix_name:<28} {dtype_name:<8} {op:<5} {meta.get('alg', alg):<18} "
            f"err_cpu={_fmt(cpu_profile['global_err'], 4):>10} "
            f"rows>1={_fmt(cpu_profile['rows_gt_1'], 0):>7} "
            f"status={cpu_profile['status']}"
        )

    return case_summary, case_worst_rows, case_worst_values, case_worst_contribs, case_buckets, case_launch


def main():
    parser = argparse.ArgumentParser(description="Diagnose CSR SpMM f32/f64 route errors with CPU reference.")
    parser.add_argument("input", nargs="+", help=".mtx file(s) or directories")
    parser.add_argument("--out-dir", required=True, help="Output directory for diagnostic CSV files")
    parser.add_argument("--dtype", default=",".join(DEFAULT_DTYPES), help="float32,float64, or all")
    parser.add_argument("--op", default=",".join(DEFAULT_OPS), help="non, trans, conj, comma list, or all")
    parser.add_argument("--alg", default="all", help="all or comma-separated CSR route algorithm names")
    parser.add_argument("--dense-cols", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-ref", dest="cpu_ref", action="store_true", default=True, help="Enable CPU reference")
    parser.add_argument("--no-cpu-ref", dest="cpu_ref", action="store_false", help="Disable CPU reference")
    parser.add_argument("--top-rows", type=int, default=512, help="Rows to keep in worst_rows.csv per case")
    parser.add_argument("--top-values", type=int, default=128, help="Elements to keep in worst_values.csv per case")
    parser.add_argument(
        "--top-contribs",
        type=int,
        default=32,
        help="Worst elements per case to expand into contribution/order statistics",
    )
    parser.add_argument(
        "--contrib-top-products",
        type=int,
        default=8,
        help="Largest |A*B| contribution terms serialized per expanded element",
    )
    parser.add_argument("--no-cusparse", action="store_true", help="Disable CuPy/cuSPARSE reference")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    try:
        dtype_names = _parse_csv_names(args.dtype, tuple(DTYPE_MAP), "--dtype")
        op_names = _parse_csv_names(args.op, ALL_OPS, "--op")
        alg_names = _parse_algs(args.alg)
    except ValueError as exc:
        parser.error(str(exc))

    paths = _resolve_input_paths(args.input)
    if not paths:
        raise ValueError(f"No .mtx files found from input: {args.input}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(int(args.seed))
    summary_rows = []
    worst_rows = []
    worst_values = []
    worst_contribs = []
    bucket_rows = []
    launch_rows = []

    for dtype_name in dtype_names:
        dtype = DTYPE_MAP[dtype_name]
        for op in op_names:
            for path in paths:
                with torch.no_grad():
                    case = _run_one_case(args, path, dtype, op, alg_names)
                summary_rows.extend(case[0])
                worst_rows.extend(case[1])
                worst_values.extend(case[2])
                worst_contribs.extend(case[3])
                bucket_rows.extend(case[4])
                launch_rows.extend(case[5])

    _write_csv(os.path.join(args.out_dir, "summary.csv"), summary_rows, SUMMARY_FIELDS)
    _write_csv(os.path.join(args.out_dir, "worst_rows.csv"), worst_rows, WORST_ROW_FIELDS)
    _write_csv(os.path.join(args.out_dir, "worst_values.csv"), worst_values, WORST_VALUE_FIELDS)
    _write_csv(os.path.join(args.out_dir, "worst_contribs.csv"), worst_contribs, WORST_CONTRIB_FIELDS)
    _write_csv(os.path.join(args.out_dir, "bucket_stats.csv"), bucket_rows, BUCKET_FIELDS)
    _write_csv(os.path.join(args.out_dir, "launch_stats.csv"), launch_rows, LAUNCH_FIELDS)
    print(f"Wrote diagnostics to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
