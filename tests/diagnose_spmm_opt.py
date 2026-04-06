"""
Batch-capable diagnostics for CSR SpMM-opt accuracy.

Usage:
    python tests/diagnose_spmm_opt.py path/to/matrix.mtx --dense-cols 32 --seed 0
    python tests/diagnose_spmm_opt.py path/to/mtx_dir --dense-cols 32 --seed 0 --out-dir diag_out
    python tests/diagnose_spmm_opt.py path/to/mtx_dir --csv spmm_opt.csv --only-status fail --out-dir diag_out
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
import flagsparse.sparse_operations.spmm_csr as fs_spmm_csr
from test_spmm_opt import _build_reference, _seeded_dense_matrix, load_mtx_to_csr_torch


SUMMARY_FIELDS = [
    "matrix",
    "value_dtype",
    "dense_cols",
    "seed",
    "compare_stable",
    "n_rows",
    "n_cols",
    "nnz",
    "avg_nnz_per_row",
    "global_err_base",
    "legacy_err_opt",
    "stable_err_opt",
    "legacy_rows_err_gt_1",
    "stable_rows_err_gt_1",
    "legacy_rows_err_gt_0_5",
    "stable_rows_err_gt_0_5",
    "legacy_worst_row",
    "stable_worst_row",
    "legacy_worst_row_nnz",
    "stable_worst_row_nnz",
    "legacy_worst_col",
    "stable_worst_col",
    "legacy_worst_ratio",
    "stable_worst_ratio",
    "legacy_status",
    "stable_status",
    "status",
]

ROW_FIELDS = [
    "row",
    "row_nnz",
    "bucket_kind",
    "legacy_max_ratio",
    "stable_max_ratio",
    "legacy_mean_ratio",
    "stable_mean_ratio",
    "legacy_max_abs_diff",
    "stable_max_abs_diff",
    "legacy_worst_col",
    "stable_worst_col",
    "legacy_worst_ref",
    "stable_worst_ref",
    "legacy_worst_opt",
    "stable_worst_opt",
    "legacy_worst_base",
    "stable_worst_base",
]

BUCKET_FIELDS = [
    "bucket_kind",
    "row_count",
    "row_nnz_min",
    "row_nnz_mean",
    "row_nnz_max",
    "legacy_bad_rows_gt_1",
    "stable_bad_rows_gt_1",
    "legacy_bad_rows_gt_0_5",
    "stable_bad_rows_gt_0_5",
    "legacy_worst_ratio",
    "stable_worst_ratio",
    "legacy_p50_ratio",
    "stable_p50_ratio",
    "legacy_p90_ratio",
    "stable_p90_ratio",
    "legacy_p99_ratio",
    "stable_p99_ratio",
]


def _dtype_from_name(name):
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    return mapping[name]


def _ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_csv(path, rows, fieldnames):
    _ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _normalize_matrix_name(name):
    base = os.path.basename(str(name).strip())
    if not base:
        return ""
    return base if base.endswith(".mtx") else f"{base}.mtx"


def _safe_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _resolve_input_paths(input_path):
    if os.path.isfile(input_path):
        if not str(input_path).lower().endswith(".mtx"):
            raise ValueError(f"Input file must be a .mtx file, got: {input_path}")
        return [os.path.abspath(input_path)]
    if os.path.isdir(input_path):
        paths = sorted(glob.glob(os.path.join(input_path, "*.mtx")))
        if not paths:
            raise ValueError(f"No .mtx files found under directory: {input_path}")
        return [os.path.abspath(path) for path in paths]
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _load_status_metadata(csv_path):
    if csv_path is None:
        return {}
    metadata = {}
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            matrix = _normalize_matrix_name(row.get("matrix", ""))
            if not matrix:
                continue
            metadata[matrix] = row
    return metadata


def _quantiles(values):
    if values.numel() == 0:
        return 0.0, 0.0, 0.0
    compare = values.to(torch.float64)
    q = torch.quantile(
        compare,
        torch.tensor([0.5, 0.9, 0.99], device=compare.device, dtype=compare.dtype),
    )
    return float(q[0].item()), float(q[1].item()), float(q[2].item())


def _describe_quantiles(values):
    if values.numel() == 0:
        return "empty"
    p50, p90, p99 = _quantiles(values)
    compare = values.to(torch.float64)
    return (
        f"min={float(compare.min().item()):.2f}, "
        f"p50={p50:.2f}, "
        f"p90={p90:.2f}, "
        f"p99={p99:.2f}, "
        f"max={float(compare.max().item()):.2f}"
    )


def _bucket_row_nnz(prepared, rows):
    if rows.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=prepared.row_lengths.device)
    return prepared.row_lengths.index_select(0, rows.to(prepared.row_lengths.device)).to(torch.int64)


def _row_bucket_names(prepared):
    bucket_names = ["unassigned"] * prepared.n_rows
    for bucket in prepared.row_buckets:
        kind = bucket["kind"]
        rows = bucket["rows"].to(torch.int64).cpu().tolist()
        for row in rows:
            bucket_names[row] = kind
    return bucket_names


def _compute_candidate_metrics(candidate, reference, dtype):
    if dtype == torch.float32:
        atol, rtol = 1e-4, 1e-2
    else:
        atol, rtol = 1e-12, 1e-10
    diff = torch.abs(candidate - reference).to(torch.float64)
    denom = (atol + rtol * torch.abs(reference)).to(torch.float64)
    ratio = diff / denom
    row_max_ratio = torch.max(ratio, dim=1).values
    row_mean_ratio = torch.mean(ratio, dim=1)
    row_max_abs = torch.max(diff, dim=1).values
    row_worst_col = torch.argmax(ratio, dim=1)
    worst_row = int(torch.argmax(row_max_ratio).item()) if row_max_ratio.numel() > 0 else 0
    worst_col = int(row_worst_col[worst_row].item()) if row_worst_col.numel() > 0 else 0
    global_err = float(torch.max(ratio).item()) if ratio.numel() > 0 else 0.0
    return {
        "global_err": global_err,
        "ratio": ratio,
        "row_max_ratio": row_max_ratio,
        "row_mean_ratio": row_mean_ratio,
        "row_max_abs": row_max_abs,
        "row_worst_col": row_worst_col,
        "rows_err_gt_1": int(torch.count_nonzero(row_max_ratio > 1.0).item()),
        "rows_err_gt_0_5": int(torch.count_nonzero(row_max_ratio > 0.5).item()),
        "worst_row": worst_row,
        "worst_col": worst_col,
        "worst_ratio": (float(row_max_ratio[worst_row].item()) if row_max_ratio.numel() > 0 else 0.0),
        "status": "PASS" if global_err <= 1.0 else "FAIL",
    }


def _build_bucket_rows(prepared, legacy_metrics, stable_metrics):
    rows = []
    for bucket in prepared.row_buckets:
        bucket_rows = bucket["rows"]
        kind = bucket["kind"]
        if bucket_rows.numel() == 0:
            continue
        bucket_row_nnz = _bucket_row_nnz(prepared, bucket_rows)
        legacy_ratio = legacy_metrics["row_max_ratio"].index_select(0, bucket_rows.to(legacy_metrics["row_max_ratio"].device))
        stable_ratio = None
        if stable_metrics is not None:
            stable_ratio = stable_metrics["row_max_ratio"].index_select(0, bucket_rows.to(stable_metrics["row_max_ratio"].device))
        legacy_p50, legacy_p90, legacy_p99 = _quantiles(legacy_ratio)
        stable_p50, stable_p90, stable_p99 = (None, None, None)
        if stable_ratio is not None:
            stable_p50, stable_p90, stable_p99 = _quantiles(stable_ratio)
        rows.append(
            {
                "bucket_kind": kind,
                "row_count": int(bucket_rows.numel()),
                "row_nnz_min": int(bucket_row_nnz.min().item()),
                "row_nnz_mean": float(bucket_row_nnz.to(torch.float64).mean().item()),
                "row_nnz_max": int(bucket_row_nnz.max().item()),
                "legacy_bad_rows_gt_1": int(torch.count_nonzero(legacy_ratio > 1.0).item()),
                "stable_bad_rows_gt_1": (int(torch.count_nonzero(stable_ratio > 1.0).item()) if stable_ratio is not None else None),
                "legacy_bad_rows_gt_0_5": int(torch.count_nonzero(legacy_ratio > 0.5).item()),
                "stable_bad_rows_gt_0_5": (int(torch.count_nonzero(stable_ratio > 0.5).item()) if stable_ratio is not None else None),
                "legacy_worst_ratio": float(legacy_ratio.max().item()),
                "stable_worst_ratio": (float(stable_ratio.max().item()) if stable_ratio is not None else None),
                "legacy_p50_ratio": legacy_p50,
                "stable_p50_ratio": stable_p50,
                "legacy_p90_ratio": legacy_p90,
                "stable_p90_ratio": stable_p90,
                "legacy_p99_ratio": legacy_p99,
                "stable_p99_ratio": stable_p99,
            }
        )
    return rows


def _build_row_rows(prepared, ref, base, legacy_candidate, stable_candidate, legacy_metrics, stable_metrics, bucket_names):
    rows = []
    for row_id in range(prepared.n_rows):
        legacy_col = int(legacy_metrics["row_worst_col"][row_id].item())
        stable_col = None
        if stable_metrics is not None:
            stable_col = int(stable_metrics["row_worst_col"][row_id].item())
        rows.append(
            {
                "row": row_id,
                "row_nnz": int(prepared.row_lengths[row_id].item()),
                "bucket_kind": bucket_names[row_id],
                "legacy_max_ratio": float(legacy_metrics["row_max_ratio"][row_id].item()),
                "stable_max_ratio": (float(stable_metrics["row_max_ratio"][row_id].item()) if stable_metrics is not None else None),
                "legacy_mean_ratio": float(legacy_metrics["row_mean_ratio"][row_id].item()),
                "stable_mean_ratio": (float(stable_metrics["row_mean_ratio"][row_id].item()) if stable_metrics is not None else None),
                "legacy_max_abs_diff": float(legacy_metrics["row_max_abs"][row_id].item()),
                "stable_max_abs_diff": (float(stable_metrics["row_max_abs"][row_id].item()) if stable_metrics is not None else None),
                "legacy_worst_col": legacy_col,
                "stable_worst_col": stable_col,
                "legacy_worst_ref": float(ref[row_id, legacy_col].item()),
                "stable_worst_ref": (float(ref[row_id, stable_col].item()) if stable_col is not None else None),
                "legacy_worst_opt": float(legacy_candidate[row_id, legacy_col].item()),
                "stable_worst_opt": (float(stable_candidate[row_id, stable_col].item()) if stable_col is not None else None),
                "legacy_worst_base": float(base[row_id, legacy_col].item()),
                "stable_worst_base": (float(base[row_id, stable_col].item()) if stable_col is not None else None),
            }
        )
    return rows


def _print_matrix_summary(summary_row, prepared, legacy_metrics, stable_metrics, bucket_rows):
    print("=" * 120)
    print(f"Matrix: {summary_row['matrix']}")
    print(
        f"dtype={summary_row['value_dtype']}  dense_cols={summary_row['dense_cols']}  "
        f"seed={summary_row['seed']}  compare_stable={summary_row['compare_stable']}"
    )
    print(
        f"shape=({summary_row['n_rows']}, {summary_row['n_cols']})  "
        f"nnz={summary_row['nnz']}  avg_nnz_per_row={summary_row['avg_nnz_per_row']:.2f}"
    )
    print(
        f"global_err_base={summary_row['global_err_base']:.6f}  "
        f"legacy_err_opt={summary_row['legacy_err_opt']:.6f}  "
        f"stable_err_opt={summary_row['stable_err_opt'] if summary_row['stable_err_opt'] is not None else 'N/A'}  "
        f"status={summary_row['status']}"
    )
    print(f"row_nnz_quantiles: {_describe_quantiles(prepared.row_lengths)}")
    print(f"legacy_row_err_quantiles: {_describe_quantiles(legacy_metrics['row_max_ratio'])}")
    if stable_metrics is not None:
        print(f"stable_row_err_quantiles: {_describe_quantiles(stable_metrics['row_max_ratio'])}")
    print(
        f"legacy rows err>1: {summary_row['legacy_rows_err_gt_1']} / {summary_row['n_rows']}  "
        f"legacy rows err>0.5: {summary_row['legacy_rows_err_gt_0_5']} / {summary_row['n_rows']}"
    )
    if stable_metrics is not None:
        print(
            f"stable rows err>1: {summary_row['stable_rows_err_gt_1']} / {summary_row['n_rows']}  "
            f"stable rows err>0.5: {summary_row['stable_rows_err_gt_0_5']} / {summary_row['n_rows']}"
        )
    print("-" * 120)
    print("Bucket summary")
    for bucket_row in bucket_rows:
        stable_bad_gt_1 = bucket_row["stable_bad_rows_gt_1"]
        stable_worst_ratio = bucket_row["stable_worst_ratio"]
        stable_worst_str = (
            f"{float(stable_worst_ratio):>10.4f}"
            if stable_worst_ratio is not None
            else f"{'N/A':>10}"
        )
        print(
            f"kind={bucket_row['bucket_kind']:<7} rows={bucket_row['row_count']:>8} "
            f"row_nnz[min/mean/max]={bucket_row['row_nnz_min']:>4}/"
            f"{bucket_row['row_nnz_mean']:>8.2f}/"
            f"{bucket_row['row_nnz_max']:>4} "
            f"legacy_bad_gt_1={bucket_row['legacy_bad_rows_gt_1']:>6} "
            f"stable_bad_gt_1={(stable_bad_gt_1 if stable_bad_gt_1 is not None else 'N/A'):>6} "
            f"legacy_worst={bucket_row['legacy_worst_ratio']:>10.4f} "
            f"stable_worst={stable_worst_str}"
        )


def _print_top_rows(row_rows, topk, compare_stable):
    key_name = "stable_max_ratio" if compare_stable else "legacy_max_ratio"
    top_rows = sorted(
        row_rows,
        key=lambda row: (-1.0 if row[key_name] is None else float(row[key_name])),
        reverse=True,
    )[:topk]
    print("-" * 120)
    print(f"Top-{len(top_rows)} rows by {'stable' if compare_stable else 'legacy'} ratio")
    for rank, row in enumerate(top_rows, start=1):
        stable_ratio_str = (
            f"{float(row['stable_max_ratio']):>10.4f}"
            if row["stable_max_ratio"] is not None
            else f"{'N/A':>10}"
        )
        stable_col_str = (
            f"{int(row['stable_worst_col']):>4}"
            if row["stable_worst_col"] is not None
            else f"{'N/A':>4}"
        )
        print(
            f"{rank:>2}. row={row['row']:>8} row_nnz={row['row_nnz']:>6} "
            f"bucket={row['bucket_kind']:<7} "
            f"legacy_ratio={row['legacy_max_ratio']:>10.4f} "
            f"stable_ratio={stable_ratio_str} "
            f"legacy_col={row['legacy_worst_col']:>4} "
            f"stable_col={stable_col_str}"
        )


def diagnose_one(path, dense_cols, dtype, seed, topk, compare_stable=True):
    device = torch.device("cuda")
    data, indices, indptr, shape = load_mtx_to_csr_torch(path, dtype=dtype, device=device)
    n_rows, n_cols = shape
    B = _seeded_dense_matrix((n_cols, dense_cols), dtype, device, seed)

    ref = _build_reference(data, indices, indptr, B, shape, dtype)
    base = fs.flagsparse_spmm_csr(data, indices, indptr, B, shape)
    prepared = fs.prepare_spmm_csr_opt(data, indices, indptr, shape)
    legacy_opt = fs.flagsparse_spmm_csr_opt(B=B, prepared=prepared)
    stable_opt = None
    if compare_stable:
        stable_opt = fs_spmm_csr._flagsparse_spmm_csr_opt_stable_for_diagnose(prepared, B)

    base_global_err = fs_spmm_csr._spmm_opt_reference_error(base, ref, dtype)
    legacy_metrics = _compute_candidate_metrics(legacy_opt, ref, dtype)
    stable_metrics = (
        _compute_candidate_metrics(stable_opt, ref, dtype)
        if stable_opt is not None
        else None
    )
    bucket_names = _row_bucket_names(prepared)

    summary_row = {
        "matrix": os.path.basename(path),
        "value_dtype": str(dtype).replace("torch.", ""),
        "dense_cols": int(dense_cols),
        "seed": seed,
        "compare_stable": bool(compare_stable),
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "nnz": int(data.numel()),
        "avg_nnz_per_row": (float(data.numel()) / max(1, int(n_rows))),
        "global_err_base": base_global_err,
        "legacy_err_opt": legacy_metrics["global_err"],
        "stable_err_opt": (stable_metrics["global_err"] if stable_metrics is not None else None),
        "legacy_rows_err_gt_1": legacy_metrics["rows_err_gt_1"],
        "stable_rows_err_gt_1": (stable_metrics["rows_err_gt_1"] if stable_metrics is not None else None),
        "legacy_rows_err_gt_0_5": legacy_metrics["rows_err_gt_0_5"],
        "stable_rows_err_gt_0_5": (stable_metrics["rows_err_gt_0_5"] if stable_metrics is not None else None),
        "legacy_worst_row": legacy_metrics["worst_row"],
        "stable_worst_row": (stable_metrics["worst_row"] if stable_metrics is not None else None),
        "legacy_worst_row_nnz": (int(prepared.row_lengths[legacy_metrics["worst_row"]].item()) if n_rows > 0 else 0),
        "stable_worst_row_nnz": (int(prepared.row_lengths[stable_metrics["worst_row"]].item()) if stable_metrics is not None and n_rows > 0 else None),
        "legacy_worst_col": legacy_metrics["worst_col"],
        "stable_worst_col": (stable_metrics["worst_col"] if stable_metrics is not None else None),
        "legacy_worst_ratio": legacy_metrics["worst_ratio"],
        "stable_worst_ratio": (stable_metrics["worst_ratio"] if stable_metrics is not None else None),
        "legacy_status": legacy_metrics["status"],
        "stable_status": (stable_metrics["status"] if stable_metrics is not None else None),
        "status": (stable_metrics["status"] if stable_metrics is not None else legacy_metrics["status"]),
    }

    row_rows = _build_row_rows(
        prepared,
        ref,
        base,
        legacy_opt,
        stable_opt,
        legacy_metrics,
        stable_metrics,
        bucket_names,
    )
    bucket_rows = _build_bucket_rows(prepared, legacy_metrics, stable_metrics)

    _print_matrix_summary(summary_row, prepared, legacy_metrics, stable_metrics, bucket_rows)
    _print_top_rows(row_rows, max(1, min(int(topk), len(row_rows))), compare_stable)

    return summary_row, row_rows, bucket_rows


def _collect_selected_paths(all_paths, csv_metadata, only_status, only_matrices):
    allowed_names = None
    if only_matrices:
        allowed_names = {
            _normalize_matrix_name(name)
            for name in only_matrices.split(",")
            if _normalize_matrix_name(name)
        }

    selected = []
    for path in all_paths:
        matrix_name = _normalize_matrix_name(os.path.basename(path))
        if allowed_names is not None and matrix_name not in allowed_names:
            continue
        if only_status != "all" and matrix_name in csv_metadata:
            csv_status = str(csv_metadata[matrix_name].get("status", "")).strip().lower()
            if csv_status != only_status:
                continue
        selected.append(path)
    return selected


def _filter_results_by_status(results, only_status):
    if only_status == "all":
        return results
    target = only_status.upper()
    return [result for result in results if result["summary"]["status"] == target]


def _default_output_paths(out_dir):
    abs_out_dir = os.path.abspath(out_dir)
    rows_dir = os.path.join(abs_out_dir, "diag_rows")
    buckets_dir = os.path.join(abs_out_dir, "diag_buckets")
    summary_csv = os.path.join(abs_out_dir, "spmm_diag_summary.csv")
    os.makedirs(rows_dir, exist_ok=True)
    os.makedirs(buckets_dir, exist_ok=True)
    return summary_csv, rows_dir, buckets_dir


def run_batch(input_path, dtype, dense_cols, seed, topk, compare_stable=True, csv_path=None, out_dir="spmm_diag", only_status="all", only_matrices=None):
    all_paths = _resolve_input_paths(input_path)
    csv_metadata = _load_status_metadata(csv_path)
    selected_paths = _collect_selected_paths(all_paths, csv_metadata, only_status, only_matrices)
    if not selected_paths:
        raise ValueError("No matrices matched the requested filters.")

    summary_csv, rows_dir, buckets_dir = _default_output_paths(out_dir)
    results = []
    for index, path in enumerate(selected_paths, start=1):
        print(f"[{index}/{len(selected_paths)}] Diagnosing {os.path.basename(path)}")
        summary_row, row_rows, bucket_rows = diagnose_one(
            path=path,
            dense_cols=dense_cols,
            dtype=dtype,
            seed=seed,
            topk=topk,
            compare_stable=compare_stable,
        )
        results.append(
            {
                "path": path,
                "summary": summary_row,
                "rows": row_rows,
                "buckets": bucket_rows,
            }
        )

    filtered_results = _filter_results_by_status(results, only_status)
    if not filtered_results:
        raise ValueError("Diagnostics completed, but no matrices matched the requested status filter.")

    _write_csv(summary_csv, [result["summary"] for result in filtered_results], SUMMARY_FIELDS)
    for result in filtered_results:
        stem = _safe_stem(result["path"])
        _write_csv(os.path.join(rows_dir, f"{stem}.rows.csv"), result["rows"], ROW_FIELDS)
        _write_csv(os.path.join(buckets_dir, f"{stem}.buckets.csv"), result["buckets"], BUCKET_FIELDS)

    print("=" * 120)
    print(f"Wrote summary CSV to {summary_csv}")
    print(f"Wrote row diagnostics under {rows_dir}")
    print(f"Wrote bucket diagnostics under {buckets_dir}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose SpMM-opt accuracy for one matrix or a directory of matrices.")
    parser.add_argument("input_path", help="Path to one .mtx file or a directory containing .mtx files")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--dense-cols", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk", type=int, default=10, help="How many top bad rows to print")
    parser.add_argument("--csv", type=str, default=None, help="Optional existing spmm_opt.csv used for metadata or prefiltering")
    parser.add_argument("--out-dir", type=str, default="spmm_diag", help="Output directory for batch summary and detail CSVs")
    parser.add_argument("--only-status", choices=["all", "fail", "pass"], default="all", help="If CSV metadata is provided, prefilter by CSV status; otherwise filter final outputs by diagnosed status")
    parser.add_argument("--only-matrices", type=str, default=None, help="Comma-separated matrix basenames to include, with or without .mtx suffix")
    parser.add_argument("--row-csv", type=str, default=None, help="Single-file mode only: explicit output path for row diagnostics CSV")
    parser.add_argument("--compare-stable", dest="compare_stable", action="store_true", default=True, help="Compare against the diagnose-only stable kernel")
    parser.add_argument("--no-compare-stable", dest="compare_stable", action="store_false", help="Disable stable-kernel comparison and only diagnose legacy opt")
    args = parser.parse_args()

    dtype = _dtype_from_name(args.dtype)
    input_is_file = os.path.isfile(args.input_path)

    if input_is_file:
        summary_row, row_rows, bucket_rows = diagnose_one(
            path=args.input_path,
            dense_cols=args.dense_cols,
            dtype=dtype,
            seed=args.seed,
            topk=args.topk,
            compare_stable=args.compare_stable,
        )
        if args.row_csv:
            _write_csv(args.row_csv, row_rows, ROW_FIELDS)
            print("-" * 120)
            print(f"Wrote row diagnostics to {args.row_csv}")
        else:
            summary_csv, rows_dir, buckets_dir = _default_output_paths(args.out_dir)
            stem = _safe_stem(args.input_path)
            _write_csv(summary_csv, [summary_row], SUMMARY_FIELDS)
            _write_csv(os.path.join(rows_dir, f"{stem}.rows.csv"), row_rows, ROW_FIELDS)
            _write_csv(os.path.join(buckets_dir, f"{stem}.buckets.csv"), bucket_rows, BUCKET_FIELDS)
            print("-" * 120)
            print(f"Wrote summary CSV to {summary_csv}")
            print(f"Wrote row diagnostics to {os.path.join(rows_dir, f'{stem}.rows.csv')}")
            print(f"Wrote bucket diagnostics to {os.path.join(buckets_dir, f'{stem}.buckets.csv')}")
        return

    if args.row_csv:
        raise ValueError("--row-csv is only supported in single-file mode; use --out-dir in directory mode")

    run_batch(
        input_path=args.input_path,
        dtype=dtype,
        dense_cols=args.dense_cols,
        seed=args.seed,
        topk=args.topk,
        compare_stable=args.compare_stable,
        csv_path=args.csv,
        out_dir=args.out_dir,
        only_status=args.only_status,
        only_matrices=args.only_matrices,
    )


if __name__ == "__main__":
    main()
