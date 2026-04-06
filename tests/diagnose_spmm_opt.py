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
from test_spmm_opt import _build_reference, _seeded_dense_matrix, load_mtx_to_csr_torch


SUMMARY_FIELDS = [
    "matrix",
    "value_dtype",
    "dense_cols",
    "seed",
    "n_rows",
    "n_cols",
    "nnz",
    "avg_nnz_per_row",
    "global_err_opt",
    "global_err_base",
    "rows_err_gt_1",
    "rows_err_gt_0_5",
    "worst_row",
    "worst_row_nnz",
    "worst_col",
    "worst_ratio",
    "status",
]

ROW_FIELDS = [
    "row",
    "row_nnz",
    "max_ratio",
    "mean_ratio",
    "max_abs_diff",
    "worst_col",
    "worst_ref",
    "worst_opt",
    "worst_base",
    "bucket_kind",
]

BUCKET_FIELDS = [
    "bucket_kind",
    "row_count",
    "row_nnz_min",
    "row_nnz_mean",
    "row_nnz_max",
    "bad_rows_gt_1",
    "bad_rows_gt_0_5",
    "worst_ratio",
    "p50_ratio",
    "p90_ratio",
    "p99_ratio",
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


def _build_bucket_rows(prepared, row_max_ratio):
    rows = []
    for bucket in prepared.row_buckets:
        bucket_rows = bucket["rows"]
        kind = bucket["kind"]
        if bucket_rows.numel() == 0:
            continue
        bucket_row_nnz = _bucket_row_nnz(prepared, bucket_rows)
        bucket_ratio = row_max_ratio.index_select(0, bucket_rows.to(row_max_ratio.device))
        p50_ratio, p90_ratio, p99_ratio = _quantiles(bucket_ratio)
        rows.append(
            {
                "bucket_kind": kind,
                "row_count": int(bucket_rows.numel()),
                "row_nnz_min": int(bucket_row_nnz.min().item()),
                "row_nnz_mean": float(bucket_row_nnz.to(torch.float64).mean().item()),
                "row_nnz_max": int(bucket_row_nnz.max().item()),
                "bad_rows_gt_1": int(torch.count_nonzero(bucket_ratio > 1.0).item()),
                "bad_rows_gt_0_5": int(torch.count_nonzero(bucket_ratio > 0.5).item()),
                "worst_ratio": float(bucket_ratio.max().item()),
                "p50_ratio": p50_ratio,
                "p90_ratio": p90_ratio,
                "p99_ratio": p99_ratio,
            }
        )
    return rows


def _build_row_rows(
    prepared,
    ref,
    opt,
    base,
    row_max_ratio,
    row_mean_ratio,
    row_max_abs,
    row_worst_col,
    bucket_names,
):
    rows = []
    for row_id in range(prepared.n_rows):
        worst_col = int(row_worst_col[row_id].item())
        rows.append(
            {
                "row": row_id,
                "row_nnz": int(prepared.row_lengths[row_id].item()),
                "max_ratio": float(row_max_ratio[row_id].item()),
                "mean_ratio": float(row_mean_ratio[row_id].item()),
                "max_abs_diff": float(row_max_abs[row_id].item()),
                "worst_col": worst_col,
                "worst_ref": float(ref[row_id, worst_col].item()),
                "worst_opt": float(opt[row_id, worst_col].item()),
                "worst_base": float(base[row_id, worst_col].item()),
                "bucket_kind": bucket_names[row_id],
            }
        )
    return rows


def _print_matrix_summary(summary_row, prepared, row_max_ratio, bucket_rows):
    print("=" * 120)
    print(f"Matrix: {summary_row['matrix']}")
    print(
        f"dtype={summary_row['value_dtype']}  dense_cols={summary_row['dense_cols']}  seed={summary_row['seed']}"
    )
    print(
        f"shape=({summary_row['n_rows']}, {summary_row['n_cols']})  "
        f"nnz={summary_row['nnz']}  avg_nnz_per_row={summary_row['avg_nnz_per_row']:.2f}"
    )
    print(
        f"global_err_base={summary_row['global_err_base']:.6f}  "
        f"global_err_opt={summary_row['global_err_opt']:.6f}  "
        f"status={summary_row['status']}"
    )
    print(f"row_nnz_quantiles: {_describe_quantiles(prepared.row_lengths)}")
    print(f"row_err_quantiles: {_describe_quantiles(row_max_ratio)}")
    print(
        f"rows_with_err_gt_1: {summary_row['rows_err_gt_1']} / {summary_row['n_rows']}  "
        f"rows_with_err_gt_0_5: {summary_row['rows_err_gt_0_5']} / {summary_row['n_rows']}"
    )
    print("-" * 120)
    print("Bucket summary")
    for bucket_row in bucket_rows:
        print(
            f"kind={bucket_row['bucket_kind']:<7} rows={bucket_row['row_count']:>8} "
            f"row_nnz[min/mean/max]={bucket_row['row_nnz_min']:>4}/"
            f"{bucket_row['row_nnz_mean']:>8.2f}/"
            f"{bucket_row['row_nnz_max']:>4} "
            f"bad_rows_gt_1={bucket_row['bad_rows_gt_1']:>6} "
            f"bad_rows_gt_0_5={bucket_row['bad_rows_gt_0_5']:>6} "
            f"worst_ratio={bucket_row['worst_ratio']:>10.4f}"
        )


def _print_top_rows(row_rows, topk):
    top_rows = sorted(row_rows, key=lambda row: row["max_ratio"], reverse=True)[:topk]
    print("-" * 120)
    print(f"Top-{len(top_rows)} bad rows")
    for rank, row in enumerate(top_rows, start=1):
        print(
            f"{rank:>2}. row={row['row']:>8} row_nnz={row['row_nnz']:>6} "
            f"bucket={row['bucket_kind']:<7} max_ratio={row['max_ratio']:>10.4f} "
            f"mean_ratio={row['mean_ratio']:>10.4f} max_abs_diff={row['max_abs_diff']:>10.4e} "
            f"worst_col={row['worst_col']:>4} ref={row['worst_ref']:>12.4e} "
            f"opt={row['worst_opt']:>12.4e} base={row['worst_base']:>12.4e}"
        )


def diagnose_one(path, dense_cols, dtype, seed, topk):
    device = torch.device("cuda")
    data, indices, indptr, shape = load_mtx_to_csr_torch(path, dtype=dtype, device=device)
    n_rows, n_cols = shape
    B = _seeded_dense_matrix((n_cols, dense_cols), dtype, device, seed)

    ref = _build_reference(data, indices, indptr, B, shape, dtype)
    base = fs.flagsparse_spmm_csr(data, indices, indptr, B, shape)
    prepared = fs.prepare_spmm_csr_opt(data, indices, indptr, shape)
    opt = fs.flagsparse_spmm_csr_opt(B=B, prepared=prepared)

    if dtype == torch.float32:
        atol, rtol = 1e-4, 1e-2
    else:
        atol, rtol = 1e-12, 1e-10

    diff_opt = torch.abs(opt - ref).to(torch.float64)
    diff_base = torch.abs(base - ref).to(torch.float64)
    denom = (atol + rtol * torch.abs(ref)).to(torch.float64)
    ratio_opt = diff_opt / denom
    ratio_base = diff_base / denom
    row_max_ratio = torch.max(ratio_opt, dim=1).values
    row_mean_ratio = torch.mean(ratio_opt, dim=1)
    row_max_abs = torch.max(diff_opt, dim=1).values
    row_worst_col = torch.argmax(ratio_opt, dim=1)
    worst_row = int(torch.argmax(row_max_ratio).item()) if n_rows > 0 else 0
    worst_col = int(row_worst_col[worst_row].item()) if n_rows > 0 else 0
    bucket_names = _row_bucket_names(prepared)

    summary_row = {
        "matrix": os.path.basename(path),
        "value_dtype": str(dtype).replace("torch.", ""),
        "dense_cols": int(dense_cols),
        "seed": seed,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "nnz": int(data.numel()),
        "avg_nnz_per_row": (float(data.numel()) / max(1, int(n_rows))),
        "global_err_opt": (float(torch.max(ratio_opt).item()) if ratio_opt.numel() > 0 else 0.0),
        "global_err_base": (float(torch.max(ratio_base).item()) if ratio_base.numel() > 0 else 0.0),
        "rows_err_gt_1": int(torch.count_nonzero(row_max_ratio > 1.0).item()),
        "rows_err_gt_0_5": int(torch.count_nonzero(row_max_ratio > 0.5).item()),
        "worst_row": worst_row,
        "worst_row_nnz": (int(prepared.row_lengths[worst_row].item()) if n_rows > 0 else 0),
        "worst_col": worst_col,
        "worst_ratio": (float(row_max_ratio[worst_row].item()) if n_rows > 0 else 0.0),
        "status": ("PASS" if (float(torch.max(ratio_opt).item()) if ratio_opt.numel() > 0 else 0.0) <= 1.0 else "FAIL"),
    }

    row_rows = _build_row_rows(
        prepared,
        ref,
        opt,
        base,
        row_max_ratio,
        row_mean_ratio,
        row_max_abs,
        row_worst_col,
        bucket_names,
    )
    bucket_rows = _build_bucket_rows(prepared, row_max_ratio)

    _print_matrix_summary(summary_row, prepared, row_max_ratio, bucket_rows)
    _print_top_rows(row_rows, max(1, min(int(topk), len(row_rows))))

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


def run_batch(input_path, dtype, dense_cols, seed, topk, csv_path=None, out_dir="spmm_diag", only_status="all", only_matrices=None):
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
        csv_path=args.csv,
        out_dir=args.out_dir,
        only_status=args.only_status,
        only_matrices=args.only_matrices,
    )


if __name__ == "__main__":
    main()
