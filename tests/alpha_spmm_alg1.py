"""
Diagnostics for the experimental AlphaSparse ALG1 Triton SpMM path.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as fs

from test_alpha_spmm_alg1 import _resolve_input_paths, run_one_case
from test_spmm_opt import load_mtx_to_csr_torch


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


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
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
    parser = argparse.ArgumentParser(description="Diagnose experimental AlphaSparse ALG1 Triton SpMM.")
    parser.add_argument("input_path", help=".mtx file or directory")
    parser.add_argument("--out-dir", required=True, help="Output directory for diagnostic CSV files")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--dense-cols", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-cusparse", action="store_true")
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    index_dtype = torch.int32
    device = torch.device("cuda")
    paths = _resolve_input_paths([args.input_path])
    if not paths:
        raise ValueError(f"No .mtx files found from input: {args.input_path}")

    summary_rows = []
    launch_rows = []
    for path in paths:
        data, indices, indptr, shape = load_mtx_to_csr_torch(path, dtype=dtype, device=device)
        indices = indices.to(index_dtype)
        result = run_one_case(
            os.path.basename(path),
            data,
            indices,
            indptr,
            shape,
            dtype,
            index_dtype,
            args.dense_cols,
            args.warmup,
            args.iters,
            args.seed,
            args.with_cusparse,
            return_details=True,
        )
        summary = result["summary"]
        summary_rows.append(summary)
        B = torch.randn(shape[1], args.dense_cols, dtype=dtype, device=device)
        launch_rows.append(
            _build_launch_row(
                summary["matrix"],
                args.dtype,
                args.dense_cols,
                result["prepared_alpha"],
                B,
            )
        )
        print(
            f"{summary['matrix']}: matrix_status={summary['matrix_status']}  "
            f"alpha_alg1_vs_torch_err={summary['alpha_alg1_vs_torch_err']}"
        )

    _write_csv(os.path.join(args.out_dir, "summary.csv"), summary_rows, SUMMARY_FIELDS)
    _write_csv(os.path.join(args.out_dir, "launch_stats.csv"), launch_rows, LAUNCH_FIELDS)
    print(f"Wrote diagnostics to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
