"""Shared benchmark-script helpers for runtime and vendor-baseline reporting."""

from __future__ import annotations

import torch

from flagsparse.sparse_operations import _common as fs_common


def runtime_backend_name() -> str:
    if fs_common._is_rocm_runtime():
        return "ROCm/DCU"
    if getattr(torch.version, "cuda", None):
        return "CUDA"
    return "unknown"


def sparse_backend_label(backend) -> str:
    return {
        "hipsparse": "hipSPARSE",
        "cupy_cusparse": "CuPy/cuSPARSE",
        "torch": "PyTorch",
        None: "N/A",
    }.get(backend, str(backend))


def expected_vendor_label() -> str:
    return "hipSPARSE" if fs_common._is_rocm_runtime() else "CuPy/cuSPARSE"


def expected_vendor_short() -> str:
    return "HS" if fs_common._is_rocm_runtime() else "CU"


def print_backend_summary(
    *,
    op_name: str,
    native_format: str,
    correctness_ref: str,
    vendor_backend=None,
    vendor_reason=None,
    run_vendor: bool = True,
) -> None:
    print(f"Runtime backend: {runtime_backend_name()}")
    print(f"FlagSparse native path: {op_name} ({native_format})")
    print(f"Correctness ref: {correctness_ref}")
    if not run_vendor:
        print("Vendor sparse baseline: disabled")
        return
    if vendor_backend is None:
        reason = vendor_reason or "no matching vendor sparse baseline"
        print(f"Vendor sparse baseline: N/A ({reason})")
        return
    print(f"Vendor sparse baseline: {sparse_backend_label(vendor_backend)}")
