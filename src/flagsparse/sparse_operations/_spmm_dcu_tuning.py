"""DCU/HIP-aware SpMM launch strategy helpers.

The strategies in this module are intentionally conservative candidates for
benchmark sweeps.  They do not change CUDA defaults by themselves; callers can
opt in to a strategy and pass the returned launch overrides to CSR/COO SpMM.
"""

from dataclasses import dataclass

from ._common import torch


SPMM_DCU_TUNING_STRATEGIES = (
    "default",
    "dcu_small_n",
    "dcu_balanced",
    "dcu_long_row",
    "dcu_wide_n",
    "dcu_wave64",
)


@dataclass(frozen=True)
class SpmmDcuLaunchStrategy:
    strategy_name: str
    block_n: int | None
    block_nnz: int | None
    num_warps: int | None
    num_stages: int | None
    backend: str
    device_name: str
    device_warp_size: int

    def as_dict(self):
        return {
            "strategy_name": self.strategy_name,
            "block_n": self.block_n,
            "block_nnz": self.block_nnz,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "backend": self.backend,
            "device_name": self.device_name,
            "device_warp_size": self.device_warp_size,
        }


def normalize_spmm_dcu_strategy(strategy):
    token = "default" if strategy is None else str(strategy).strip().lower()
    if token not in SPMM_DCU_TUNING_STRATEGIES:
        allowed = ", ".join(SPMM_DCU_TUNING_STRATEGIES)
        raise ValueError(f"unsupported SpMM tuning strategy {strategy!r}; allowed: {allowed}")
    return token


def get_spmm_backend_info(device=None):
    if not torch.cuda.is_available():
        return {
            "backend": "unavailable",
            "device_name": "",
            "device_warp_size": 64 if getattr(torch.version, "hip", None) else 32,
        }
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    backend = "hip" if getattr(torch.version, "hip", None) is not None else "cuda"
    return {
        "backend": backend,
        "device_name": str(getattr(props, "name", "")),
        "device_warp_size": int(getattr(props, "warp_size", 64 if backend == "hip" else 32) or 32),
    }


def _round_up_to(value, multiple):
    value = max(1, int(value))
    multiple = max(1, int(multiple))
    return ((value + multiple - 1) // multiple) * multiple


def resolve_spmm_dcu_launch_strategy(
    strategy,
    *,
    n_dense_cols,
    max_row_nnz=0,
    nnz=0,
    fmt="csr",
    dtype=None,
    device=None,
):
    """Return launch overrides for one SpMM DCU tuning strategy.

    ``fmt`` is used only to bias the candidate configuration.  ``default``
    returns ``None`` overrides so existing operator heuristics remain active.
    """
    del dtype
    strategy = normalize_spmm_dcu_strategy(strategy)
    info = get_spmm_backend_info(device)
    dense_n = max(1, int(n_dense_cols))
    max_row_nnz = max(0, int(max_row_nnz or 0))
    nnz = max(0, int(nnz or 0))
    fmt = str(fmt).strip().lower()
    wave = max(1, int(info["device_warp_size"] or 64))

    block_n = None
    block_nnz = None
    num_warps = None
    num_stages = None

    if strategy == "default":
        pass
    elif strategy == "dcu_small_n":
        block_n = min(32, _round_up_to(dense_n, 8))
        block_nnz = 64 if fmt == "csr" else 128
        num_warps = 1
        num_stages = 1
    elif strategy == "dcu_balanced":
        block_n = 32 if dense_n <= 32 else 64
        block_nnz = 128 if fmt == "csr" else 256
        num_warps = 2 if dense_n <= 32 else 4
        num_stages = 1
    elif strategy == "dcu_long_row":
        block_n = 32 if dense_n <= 32 else 64
        if max_row_nnz >= 1024 or nnz >= 1_000_000:
            block_nnz = 512
        else:
            block_nnz = 256
        num_warps = 4 if dense_n <= 64 else 8
        num_stages = 1
    elif strategy == "dcu_wide_n":
        block_n = 64 if dense_n <= 64 else 128
        block_nnz = 128 if fmt == "csr" else 256
        num_warps = 4 if dense_n <= 64 else 8
        num_stages = 1
    elif strategy == "dcu_wave64":
        block_n = min(128, max(16, _round_up_to(min(dense_n, wave * 2), 16)))
        block_nnz = 64 if max_row_nnz <= wave else 128
        if max_row_nnz >= 512 or nnz >= 1_000_000:
            block_nnz = 256
        num_warps = 1 if dense_n <= 16 else (2 if dense_n <= 64 else 4)
        num_stages = 1

    return SpmmDcuLaunchStrategy(
        strategy_name=strategy,
        block_n=None if block_n is None else int(block_n),
        block_nnz=None if block_nnz is None else int(block_nnz),
        num_warps=None if num_warps is None else int(num_warps),
        num_stages=None if num_stages is None else int(num_stages),
        backend=info["backend"],
        device_name=info["device_name"],
        device_warp_size=int(info["device_warp_size"]),
    )


__all__ = (
    "SPMM_DCU_TUNING_STRATEGIES",
    "SpmmDcuLaunchStrategy",
    "get_spmm_backend_info",
    "normalize_spmm_dcu_strategy",
    "resolve_spmm_dcu_launch_strategy",
)
