"""Small DCU/HIP-aware SpMM launch strategy helpers.

This is deliberately not a large autotune grid.  ``default`` means the current
operator heuristic, while ``dcu`` applies one mild HIP/DCU-biased adjustment for
the kernels that accept explicit launch overrides.
"""

from dataclasses import dataclass

from ._common import torch


SPMM_DCU_TUNING_STRATEGIES = (
    "default",
    "dcu",
)

_SPMM_DCU_STRATEGY_ALIASES = {
    "hip": "dcu",
    "dcu_auto": "dcu",
    "dcu_small_n": "dcu",
    "dcu_balanced": "dcu",
    "dcu_long_row": "dcu",
    "dcu_wide_n": "dcu",
    "dcu_wave64": "dcu",
}


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
    token = _SPMM_DCU_STRATEGY_ALIASES.get(token, token)
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
    elif strategy == "dcu":
        # DCU/BW-class HIP devices generally expose wavefront=64.  Keep dense-N
        # tiles modest to reduce lane waste on small dense widths, and avoid the
        # very aggressive long-row candidates that caused excessive runtimes in
        # broad matrix sweeps.
        if dense_n <= 16:
            block_n = 16
            num_warps = 1
        elif dense_n <= 64:
            block_n = 32 if wave >= 64 else 64
            num_warps = 2
        else:
            block_n = 64
            num_warps = 4
        block_nnz = 64 if fmt == "csr" else 128
        if max_row_nnz >= 512 or nnz >= 1_000_000:
            block_nnz = 128 if fmt == "csr" else 256
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
