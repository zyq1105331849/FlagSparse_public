"""Unit-diagonal CSC/transpose CW SpSV kernel entry.

The Triton implementation is shared with the non-unit transpose CW path and
selected by the UNIT_DIAG constexpr at launch time.  This module keeps the
allinone-style file split explicit; ``spsv.py`` imports it for unit-diagonal
dispatch.
"""

from .spsv_csc_n_cw import (
    _spsv_csc_preprocess_kernel,
    _spsv_csr_transpose_cw_kernel,
    _spsv_csr_transpose_cw_kernel_complex,
)

__all__ = [
    "_spsv_csc_preprocess_kernel",
    "_spsv_csr_transpose_cw_kernel",
    "_spsv_csr_transpose_cw_kernel_complex",
]
