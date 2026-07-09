"""Unit-diagonal CSR CW SpSV kernel entry.

The Triton implementation is shared with the non-unit CW path and selected by
the UNIT_DIAG constexpr at launch time.  This module keeps the allinone-style
file split explicit; ``spsv.py`` imports it for unit-diagonal dispatch.
"""

from .spsv_csr_n_cw import (
    _spsv_csr_cw_kernel,
    _spsv_csr_cw_kernel_complex,
)

__all__ = [
    "_spsv_csr_cw_kernel",
    "_spsv_csr_cw_kernel_complex",
]
