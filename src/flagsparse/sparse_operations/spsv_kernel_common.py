"""Shared Triton helpers for SpSV kernels."""

import triton
import triton.language as tl

@triton.jit
def _publish_ready_flag_i32(flag_ptr, idx):
    """Publish a ready flag through an atomic write-like operation."""

    tl.atomic_add(flag_ptr + idx, 1)

@triton.jit
def _load_ready_flag_i32(flag_ptr, idx):
    """Mirror the original volatile/atomic polling pattern more closely."""

    return tl.atomic_add(flag_ptr + idx, 0)

@triton.jit
def _load_counter_i32_acquire(counter_ptr, idx):
    """Poll an int32 dependency counter with acquire semantics."""

    return tl.atomic_add(counter_ptr + idx, 0, sem="acquire")

@triton.jit
def _release_decrement_counter_i32(counter_ptr, idx, mask):
    """Release prior contribution writes before decrementing a dependency counter."""

    tl.atomic_add(counter_ptr + idx, -1, mask=mask, sem="release")

@triton.jit
def _release_decrement_counter_i32_scalar(counter_ptr, idx):
    """Scalar form of dependency release for kernels without a vector mask."""

    tl.atomic_add(counter_ptr + idx, -1, sem="release")

@triton.jit
def _publish_i32_once(slot_ptr, idx, value):
    """Publish a single int32 payload via an atomic write-like update."""

    tl.atomic_add(slot_ptr + idx, value)

@triton.jit
def _load_scalar_fp32(ptr, idx):
    return tl.atomic_add(ptr + idx, 0.0)

@triton.jit
def _load_scalar_fp64(ptr, idx):
    return tl.atomic_add(ptr + idx, 0.0)

@triton.jit
def _complex_atomic_add_interleaved(ptr_ri, idx, delta_re, delta_im, mask):
    """Complex atomicAdd equivalent for interleaved real/imag buffers."""

    tl.atomic_add(ptr_ri + idx * 2, delta_re, mask=mask)
    tl.atomic_add(ptr_ri + idx * 2 + 1, delta_im, mask=mask)

@triton.jit
def _propagate_real(residual_ptr, idx, delta, mask):
    """Publish a real contribution into shared residual state."""

    tl.atomic_add(residual_ptr + idx, delta, mask=mask)

@triton.jit
def _propagate_then_release_real(residual_ptr, indegree_ptr, idx, delta, mask):
    """Approximate 'write contribution then decrement dependency count'."""

    _propagate_real(residual_ptr, idx, delta, mask)
    _release_decrement_counter_i32(indegree_ptr, idx, mask)

@triton.jit
def _propagate_complex(residual_ri_ptr, idx, delta_re, delta_im, mask):
    """Publish a complex contribution into shared residual state."""

    _complex_atomic_add_interleaved(residual_ri_ptr, idx, delta_re, delta_im, mask)

@triton.jit
def _propagate_then_release_complex(residual_ri_ptr, indegree_ptr, idx, delta_re, delta_im, mask):
    """Complex propagation + dependency release for transpose-style solve."""

    _propagate_complex(residual_ri_ptr, idx, delta_re, delta_im, mask)
    _release_decrement_counter_i32(indegree_ptr, idx, mask)
