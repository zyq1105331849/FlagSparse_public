"""Transpose/CONJ SpSV Triton kernels."""

import triton
import triton.language as tl

from .spsv_kernel_common import (
    _load_counter_i32_acquire,
    _propagate_then_release_complex,
    _propagate_then_release_real,
)

@triton.jit
def _spsv_csc_preprocess_kernel(
    indices_ptr,
    indptr_ptr,
    indegree_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
):
    col = tl.program_id(0)
    if col >= n_rows:
        return
    start = tl.load(indptr_ptr + col)
    end = tl.load(indptr_ptr + col + 1)
    for seg in range(MAX_SEGMENTS):
        idx = start + seg * BLOCK_NNZ
        offsets = idx + tl.arange(0, BLOCK_NNZ)
        mask = offsets < end
        row = tl.load(indices_ptr + offsets, mask=mask, other=0)
        if LOWER:
            dep_mask = mask & (row > col)
        else:
            dep_mask = mask & (row < col)
        tl.atomic_add(indegree_ptr + row, 1, mask=dep_mask)

@triton.jit
def _spsv_csr_transpose_cw_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    indegree_ptr,
    residual_ptr,
    x_ptr,
    row_counter_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    REVERSE_ORDER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    logical_row = tl.atomic_add(row_counter_ptr, 1)
    while logical_row < n_rows:
        row = tl.where(REVERSE_ORDER, n_rows - 1 - logical_row, logical_row)
        dep_ready = _load_counter_i32_acquire(indegree_ptr, row)
        while dep_ready != 0:
            dep_ready = _load_counter_i32_acquire(indegree_ptr, row)

        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
        rhs = tl.load(residual_ptr + row)
        if UNIT_DIAG:
            diag = rhs * 0 + 1.0
            diag_count = 1
        else:
            diag = rhs * 0
            diag_count = tl.zeros((), dtype=tl.int32)
            for seg in range(MAX_SEGMENTS):
                idx = start + seg * BLOCK_NNZ
                offsets = idx + tl.arange(0, BLOCK_NNZ)
                mask = offsets < end
                a = tl.load(data_ptr + offsets, mask=mask, other=0.0)
                dep_row = tl.load(indices_ptr + offsets, mask=mask, other=0)
                is_diag = dep_row == row
                diag_count = diag_count + tl.sum(
                    tl.where(mask & is_diag, 1, 0), axis=0
                )
                diag = diag + tl.sum(tl.where(mask & is_diag, a, 0.0), axis=0)
        diag_safe = tl.where(tl.abs(diag) < DIAG_EPS, 1.0, diag)
        x_row = tl.where(diag_count == 0, rhs * 0, rhs / diag_safe)
        x_row = tl.where(x_row == x_row, x_row, 0.0)
        tl.store(x_ptr + row, x_row)

        for seg in range(MAX_SEGMENTS):
            idx = start + seg * BLOCK_NNZ
            offsets = idx + tl.arange(0, BLOCK_NNZ)
            mask = offsets < end
            a = tl.load(data_ptr + offsets, mask=mask, other=0.0)
            col = tl.load(indices_ptr + offsets, mask=mask, other=0)
            if LOWER:
                target_mask = mask & (col > row)
            else:
                target_mask = mask & (col < row)
            _propagate_then_release_real(
                residual_ptr, indegree_ptr, col, -a * x_row, target_mask
            )
        logical_row = tl.atomic_add(row_counter_ptr, 1)

@triton.jit
def _spsv_csr_transpose_cw_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    indegree_ptr,
    residual_ri_ptr,
    x_ri_ptr,
    row_counter_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    REVERSE_ORDER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    CONJ_TRANS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    logical_row = tl.atomic_add(row_counter_ptr, 1)
    lane2 = tl.arange(0, 2)
    while logical_row < n_rows:
        row = tl.where(REVERSE_ORDER, n_rows - 1 - logical_row, logical_row)
        dep_ready = _load_counter_i32_acquire(indegree_ptr, row)
        while dep_ready != 0:
            dep_ready = _load_counter_i32_acquire(indegree_ptr, row)
        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)

        rhs_re = tl.load(residual_ri_ptr + row * 2)
        rhs_im = tl.load(residual_ri_ptr + row * 2 + 1)
        if USE_FP64_ACC:
            rhs_re = rhs_re.to(tl.float64)
            rhs_im = rhs_im.to(tl.float64)
        else:
            rhs_re = rhs_re.to(tl.float32)
            rhs_im = rhs_im.to(tl.float32)

        if UNIT_DIAG:
            diag_re = rhs_re * 0 + 1.0
            diag_im = rhs_im * 0
            diag_count = 1
        else:
            if USE_FP64_ACC:
                diag_re = tl.zeros((), dtype=tl.float64)
                diag_im = tl.zeros((), dtype=tl.float64)
            else:
                diag_re = tl.zeros((), dtype=tl.float32)
                diag_im = tl.zeros((), dtype=tl.float32)
            diag_count = tl.zeros((), dtype=tl.int32)
            for seg in range(MAX_SEGMENTS):
                idx = start + seg * BLOCK_NNZ
                offsets = idx + tl.arange(0, BLOCK_NNZ)
                mask = offsets < end
                dep_row = tl.load(indices_ptr + offsets, mask=mask, other=0)
                a_re = tl.load(data_ri_ptr + offsets * 2, mask=mask, other=0.0)
                a_im = tl.load(data_ri_ptr + offsets * 2 + 1, mask=mask, other=0.0)
                if CONJ_TRANS:
                    a_im = -a_im
                if USE_FP64_ACC:
                    a_re = a_re.to(tl.float64)
                    a_im = a_im.to(tl.float64)
                else:
                    a_re = a_re.to(tl.float32)
                    a_im = a_im.to(tl.float32)
                is_diag = dep_row == row
                diag_count = diag_count + tl.sum(
                    tl.where(mask & is_diag, 1, 0), axis=0
                )
                diag_re = diag_re + tl.sum(tl.where(mask & is_diag, a_re, 0.0), axis=0)
                diag_im = diag_im + tl.sum(tl.where(mask & is_diag, a_im, 0.0), axis=0)

        den = diag_re * diag_re + diag_im * diag_im
        den_safe = tl.where(den < (DIAG_EPS * DIAG_EPS), 1.0, den)
        x_re_div = (rhs_re * diag_re + rhs_im * diag_im) / den_safe
        x_im_div = (rhs_im * diag_re - rhs_re * diag_im) / den_safe
        x_re_out = tl.where(diag_count == 0, rhs_re * 0, x_re_div)
        x_im_out = tl.where(diag_count == 0, rhs_im * 0, x_im_div)
        x_re_out = tl.where(x_re_out == x_re_out, x_re_out, 0.0)
        x_im_out = tl.where(x_im_out == x_im_out, x_im_out, 0.0)

        out_vals = tl.where(lane2 == 0, x_re_out, x_im_out)
        tl.store(x_ri_ptr + row * 2 + lane2, out_vals)

        for seg in range(MAX_SEGMENTS):
            idx = start + seg * BLOCK_NNZ
            offsets = idx + tl.arange(0, BLOCK_NNZ)
            mask = offsets < end
            col = tl.load(indices_ptr + offsets, mask=mask, other=0)
            a_re = tl.load(data_ri_ptr + offsets * 2, mask=mask, other=0.0)
            a_im = tl.load(data_ri_ptr + offsets * 2 + 1, mask=mask, other=0.0)
            if CONJ_TRANS:
                a_im = -a_im
            if USE_FP64_ACC:
                a_re = a_re.to(tl.float64)
                a_im = a_im.to(tl.float64)
            else:
                a_re = a_re.to(tl.float32)
                a_im = a_im.to(tl.float32)
            if LOWER:
                target_mask = mask & (col > row)
            else:
                target_mask = mask & (col < row)
            prod_re = a_re * x_re_out - a_im * x_im_out
            prod_im = a_re * x_im_out + a_im * x_re_out
            _propagate_then_release_complex(
                residual_ri_ptr,
                indegree_ptr,
                col,
                -prod_re,
                -prod_im,
                target_mask,
            )
        logical_row = tl.atomic_add(row_counter_ptr, 1)
