"""ALG3/ROC SpSV Triton kernels."""

import triton
import triton.language as tl

from .spsv_kernel_common import _publish_ready_flag_i32

@triton.jit
def _spsv_csr_roc_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    row_map_ptr,
    b_ptr,
    x_ptr,
    ready_ptr,
    n_rows,
    LOWER: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
    WARP_SIZE: tl.constexpr,
    LEVEL_SCHEDULED: tl.constexpr,
):
    logical_row = tl.program_id(0)
    if logical_row >= n_rows:
        return
    row = tl.load(row_map_ptr + logical_row)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    lanes = tl.arange(0, WARP_SIZE)
    ptr = start + lanes
    if USE_FP64_ACC:
        rhs = tl.load(b_ptr + row).to(tl.float64)
        local_sum = tl.where(lanes == 0, rhs, 0.0).to(tl.float64)
        zero_vec = tl.zeros((WARP_SIZE,), dtype=tl.float64)
    else:
        rhs = tl.load(b_ptr + row).to(tl.float32)
        local_sum = tl.where(lanes == 0, rhs, 0.0).to(tl.float32)
        zero_vec = tl.zeros((WARP_SIZE,), dtype=tl.float32)

    loop_done = 0
    while loop_done == 0:
        active = ptr < end
        col = tl.load(indices_ptr + ptr, mask=active, other=row)
        dep_mask = active & (col < row if LOWER else col > row)
        if tl.sum(dep_mask.to(tl.int32), axis=0) == 0:
            loop_done = 1
        else:
            if LEVEL_SCHEDULED:
                advance_mask = dep_mask
            else:
                dep_ready = tl.atomic_add(
                    ready_ptr + col,
                    tl.zeros((WARP_SIZE,), dtype=tl.int32),
                    mask=dep_mask,
                )
                advance_mask = dep_mask & (dep_ready != 0)
            a = tl.load(data_ptr + ptr, mask=advance_mask, other=0.0)
            if USE_FP64_ACC:
                a = a.to(tl.float64)
                if LEVEL_SCHEDULED:
                    y_dep = tl.load(x_ptr + col, mask=advance_mask, other=0.0).to(tl.float64)
                else:
                    y_dep = tl.atomic_add(x_ptr + col, zero_vec, mask=advance_mask).to(tl.float64)
            else:
                a = a.to(tl.float32)
                if LEVEL_SCHEDULED:
                    y_dep = tl.load(x_ptr + col, mask=advance_mask, other=0.0).to(tl.float32)
                else:
                    y_dep = tl.atomic_add(x_ptr + col, zero_vec, mask=advance_mask).to(tl.float32)
            local_sum += tl.where(advance_mask, -a * y_dep, 0.0)
            ptr = ptr + tl.where(advance_mask, WARP_SIZE, 0)

    active = ptr < end
    col = tl.load(indices_ptr + ptr, mask=active, other=row + 1)
    diag_mask = active & (col == row)
    diag = tl.load(data_ptr + ptr, mask=diag_mask, other=0.0)
    if USE_FP64_ACC:
        diag = diag.to(tl.float64)
    else:
        diag = diag.to(tl.float32)
    diag_val = tl.sum(diag, axis=0)
    diag_safe = tl.where(tl.abs(diag_val) < DIAG_EPS, 1.0, diag_val)
    out = tl.sum(local_sum, axis=0) / diag_safe
    out = tl.where(out == out, out, 0.0)
    if LEVEL_SCHEDULED:
        tl.store(x_ptr + row, out)
    elif USE_FP64_ACC:
        tl.atomic_add(x_ptr + row, out.to(tl.float64))
    else:
        tl.atomic_add(x_ptr + row, out.to(tl.float32))
    if not LEVEL_SCHEDULED:
        _publish_ready_flag_i32(ready_ptr, row)

@triton.jit
def _spsv_csr_roc_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    row_map_ptr,
    b_ri_ptr,
    x_ri_ptr,
    ready_ptr,
    n_rows,
    LOWER: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
    WARP_SIZE: tl.constexpr,
    LEVEL_SCHEDULED: tl.constexpr,
):
    logical_row = tl.program_id(0)
    if logical_row >= n_rows:
        return
    row = tl.load(row_map_ptr + logical_row)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    lanes = tl.arange(0, WARP_SIZE)
    ptr = start + lanes

    rhs_re = tl.load(b_ri_ptr + row * 2)
    rhs_im = tl.load(b_ri_ptr + row * 2 + 1)
    if USE_FP64_ACC:
        rhs_re = rhs_re.to(tl.float64)
        rhs_im = rhs_im.to(tl.float64)
        local_sum_re = tl.where(lanes == 0, rhs_re, 0.0).to(tl.float64)
        local_sum_im = tl.where(lanes == 0, rhs_im, 0.0).to(tl.float64)
        zero_vec = tl.zeros((WARP_SIZE,), dtype=tl.float64)
    else:
        rhs_re = rhs_re.to(tl.float32)
        rhs_im = rhs_im.to(tl.float32)
        local_sum_re = tl.where(lanes == 0, rhs_re, 0.0).to(tl.float32)
        local_sum_im = tl.where(lanes == 0, rhs_im, 0.0).to(tl.float32)
        zero_vec = tl.zeros((WARP_SIZE,), dtype=tl.float32)

    loop_done = 0
    while loop_done == 0:
        active = ptr < end
        col = tl.load(indices_ptr + ptr, mask=active, other=row)
        dep_mask = active & (col < row if LOWER else col > row)
        if tl.sum(dep_mask.to(tl.int32), axis=0) == 0:
            loop_done = 1
        else:
            if LEVEL_SCHEDULED:
                advance_mask = dep_mask
            else:
                dep_ready = tl.atomic_add(
                    ready_ptr + col,
                    tl.zeros((WARP_SIZE,), dtype=tl.int32),
                    mask=dep_mask,
                )
                advance_mask = dep_mask & (dep_ready != 0)
            a_re = tl.load(data_ri_ptr + ptr * 2, mask=advance_mask, other=0.0)
            a_im = tl.load(data_ri_ptr + ptr * 2 + 1, mask=advance_mask, other=0.0)
            if LEVEL_SCHEDULED:
                x_re = tl.load(x_ri_ptr + col * 2, mask=advance_mask, other=0.0)
                x_im = tl.load(x_ri_ptr + col * 2 + 1, mask=advance_mask, other=0.0)
            else:
                x_re = tl.atomic_add(x_ri_ptr + col * 2, zero_vec, mask=advance_mask)
                x_im = tl.atomic_add(x_ri_ptr + col * 2 + 1, zero_vec, mask=advance_mask)
            if USE_FP64_ACC:
                a_re = a_re.to(tl.float64)
                a_im = a_im.to(tl.float64)
                x_re = x_re.to(tl.float64)
                x_im = x_im.to(tl.float64)
            else:
                a_re = a_re.to(tl.float32)
                a_im = a_im.to(tl.float32)
                x_re = x_re.to(tl.float32)
                x_im = x_im.to(tl.float32)
            prod_re = a_re * x_re - a_im * x_im
            prod_im = a_re * x_im + a_im * x_re
            local_sum_re += tl.where(advance_mask, -prod_re, 0.0)
            local_sum_im += tl.where(advance_mask, -prod_im, 0.0)
            ptr = ptr + tl.where(advance_mask, WARP_SIZE, 0)

    active = ptr < end
    col = tl.load(indices_ptr + ptr, mask=active, other=row + 1)
    diag_mask = active & (col == row)
    diag_re = tl.load(data_ri_ptr + ptr * 2, mask=diag_mask, other=0.0)
    diag_im = tl.load(data_ri_ptr + ptr * 2 + 1, mask=diag_mask, other=0.0)
    if USE_FP64_ACC:
        diag_re = diag_re.to(tl.float64)
        diag_im = diag_im.to(tl.float64)
    else:
        diag_re = diag_re.to(tl.float32)
        diag_im = diag_im.to(tl.float32)
    diag_re = tl.sum(diag_re, axis=0)
    diag_im = tl.sum(diag_im, axis=0)
    sum_re = tl.sum(local_sum_re, axis=0)
    sum_im = tl.sum(local_sum_im, axis=0)
    den = diag_re * diag_re + diag_im * diag_im
    den_safe = tl.where(den < (DIAG_EPS * DIAG_EPS), 1.0, den)
    out_re = (sum_re * diag_re + sum_im * diag_im) / den_safe
    out_im = (sum_im * diag_re - sum_re * diag_im) / den_safe
    out_re = tl.where(out_re == out_re, out_re, 0.0)
    out_im = tl.where(out_im == out_im, out_im, 0.0)
    if LEVEL_SCHEDULED:
        tl.store(x_ri_ptr + row * 2, out_re)
        tl.store(x_ri_ptr + row * 2 + 1, out_im)
    else:
        tl.atomic_add(x_ri_ptr + row * 2, out_re)
        tl.atomic_add(x_ri_ptr + row * 2 + 1, out_im)
        _publish_ready_flag_i32(ready_ptr, row)
