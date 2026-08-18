"""ALG8/NNZ-balance SpSV Triton kernels."""

import triton
import triton.language as tl

from .spsv_kernel_common import (
    _load_counter_i32_acquire,
    _load_ready_flag_i32,
    _load_scalar_fp32,
    _load_scalar_fp64,
    _publish_ready_flag_i32,
    _release_decrement_counter_i32_scalar,
)

@triton.jit
def _spsv_nnz_balance_preprocess_kernel(
    indices_ptr,
    indptr_ptr,
    indegree_ptr,
    row_idx_ptr,
    n_rows,
    WARP_SIZE: tl.constexpr,
    UNIT_DIAGONAL: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    lane = tl.arange(0, WARP_SIZE)
    ptr = start + lane
    degree = tl.zeros((WARP_SIZE,), dtype=tl.int32)
    active = ptr < end
    while tl.sum(active.to(tl.int32), axis=0) > 0:
        cols = tl.load(indices_ptr + ptr, mask=active, other=row + 1)
        if UNIT_DIAGONAL:
            valid = active & (cols < row)
        else:
            valid = active & (cols <= row)
        tl.store(row_idx_ptr + ptr, row, mask=valid)
        degree += valid.to(tl.int32)
        ptr = ptr + WARP_SIZE
        active = valid & (ptr < end)
    tl.store(indegree_ptr + row, tl.sum(degree, axis=0))

@triton.jit
def _spsv_csr_nnz_balance_kernel(
    launch_order_ptr,
    row_idx_ptr,
    col_idx_ptr,
    val_ptr,
    b_ptr,
    x_ptr,
    tmp_sum_ptr,
    ready_ptr,
    indegree_ptr,
    nnz,
    LOWER: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    val_id = tl.program_id(0)
    if val_id >= nnz:
        return
    entry_id = tl.load(launch_order_ptr + val_id)
    row = tl.load(row_idx_ptr + entry_id)
    col = tl.load(col_idx_ptr + entry_id)
    if LOWER:
        if row < col:
            return
    else:
        if row > col:
            return
    if USE_FP64_ACC:
        a = tl.load(val_ptr + entry_id).to(tl.float64)
    else:
        a = tl.load(val_ptr + entry_id).to(tl.float32)
    done = 0
    while done == 0:
        if row != col:
            dep_ready = _load_ready_flag_i32(ready_ptr, col)
            if dep_ready == 1:
                if USE_FP64_ACC:
                    dep_x = _load_scalar_fp64(x_ptr, col).to(tl.float64)
                else:
                    dep_x = _load_scalar_fp32(x_ptr, col).to(tl.float32)
                tl.atomic_add(tmp_sum_ptr + row, dep_x * a)
                _release_decrement_counter_i32_scalar(indegree_ptr, row)
                done = 1
        else:
            diag_degree = _load_counter_i32_acquire(indegree_ptr, row)
            if diag_degree == 1:
                if USE_FP64_ACC:
                    rhs = tl.load(b_ptr + row).to(tl.float64)
                    sum_val = tl.atomic_add(tmp_sum_ptr + row, 0.0).to(tl.float64)
                else:
                    rhs = tl.load(b_ptr + row).to(tl.float32)
                    sum_val = tl.atomic_add(tmp_sum_ptr + row, 0.0).to(tl.float32)
                diag_safe = tl.where(tl.abs(a) < DIAG_EPS, 1.0, a)
                out = (rhs - sum_val) / diag_safe
                out = tl.where(out == out, out, 0.0)
                tl.store(x_ptr + row, out)
                _publish_ready_flag_i32(ready_ptr, row)
                done = 1

@triton.jit
def _spsv_csr_nnz_balance_kernel_complex(
    launch_order_ptr,
    row_idx_ptr,
    col_idx_ptr,
    val_ri_ptr,
    b_ri_ptr,
    x_ri_ptr,
    tmp_sum_ri_ptr,
    ready_ptr,
    indegree_ptr,
    nnz,
    LOWER: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    val_id = tl.program_id(0)
    if val_id >= nnz:
        return
    entry_id = tl.load(launch_order_ptr + val_id)
    row = tl.load(row_idx_ptr + entry_id)
    col = tl.load(col_idx_ptr + entry_id)
    if LOWER:
        if row < col:
            return
    else:
        if row > col:
            return
    val_re = tl.load(val_ri_ptr + entry_id * 2)
    val_im = tl.load(val_ri_ptr + entry_id * 2 + 1)
    if USE_FP64_ACC:
        val_re = val_re.to(tl.float64)
        val_im = val_im.to(tl.float64)
    else:
        val_re = val_re.to(tl.float32)
        val_im = val_im.to(tl.float32)
    done = 0
    while done == 0:
        if row != col:
            dep_ready = _load_ready_flag_i32(ready_ptr, col)
            if dep_ready == 1:
                dep_x_re = tl.atomic_add(x_ri_ptr + col * 2, 0.0)
                dep_x_im = tl.atomic_add(x_ri_ptr + col * 2 + 1, 0.0)
                if USE_FP64_ACC:
                    dep_x_re = dep_x_re.to(tl.float64)
                    dep_x_im = dep_x_im.to(tl.float64)
                else:
                    dep_x_re = dep_x_re.to(tl.float32)
                    dep_x_im = dep_x_im.to(tl.float32)
                prod_re = dep_x_re * val_re - dep_x_im * val_im
                prod_im = dep_x_re * val_im + dep_x_im * val_re
                tl.atomic_add(tmp_sum_ri_ptr + row * 2, prod_re)
                tl.atomic_add(tmp_sum_ri_ptr + row * 2 + 1, prod_im)
                _release_decrement_counter_i32_scalar(indegree_ptr, row)
                done = 1
        if row == col:
            diag_degree = _load_counter_i32_acquire(indegree_ptr, row)
            if diag_degree == 1:
                rhs_re = tl.load(b_ri_ptr + row * 2)
                rhs_im = tl.load(b_ri_ptr + row * 2 + 1)
                sum_re = tl.atomic_add(tmp_sum_ri_ptr + row * 2, 0.0)
                sum_im = tl.atomic_add(tmp_sum_ri_ptr + row * 2 + 1, 0.0)
                if USE_FP64_ACC:
                    rhs_re = rhs_re.to(tl.float64)
                    rhs_im = rhs_im.to(tl.float64)
                    sum_re = sum_re.to(tl.float64)
                    sum_im = sum_im.to(tl.float64)
                else:
                    rhs_re = rhs_re.to(tl.float32)
                    rhs_im = rhs_im.to(tl.float32)
                    sum_re = sum_re.to(tl.float32)
                    sum_im = sum_im.to(tl.float32)
                num_re = rhs_re - sum_re
                num_im = rhs_im - sum_im
                den = val_re * val_re + val_im * val_im
                den_safe = tl.where(den < (DIAG_EPS * DIAG_EPS), 1.0, den)
                out_re = (num_re * val_re + num_im * val_im) / den_safe
                out_im = (num_im * val_re - num_re * val_im) / den_safe
                out_re = tl.where(out_re == out_re, out_re, 0.0)
                out_im = tl.where(out_im == out_im, out_im, 0.0)
                tl.store(x_ri_ptr + row * 2, out_re)
                tl.store(x_ri_ptr + row * 2 + 1, out_im)
                _publish_ready_flag_i32(ready_ptr, row)
                done = 1
