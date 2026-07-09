"""ALG2/level-schedule SpSV Triton kernels."""

import triton
import triton.language as tl

from .spsv_kernel_common import (
    _load_ready_flag_i32,
    _load_scalar_fp32,
    _load_scalar_fp64,
    _publish_i32_once,
    _publish_ready_flag_i32,
)

@triton.jit
def _spsv_levelschd_analysis_kernel(
    indices_ptr,
    indptr_ptr,
    levels_ptr,
    ready_ptr,
    indegree_ptr,
    n_rows,
    BLOCK_ROWS: tl.constexpr,
    UNIT_DIAGONAL: tl.constexpr,
):
    first_row = tl.program_id(0) * BLOCK_ROWS
    local_rows = tl.arange(0, BLOCK_ROWS)
    local_levels = tl.zeros((BLOCK_ROWS,), dtype=tl.int32)
    for local_row in range(BLOCK_ROWS):
        row = first_row + local_row
        if row < n_rows:
            start = tl.load(indptr_ptr + row)
            end = tl.load(indptr_ptr + row + 1)
            ptr = start
            max_level = tl.zeros((), dtype=tl.int32)
            degree = tl.zeros((), dtype=tl.int32)
            row_done = 0
            while row_done == 0:
                if ptr >= end:
                    row_done = 1
                else:
                    col = tl.load(indices_ptr + ptr)
                    if col < first_row:
                        dep_ready = _load_ready_flag_i32(ready_ptr, col)
                        while dep_ready == 0:
                            dep_ready = _load_ready_flag_i32(ready_ptr, col)
                        dep_level = tl.atomic_add(levels_ptr + col, 0)
                        max_level = tl.maximum(max_level, dep_level)
                        degree += 1
                        ptr += 1
                    elif col < row:
                        local_idx = col - first_row
                        dep_level = tl.sum(
                            tl.where(local_rows == local_idx, local_levels, 0),
                            axis=0,
                        )
                        max_level = tl.maximum(max_level, dep_level)
                        degree += 1
                        ptr += 1
                    else:
                        if (not UNIT_DIAGONAL) and (col == row):
                            degree += 1
                        row_done = 1
            row_level = max_level + 1
            _publish_i32_once(levels_ptr, row, row_level)
            tl.store(indegree_ptr + row, degree)
            local_levels = tl.where(local_rows == local_row, row_level, local_levels)
            _publish_ready_flag_i32(ready_ptr, row)

@triton.jit
def _spsv_csr_cw_levelschd_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    row_map_ptr,
    b_ptr,
    x_ptr,
    ready_ptr,
    n_rows,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    logical_row = tl.program_id(0)
    if logical_row >= n_rows:
        return
    row = tl.load(row_map_ptr + logical_row)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    ptr = start
    if USE_FP64_ACC:
        rhs = tl.load(b_ptr + row).to(tl.float64)
        tmp_sum = tl.zeros((), dtype=tl.float64)
    else:
        rhs = tl.load(b_ptr + row).to(tl.float32)
        tmp_sum = tl.zeros((), dtype=tl.float32)
    row_done = 0
    while row_done == 0:
        if ptr >= end:
            x_row = rhs * 0
            if USE_FP64_ACC:
                tl.atomic_add(x_ptr + row, x_row.to(tl.float64))
            else:
                tl.atomic_add(x_ptr + row, x_row.to(tl.float32))
            row_done = 1
        else:
            col = tl.load(indices_ptr + ptr)
            if col == row:
                if USE_FP64_ACC:
                    diag = tl.load(data_ptr + ptr).to(tl.float64)
                else:
                    diag = tl.load(data_ptr + ptr).to(tl.float32)
                diag_safe = tl.where(tl.abs(diag) < DIAG_EPS, 1.0, diag)
                x_row = (rhs - tmp_sum) / diag_safe
                x_row = tl.where(x_row == x_row, x_row, 0.0)
                if USE_FP64_ACC:
                    tl.atomic_add(x_ptr + row, x_row.to(tl.float64))
                else:
                    tl.atomic_add(x_ptr + row, x_row.to(tl.float32))
                row_done = 1
            else:
                dep_ready = _load_ready_flag_i32(ready_ptr, col)
                while dep_ready != 1:
                    dep_ready = _load_ready_flag_i32(ready_ptr, col)
                if USE_FP64_ACC:
                    a = tl.load(data_ptr + ptr).to(tl.float64)
                    y_dep = _load_scalar_fp64(x_ptr, col).to(tl.float64)
                else:
                    a = tl.load(data_ptr + ptr).to(tl.float32)
                    y_dep = _load_scalar_fp32(x_ptr, col).to(tl.float32)
                tmp_sum += a * y_dep
                ptr += 1
    _publish_ready_flag_i32(ready_ptr, row)

@triton.jit
def _spsv_csr_cw_levelschd_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    row_map_ptr,
    b_ri_ptr,
    x_ri_ptr,
    ready_ptr,
    n_rows,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    logical_row = tl.program_id(0)
    if logical_row >= n_rows:
        return
    row = tl.load(row_map_ptr + logical_row)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    ptr = start
    rhs_re = tl.load(b_ri_ptr + row * 2)
    rhs_im = tl.load(b_ri_ptr + row * 2 + 1)
    if USE_FP64_ACC:
        rhs_re = rhs_re.to(tl.float64)
        rhs_im = rhs_im.to(tl.float64)
        tmp_sum_re = tl.zeros((), dtype=tl.float64)
        tmp_sum_im = tl.zeros((), dtype=tl.float64)
    else:
        rhs_re = rhs_re.to(tl.float32)
        rhs_im = rhs_im.to(tl.float32)
        tmp_sum_re = tl.zeros((), dtype=tl.float32)
        tmp_sum_im = tl.zeros((), dtype=tl.float32)
    row_done = 0
    while row_done == 0:
        if ptr >= end:
            zero = rhs_re * 0
            tl.atomic_add(x_ri_ptr + row * 2, zero)
            tl.atomic_add(x_ri_ptr + row * 2 + 1, zero)
            row_done = 1
        else:
            col = tl.load(indices_ptr + ptr)
            if col == row:
                diag_re = tl.load(data_ri_ptr + ptr * 2)
                diag_im = tl.load(data_ri_ptr + ptr * 2 + 1)
                if USE_FP64_ACC:
                    diag_re = diag_re.to(tl.float64)
                    diag_im = diag_im.to(tl.float64)
                else:
                    diag_re = diag_re.to(tl.float32)
                    diag_im = diag_im.to(tl.float32)
                sum_re = rhs_re - tmp_sum_re
                sum_im = rhs_im - tmp_sum_im
                den = diag_re * diag_re + diag_im * diag_im
                den_safe = tl.where(den < (DIAG_EPS * DIAG_EPS), 1.0, den)
                out_re = (sum_re * diag_re + sum_im * diag_im) / den_safe
                out_im = (sum_im * diag_re - sum_re * diag_im) / den_safe
                out_re = tl.where(out_re == out_re, out_re, 0.0)
                out_im = tl.where(out_im == out_im, out_im, 0.0)
                tl.atomic_add(x_ri_ptr + row * 2, out_re)
                tl.atomic_add(x_ri_ptr + row * 2 + 1, out_im)
                row_done = 1
            else:
                dep_ready = _load_ready_flag_i32(ready_ptr, col)
                while dep_ready != 1:
                    dep_ready = _load_ready_flag_i32(ready_ptr, col)
                a_re = tl.load(data_ri_ptr + ptr * 2)
                a_im = tl.load(data_ri_ptr + ptr * 2 + 1)
                x_re = tl.atomic_add(x_ri_ptr + col * 2, 0.0)
                x_im = tl.atomic_add(x_ri_ptr + col * 2 + 1, 0.0)
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
                tmp_sum_re += a_re * x_re - a_im * x_im
                tmp_sum_im += a_re * x_im + a_im * x_re
                ptr += 1
    _publish_ready_flag_i32(ready_ptr, row)
