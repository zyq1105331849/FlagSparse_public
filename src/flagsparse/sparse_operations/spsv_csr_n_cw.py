"""ALG1/CW SpSV Triton kernels."""

import triton
import triton.language as tl

from .spsv_kernel_common import _publish_ready_flag_i32

@triton.jit
def _spsv_csr_cw_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    x_ptr,
    ready_ptr,
    row_counter_ptr,
    n_rows,
    LOWER: tl.constexpr,
    REVERSE_ORDER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    logical_row = tl.atomic_add(row_counter_ptr, 1)
    while logical_row < n_rows:
        row = tl.where(REVERSE_ORDER, n_rows - 1 - logical_row, logical_row)
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
            if UNIT_DIAG:
                if ptr >= end:
                    x_row = rhs - tmp_sum
                    x_row = tl.where(x_row == x_row, x_row, 0.0)
                    tl.store(x_ptr + row, x_row)
                    row_done = 1
                else:
                    col = tl.load(indices_ptr + ptr)
                    stop_at_diag = (col >= row) if LOWER else (col <= row)
                    if stop_at_diag:
                        x_row = rhs - tmp_sum
                        x_row = tl.where(x_row == x_row, x_row, 0.0)
                        tl.store(x_ptr + row, x_row)
                        row_done = 1
                    else:
                        dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        while dep_ready != 1:
                            dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        if USE_FP64_ACC:
                            a = tl.load(data_ptr + ptr).to(tl.float64)
                            y_dep = tl.load(x_ptr + col).to(tl.float64)
                        else:
                            a = tl.load(data_ptr + ptr).to(tl.float32)
                            y_dep = tl.load(x_ptr + col).to(tl.float32)
                        tmp_sum += a * y_dep
                        ptr += 1
            else:
                if ptr >= end:
                    x_row = rhs * 0
                    tl.store(x_ptr + row, x_row)
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
                        tl.store(x_ptr + row, x_row)
                        row_done = 1
                    else:
                        dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        while dep_ready != 1:
                            dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        if USE_FP64_ACC:
                            a = tl.load(data_ptr + ptr).to(tl.float64)
                            y_dep = tl.load(x_ptr + col).to(tl.float64)
                        else:
                            a = tl.load(data_ptr + ptr).to(tl.float32)
                            y_dep = tl.load(x_ptr + col).to(tl.float32)
                        tmp_sum += a * y_dep
                        ptr += 1
        _publish_ready_flag_i32(ready_ptr, row)
        logical_row = tl.atomic_add(row_counter_ptr, 1)

@triton.jit
def _spsv_csr_cw_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    b_ri_ptr,
    x_ri_ptr,
    ready_ptr,
    row_counter_ptr,
    n_rows,
    LOWER: tl.constexpr,
    REVERSE_ORDER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    logical_row = tl.atomic_add(row_counter_ptr, 1)
    lane2 = tl.arange(0, 2)
    while logical_row < n_rows:
        row = tl.where(REVERSE_ORDER, n_rows - 1 - logical_row, logical_row)
        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
        rhs_re = tl.load(b_ri_ptr + row * 2)
        rhs_im = tl.load(b_ri_ptr + row * 2 + 1)
        if USE_FP64_ACC:
            rhs_re = rhs_re.to(tl.float64)
            rhs_im = rhs_im.to(tl.float64)
            tmp_re = tl.zeros((), dtype=tl.float64)
            tmp_im = tl.zeros((), dtype=tl.float64)
        else:
            rhs_re = rhs_re.to(tl.float32)
            rhs_im = rhs_im.to(tl.float32)
            tmp_re = tl.zeros((), dtype=tl.float32)
            tmp_im = tl.zeros((), dtype=tl.float32)
        ptr = start
        row_done = 0
        while row_done == 0:
            if UNIT_DIAG:
                if ptr >= end:
                    x_re_out = rhs_re - tmp_re
                    x_im_out = rhs_im - tmp_im
                    x_re_out = tl.where(x_re_out == x_re_out, x_re_out, 0.0)
                    x_im_out = tl.where(x_im_out == x_im_out, x_im_out, 0.0)
                    out_vals = tl.where(lane2 == 0, x_re_out, x_im_out)
                    tl.store(x_ri_ptr + row * 2 + lane2, out_vals)
                    row_done = 1
                else:
                    col = tl.load(indices_ptr + ptr)
                    stop_at_diag = (col >= row) if LOWER else (col <= row)
                    if stop_at_diag:
                        x_re_out = rhs_re - tmp_re
                        x_im_out = rhs_im - tmp_im
                        x_re_out = tl.where(x_re_out == x_re_out, x_re_out, 0.0)
                        x_im_out = tl.where(x_im_out == x_im_out, x_im_out, 0.0)
                        out_vals = tl.where(lane2 == 0, x_re_out, x_im_out)
                        tl.store(x_ri_ptr + row * 2 + lane2, out_vals)
                        row_done = 1
                    else:
                        dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        while dep_ready != 1:
                            dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        x_re = tl.load(x_ri_ptr + col * 2)
                        x_im = tl.load(x_ri_ptr + col * 2 + 1)
                        a_re = tl.load(data_ri_ptr + ptr * 2)
                        a_im = tl.load(data_ri_ptr + ptr * 2 + 1)
                        if USE_FP64_ACC:
                            x_re = x_re.to(tl.float64)
                            x_im = x_im.to(tl.float64)
                            a_re = a_re.to(tl.float64)
                            a_im = a_im.to(tl.float64)
                        else:
                            x_re = x_re.to(tl.float32)
                            x_im = x_im.to(tl.float32)
                            a_re = a_re.to(tl.float32)
                            a_im = a_im.to(tl.float32)
                        tmp_re += a_re * x_re - a_im * x_im
                        tmp_im += a_re * x_im + a_im * x_re
                        ptr += 1
            else:
                if ptr >= end:
                    out_vals = tl.where(lane2 == 0, rhs_re * 0, rhs_im * 0)
                    tl.store(x_ri_ptr + row * 2 + lane2, out_vals)
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
                        num_re = rhs_re - tmp_re
                        num_im = rhs_im - tmp_im
                        den = diag_re * diag_re + diag_im * diag_im
                        den_safe = tl.where(den < (DIAG_EPS * DIAG_EPS), 1.0, den)
                        x_re_out = (num_re * diag_re + num_im * diag_im) / den_safe
                        x_im_out = (num_im * diag_re - num_re * diag_im) / den_safe
                        x_re_out = tl.where(x_re_out == x_re_out, x_re_out, 0.0)
                        x_im_out = tl.where(x_im_out == x_im_out, x_im_out, 0.0)
                        out_vals = tl.where(lane2 == 0, x_re_out, x_im_out)
                        tl.store(x_ri_ptr + row * 2 + lane2, out_vals)
                        row_done = 1
                    else:
                        dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        while dep_ready != 1:
                            dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        x_re = tl.load(x_ri_ptr + col * 2)
                        x_im = tl.load(x_ri_ptr + col * 2 + 1)
                        a_re = tl.load(data_ri_ptr + ptr * 2)
                        a_im = tl.load(data_ri_ptr + ptr * 2 + 1)
                        if USE_FP64_ACC:
                            x_re = x_re.to(tl.float64)
                            x_im = x_im.to(tl.float64)
                            a_re = a_re.to(tl.float64)
                            a_im = a_im.to(tl.float64)
                        else:
                            x_re = x_re.to(tl.float32)
                            x_im = x_im.to(tl.float32)
                            a_re = a_re.to(tl.float32)
                            a_im = a_im.to(tl.float32)
                        tmp_re += a_re * x_re - a_im * x_im
                        tmp_im += a_re * x_im + a_im * x_re
                        ptr += 1
        _publish_ready_flag_i32(ready_ptr, row)
        logical_row = tl.atomic_add(row_counter_ptr, 1)
