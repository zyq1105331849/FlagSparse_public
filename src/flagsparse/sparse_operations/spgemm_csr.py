"""CSR SpGEMM (A@B) with two-phase structure/value build."""

import ctypes

from . import _common as _common_mod
from ._common import *

hip = _common_mod.hip
hipsparse = _common_mod.hipsparse
HipPointer = _common_mod.HipPointer
_hip_check_result = _common_mod._hip_check_result
_hipsparse_lookup = _common_mod._hipsparse_lookup
_hipsparse_unavailable_reason = _common_mod._hipsparse_unavailable_reason
_hipsparse_value_type = _common_mod._hipsparse_value_type
_hipsparse_scalar = _common_mod._hipsparse_scalar
_hipsparse_index_type = _common_mod._hipsparse_index_type

SUPPORTED_SPGEMM_VALUE_DTYPES = (torch.float32, torch.float64)
_SPGEMM_COUNT_MAX_EXPANDED = 2_000_000
_SPGEMM_FILL_MAX_EXPANDED = 1_200_000
_SPGEMM_MAX_ROWS_PER_CHUNK = 4096
_SPGEMM_BUCKET_SHORT = 0
_SPGEMM_BUCKET_MEDIUM = 1
_SPGEMM_BUCKET_LONG = 2
_SPGEMM_BUCKET_ORDER = (
    _SPGEMM_BUCKET_SHORT,
    _SPGEMM_BUCKET_MEDIUM,
    _SPGEMM_BUCKET_LONG,
)
_SPGEMM_BUCKET_LABELS = {
    _SPGEMM_BUCKET_SHORT: "short",
    _SPGEMM_BUCKET_MEDIUM: "medium",
    _SPGEMM_BUCKET_LONG: "long",
}
_SPGEMM_BUCKET_COUNT_BUDGETS = {
    _SPGEMM_BUCKET_SHORT: 4_000_000,
    _SPGEMM_BUCKET_MEDIUM: _SPGEMM_COUNT_MAX_EXPANDED,
    _SPGEMM_BUCKET_LONG: 300_000,
}
_SPGEMM_BUCKET_FILL_BUDGETS = {
    _SPGEMM_BUCKET_SHORT: 2_400_000,
    _SPGEMM_BUCKET_MEDIUM: _SPGEMM_FILL_MAX_EXPANDED,
    _SPGEMM_BUCKET_LONG: 200_000,
}
_SPGEMM_BUCKET_MAX_ROWS = {
    _SPGEMM_BUCKET_SHORT: 8192,
    _SPGEMM_BUCKET_MEDIUM: _SPGEMM_MAX_ROWS_PER_CHUNK,
    _SPGEMM_BUCKET_LONG: 256,
}
_SPGEMM_LONG_ROW_SLICE_EXPANDED = 200_000


class SpGEMMPrepared:
    """Prepared CSR metadata for repeated SpGEMM runs."""

    __slots__ = (
        "a_data",
        "a_indices",
        "a_indptr",
        "a_shape",
        "b_data",
        "b_indices",
        "b_indptr",
        "b_shape",
        "n_rows",
        "n_inner",
        "n_cols",
        "a_row_work",
        "row_bucket",
        "row_work_ready",
        "bucket_rows",
        "count_chunks",
        "fill_chunks",
        "count_chunks_by_bucket",
        "fill_chunks_by_bucket",
        "long_row_slice_expanded",
        "long_row_slices_host",
        "hash_capacity_hint",
        "block_nnz",
    )

    def __init__(
        self,
        a_data,
        a_indices,
        a_indptr,
        a_shape,
        b_data,
        b_indices,
        b_indptr,
        b_shape,
        a_row_work,
        row_bucket,
        row_work_ready,
        bucket_rows,
        count_chunks,
        fill_chunks,
        count_chunks_by_bucket,
        fill_chunks_by_bucket,
        long_row_slice_expanded,
        long_row_slices_host,
        hash_capacity_hint,
        block_nnz,
    ):
        self.a_data = a_data
        self.a_indices = a_indices
        self.a_indptr = a_indptr
        self.a_shape = (int(a_shape[0]), int(a_shape[1]))
        self.b_data = b_data
        self.b_indices = b_indices
        self.b_indptr = b_indptr
        self.b_shape = (int(b_shape[0]), int(b_shape[1]))
        self.n_rows = self.a_shape[0]
        self.n_inner = self.a_shape[1]
        self.n_cols = self.b_shape[1]
        self.a_row_work = a_row_work
        self.row_bucket = row_bucket
        self.row_work_ready = bool(row_work_ready)
        self.bucket_rows = bucket_rows
        self.count_chunks = count_chunks
        self.fill_chunks = fill_chunks
        self.count_chunks_by_bucket = count_chunks_by_bucket
        self.fill_chunks_by_bucket = fill_chunks_by_bucket
        self.long_row_slice_expanded = int(long_row_slice_expanded)
        self.long_row_slices_host = long_row_slices_host
        self.hash_capacity_hint = int(hash_capacity_hint)
        self.block_nnz = int(block_nnz)


def _validate_csr(data, indices, indptr, shape, tag):
    if len(shape) != 2:
        raise ValueError(f"{tag}_shape must be a 2-tuple")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError(f"{tag} data/indices/indptr must be 1D tensors")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows < 0 or n_cols < 0:
        raise ValueError(f"{tag}_shape dimensions must be non-negative")
    if indptr.numel() != n_rows + 1:
        raise ValueError(
            f"{tag}_indptr length must be n_rows+1={n_rows + 1}, got {indptr.numel()}"
        )
    if data.numel() != indices.numel():
        raise ValueError(f"{tag}_data and {tag}_indices must have the same length")
    if not data.is_cuda or not indices.is_cuda or not indptr.is_cuda:
        raise ValueError(f"{tag} tensors must be CUDA tensors")
    if data.dtype not in SUPPORTED_SPGEMM_VALUE_DTYPES:
        raise TypeError(f"{tag}_data dtype must be torch.float32 or torch.float64")
    if indices.dtype != torch.int32:
        raise TypeError(f"{tag}_indices dtype must be torch.int32")
    if indptr.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{tag}_indptr dtype must be torch.int32 or torch.int64")

    nnz = int(data.numel())
    indptr_i64 = indptr.to(torch.int64)
    if indptr_i64.numel() > 0 and int(indptr_i64[0].item()) != 0:
        raise ValueError(f"{tag}_indptr[0] must be 0")
    if indptr_i64.numel() > 0 and int(indptr_i64[-1].item()) != nnz:
        raise ValueError(f"{tag}_indptr[-1] must equal nnz={nnz}")
    if indptr_i64.numel() > 1 and bool(torch.any(indptr_i64[1:] < indptr_i64[:-1]).item()):
        raise ValueError(f"{tag}_indptr must be nondecreasing")
    if nnz > 0:
        min_col = int(indices.min().item())
        max_col = int(indices.max().item())
        if min_col < 0 or max_col >= n_cols:
            raise IndexError(f"{tag}_indices out of range for n_cols={n_cols}")
    return n_rows, n_cols, indptr_i64


def _prepare_spgemm_csr_inputs(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
):
    a_rows, a_cols, a_indptr64 = _validate_csr(a_data, a_indices, a_indptr, a_shape, "a")
    b_rows, b_cols, b_indptr64 = _validate_csr(b_data, b_indices, b_indptr, b_shape, "b")
    if a_cols != b_rows:
        raise ValueError(
            f"shape mismatch for A@B: A is {a_rows}x{a_cols}, B is {b_rows}x{b_cols}"
        )
    if a_data.device != b_data.device:
        raise ValueError("A and B tensors must be on the same CUDA device")
    if a_data.dtype != b_data.dtype:
        raise TypeError("A and B value dtype must match")

    a_data = a_data.contiguous()
    a_indices = a_indices.contiguous()
    a_indptr64 = a_indptr64.contiguous()
    b_data = b_data.contiguous()
    b_indices = b_indices.contiguous()
    b_indptr64 = b_indptr64.contiguous()
    return (
        a_data,
        a_indices,
        a_indptr64,
        (a_rows, a_cols),
        b_data,
        b_indices,
        b_indptr64,
        (b_rows, b_cols),
    )


@triton.jit
def _spgemm_row_work_kernel(
    a_indptr_ptr,
    a_indices_ptr,
    b_indptr_ptr,
    row_work_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    start = tl.load(a_indptr_ptr + row)
    end = tl.load(a_indptr_ptr + row + 1)
    row_nnz = end - start
    acc = tl.zeros((), dtype=tl.int32)
    for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
        offs = start + chunk_start + tl.arange(0, BLOCK_NNZ)
        mask = offs < end
        k = tl.load(a_indices_ptr + offs, mask=mask, other=0)
        b_start = tl.load(b_indptr_ptr + k, mask=mask, other=0)
        b_end = tl.load(b_indptr_ptr + k + 1, mask=mask, other=0)
        contrib = (b_end - b_start).to(tl.int32)
        acc += tl.sum(tl.where(mask, contrib, 0))
    tl.store(row_work_ptr + row, acc)


def _estimate_hash_capacity(a_row_work):
    if a_row_work.numel() == 0:
        return 256
    p95 = int(torch.quantile(a_row_work.to(torch.float32), 0.95).item())
    p95 = max(p95, 1)
    cap = 1
    while cap < p95:
        cap <<= 1
    return max(256, cap)


def _build_row_bucket(a_row_work):
    # 0: small, 1: medium, 2: long
    bucket = torch.zeros_like(a_row_work, dtype=torch.int8)
    bucket = torch.where(
        a_row_work > 4096,
        torch.full_like(bucket, _SPGEMM_BUCKET_LONG),
        bucket,
    )
    bucket = torch.where(
        (a_row_work > 256) & (a_row_work <= 4096),
        torch.full_like(bucket, _SPGEMM_BUCKET_MEDIUM),
        bucket,
    )
    return bucket


def _build_long_row_slices_host(a_indptr, a_indices, b_indptr, row_ids, max_expanded):
    if row_ids is None or row_ids.numel() == 0:
        return {}

    out = {}
    row_list = row_ids.to(torch.int64).cpu().tolist()
    a_indptr_cpu = a_indptr.to(torch.int64).cpu()
    for row in row_list:
        start = int(a_indptr_cpu[row].item())
        end = int(a_indptr_cpu[row + 1].item())
        if end <= start:
            out[int(row)] = []
            continue
        a_cols = a_indices[start:end].to(torch.int64)
        b_counts = (b_indptr[a_cols + 1] - b_indptr[a_cols]).to(torch.int64)
        counts_host = b_counts.cpu().tolist()
        slices = []
        idx = 0
        total = len(counts_host)
        while idx < total:
            seg_start = idx
            acc = 0
            while idx < total:
                w = int(counts_host[idx])
                if idx > seg_start and acc + w > int(max_expanded):
                    break
                acc += w
                idx += 1
                if idx == seg_start + 1 and w > int(max_expanded):
                    break
            slices.append((start + seg_start, start + idx))
        out[int(row)] = slices
    return out


def prepare_spgemm_csr(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
    block_nnz=256,
    analyze_rows=True,
):
    if block_nnz <= 0:
        raise ValueError("block_nnz must be positive")
    (
        a_data,
        a_indices,
        a_indptr,
        a_shape,
        b_data,
        b_indices,
        b_indptr,
        b_shape,
    ) = _prepare_spgemm_csr_inputs(
        a_data,
        a_indices,
        a_indptr,
        a_shape,
        b_data,
        b_indices,
        b_indptr,
        b_shape,
    )
    n_rows = int(a_shape[0])
    if n_rows == 0:
        row_work = torch.empty(0, dtype=torch.int32, device=a_data.device)
        row_bucket = torch.empty(0, dtype=torch.int8, device=a_data.device)
        hash_capacity_hint = 256
        row_work_ready = True
    elif analyze_rows:
        row_work = torch.empty(n_rows, dtype=torch.int32, device=a_data.device)
        try:
            _spgemm_row_work_kernel[(n_rows,)](
                a_indptr,
                a_indices,
                b_indptr,
                row_work,
                n_rows,
                BLOCK_NNZ=int(block_nnz),
            )
        except Exception:
            b_row_nnz = (b_indptr[1:] - b_indptr[:-1]).to(torch.int32)
            for row in range(n_rows):
                start = int(a_indptr[row].item())
                end = int(a_indptr[row + 1].item())
                if end <= start:
                    row_work[row] = 0
                    continue
                cols = a_indices[start:end].to(torch.int64)
                row_work[row] = torch.sum(b_row_nnz[cols]).to(torch.int32)
        row_bucket = _build_row_bucket(row_work)
        hash_capacity_hint = 256
        row_work_ready = True
    else:
        row_work = torch.zeros(n_rows, dtype=torch.int32, device=a_data.device)
        row_bucket = torch.zeros(n_rows, dtype=torch.int8, device=a_data.device)
        hash_capacity_hint = 256
        row_work_ready = False
    long_row_slices_host = {}
    if row_work_ready:
        long_rows = torch.nonzero(row_bucket == _SPGEMM_BUCKET_LONG, as_tuple=False).flatten()
        long_row_slices_host = _build_long_row_slices_host(
            a_indptr,
            a_indices,
            b_indptr,
            long_rows,
            _SPGEMM_LONG_ROW_SLICE_EXPANDED,
        )
    return SpGEMMPrepared(
        a_data=a_data,
        a_indices=a_indices,
        a_indptr=a_indptr,
        a_shape=a_shape,
        b_data=b_data,
        b_indices=b_indices,
        b_indptr=b_indptr,
        b_shape=b_shape,
        a_row_work=row_work,
        row_bucket=row_bucket,
        row_work_ready=row_work_ready,
        bucket_rows=None,
        count_chunks=None,
        fill_chunks=None,
        count_chunks_by_bucket=None,
        fill_chunks_by_bucket=None,
        long_row_slice_expanded=_SPGEMM_LONG_ROW_SLICE_EXPANDED,
        long_row_slices_host=long_row_slices_host,
        hash_capacity_hint=hash_capacity_hint,
        block_nnz=block_nnz,
    )


def _ensure_row_work(prepared):
    if prepared.row_work_ready:
        return
    if prepared.a_data.numel() == 0 or prepared.b_data.numel() == 0:
        prepared.a_row_work = torch.zeros(prepared.n_rows, dtype=torch.int32, device=prepared.a_data.device)
        prepared.row_bucket = torch.zeros(prepared.n_rows, dtype=torch.int8, device=prepared.a_data.device)
        prepared.hash_capacity_hint = 256
        prepared.row_work_ready = True
        prepared.long_row_slices_host = {}
        _clear_runtime_schedules(prepared)
        return
    row_work = torch.empty(prepared.n_rows, dtype=torch.int32, device=prepared.a_data.device)
    _spgemm_row_work_kernel[(prepared.n_rows,)](
        prepared.a_indptr,
        prepared.a_indices,
        prepared.b_indptr,
        row_work,
        prepared.n_rows,
        BLOCK_NNZ=int(prepared.block_nnz),
    )
    prepared.a_row_work = row_work
    prepared.row_bucket = _build_row_bucket(row_work)
    long_rows = torch.nonzero(prepared.row_bucket == _SPGEMM_BUCKET_LONG, as_tuple=False).flatten()
    prepared.long_row_slices_host = _build_long_row_slices_host(
        prepared.a_indptr,
        prepared.a_indices,
        prepared.b_indptr,
        long_rows,
        prepared.long_row_slice_expanded,
    )
    prepared.hash_capacity_hint = 256
    prepared.row_work_ready = True
    _clear_runtime_schedules(prepared)


def _build_row_chunks(row_work, max_expanded, max_rows_per_chunk):
    n_rows = int(row_work.numel())
    if n_rows == 0:
        return []
    all_rows = torch.arange(n_rows, device=row_work.device, dtype=torch.int64)
    return _build_row_id_chunks(row_work, all_rows, max_expanded, max_rows_per_chunk)


def _build_bucket_rows(row_bucket, device):
    out = {}
    for bucket_id in _SPGEMM_BUCKET_ORDER:
        rows = torch.nonzero(row_bucket == bucket_id, as_tuple=False).flatten()
        out[bucket_id] = rows.to(device=device, dtype=torch.int64)
    return out


def _build_row_id_chunks(row_work, row_ids, max_expanded, max_rows_per_chunk):
    if row_ids.numel() == 0:
        return []
    work_host = row_work[row_ids].detach().to("cpu", dtype=torch.int64).tolist()
    chunks = []
    idx = 0
    total_rows = len(work_host)
    while idx < total_rows:
        start_idx = idx
        acc = 0
        taken = 0
        while idx < total_rows and taken < int(max_rows_per_chunk):
            w = int(work_host[idx])
            if taken > 0 and acc + w > int(max_expanded):
                break
            acc += w
            idx += 1
            taken += 1
            if taken == 1 and w > int(max_expanded):
                break
        if idx == start_idx:
            idx += 1
        chunks.append(row_ids[start_idx:idx].contiguous())
    return chunks


def _compose_ordered_chunks(chunks_by_bucket):
    ordered = []
    for bucket_id in _SPGEMM_BUCKET_ORDER:
        ordered.extend(chunks_by_bucket.get(bucket_id, []))
    return ordered


def _clear_runtime_schedules(prepared):
    prepared.bucket_rows = None
    prepared.count_chunks = None
    prepared.fill_chunks = None
    prepared.count_chunks_by_bucket = None
    prepared.fill_chunks_by_bucket = None


def _clear_count_schedule(prepared):
    prepared.count_chunks = None
    prepared.count_chunks_by_bucket = None


def _clear_fill_schedule(prepared):
    prepared.fill_chunks = None
    prepared.fill_chunks_by_bucket = None


def _ensure_bucket_rows(prepared):
    if prepared.bucket_rows is None:
        prepared.bucket_rows = _build_bucket_rows(prepared.row_bucket, prepared.a_data.device)
    return prepared.bucket_rows


def _chunk_rows_for_bucket(prepared, bucket_id, max_expanded):
    rows = _ensure_bucket_rows(prepared)[bucket_id]
    chunks = _build_row_id_chunks(
        prepared.a_row_work,
        rows,
        max_expanded=max_expanded,
        max_rows_per_chunk=_SPGEMM_BUCKET_MAX_ROWS[bucket_id],
    )
    if bucket_id == _SPGEMM_BUCKET_LONG:
        return [chunk.to(torch.int64).cpu().tolist() for chunk in chunks]
    return chunks


def _ensure_count_chunks(prepared):
    if prepared.count_chunks_by_bucket is not None:
        return prepared.count_chunks_by_bucket
    prepared.count_chunks_by_bucket = {}
    for bucket_id in _SPGEMM_BUCKET_ORDER:
        prepared.count_chunks_by_bucket[bucket_id] = _chunk_rows_for_bucket(
            prepared,
            bucket_id,
            _SPGEMM_BUCKET_COUNT_BUDGETS[bucket_id],
        )
    prepared.count_chunks = _compose_ordered_chunks(prepared.count_chunks_by_bucket)
    return prepared.count_chunks_by_bucket


def _ensure_fill_chunks(prepared):
    if prepared.fill_chunks_by_bucket is not None:
        return prepared.fill_chunks_by_bucket
    prepared.fill_chunks_by_bucket = {}
    for bucket_id in _SPGEMM_BUCKET_ORDER:
        prepared.fill_chunks_by_bucket[bucket_id] = _chunk_rows_for_bucket(
            prepared,
            bucket_id,
            _SPGEMM_BUCKET_FILL_BUDGETS[bucket_id],
        )
    prepared.fill_chunks = _compose_ordered_chunks(prepared.fill_chunks_by_bucket)
    return prepared.fill_chunks_by_bucket


def _expand_rows_contrib(prepared, row_ids, need_values):
    device = prepared.a_data.device
    if row_ids.numel() == 0:
        empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
        if need_values:
            empty_v = torch.empty(0, dtype=prepared.a_data.dtype, device=device)
            return empty_i64, empty_v
        return empty_i64, None

    row_ids = row_ids.to(torch.int64)
    row_nnz = (prepared.a_indptr[row_ids + 1] - prepared.a_indptr[row_ids]).to(torch.int64)
    total_a = int(row_nnz.sum().item())
    if total_a == 0:
        empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
        if need_values:
            empty_v = torch.empty(0, dtype=prepared.a_data.dtype, device=device)
            return empty_i64, empty_v
        return empty_i64, None

    owner = torch.repeat_interleave(
        torch.arange(row_ids.numel(), device=device, dtype=torch.int64),
        row_nnz,
    )
    prefix = torch.cumsum(row_nnz, dim=0)
    base = prefix - row_nnz
    intra = (
        torch.arange(total_a, device=device, dtype=torch.int64)
        - torch.repeat_interleave(base, row_nnz)
    )
    row_starts = prepared.a_indptr[row_ids]
    a_pos = row_starts[owner] + intra
    rows = row_ids[owner]
    a_cols = prepared.a_indices[a_pos].to(torch.int64)
    b_starts = prepared.b_indptr[a_cols]
    b_ends = prepared.b_indptr[a_cols + 1]
    b_counts = b_ends - b_starts
    total = int(b_counts.sum().item())
    if total == 0:
        empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
        if need_values:
            empty_v = torch.empty(0, dtype=prepared.a_data.dtype, device=device)
            return empty_i64, empty_v
        return empty_i64, None

    b_owner = torch.repeat_interleave(
        torch.arange(a_cols.numel(), device=device, dtype=torch.int64),
        b_counts,
    )
    rows_expanded = rows[b_owner]
    prefix = torch.cumsum(b_counts, dim=0)
    base = prefix - b_counts
    starts_rep = torch.repeat_interleave(b_starts, b_counts)
    intra = (
        torch.arange(total, device=device, dtype=torch.int64)
        - torch.repeat_interleave(base, b_counts)
    )
    b_pos = starts_rep + intra
    cols = prepared.b_indices[b_pos].to(torch.int64)
    keys = rows_expanded * max(1, prepared.n_cols) + cols
    if not need_values:
        return keys, None

    a_vals = prepared.a_data[a_pos]
    vals = a_vals[b_owner] * prepared.b_data[b_pos]
    return keys, vals


def _expand_single_row_slice_contrib(prepared, row, a_ptr_start, a_ptr_end, need_values):
    device = prepared.a_data.device
    if a_ptr_end <= a_ptr_start:
        empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
        if need_values:
            empty_v = torch.empty(0, dtype=prepared.a_data.dtype, device=device)
            return empty_i64, empty_v
        return empty_i64, None

    a_pos = torch.arange(a_ptr_start, a_ptr_end, device=device, dtype=torch.int64)
    a_cols = prepared.a_indices[a_pos].to(torch.int64)
    b_starts = prepared.b_indptr[a_cols]
    b_ends = prepared.b_indptr[a_cols + 1]
    b_counts = b_ends - b_starts
    total = int(b_counts.sum().item())
    if total == 0:
        empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
        if need_values:
            empty_v = torch.empty(0, dtype=prepared.a_data.dtype, device=device)
            return empty_i64, empty_v
        return empty_i64, None

    owner = torch.repeat_interleave(
        torch.arange(a_pos.numel(), device=device, dtype=torch.int64),
        b_counts,
    )
    prefix = torch.cumsum(b_counts, dim=0)
    base = prefix - b_counts
    starts_rep = torch.repeat_interleave(b_starts, b_counts)
    intra = (
        torch.arange(total, device=device, dtype=torch.int64)
        - torch.repeat_interleave(base, b_counts)
    )
    b_pos = starts_rep + intra
    cols = prepared.b_indices[b_pos].to(torch.int64)
    keys = int(row) * max(1, prepared.n_cols) + cols
    if not need_values:
        return keys, None
    a_vals = prepared.a_data[a_pos]
    vals = a_vals[owner] * prepared.b_data[b_pos]
    return keys, vals


def _iter_row_a_slices(prepared, row, max_expanded):
    cached = prepared.long_row_slices_host.get(int(row)) if prepared.long_row_slices_host is not None else None
    if cached is not None:
        return cached
    start = int(prepared.a_indptr[row].item())
    end = int(prepared.a_indptr[row + 1].item())
    if end <= start:
        if prepared.long_row_slices_host is not None:
            prepared.long_row_slices_host[int(row)] = []
        return []
    a_cols = prepared.a_indices[start:end].to(torch.int64)
    b_counts = (prepared.b_indptr[a_cols + 1] - prepared.b_indptr[a_cols]).to(torch.int64)
    counts_host = b_counts.cpu().tolist()
    slices = []
    idx = 0
    total = len(counts_host)
    while idx < total:
        seg_start = idx
        acc = 0
        while idx < total:
            w = int(counts_host[idx])
            if idx > seg_start and acc + w > int(max_expanded):
                break
            acc += w
            idx += 1
            if idx == seg_start + 1 and w > int(max_expanded):
                break
        slices.append((start + seg_start, start + idx))
    if prepared.long_row_slices_host is not None:
        prepared.long_row_slices_host[int(row)] = slices
    return slices


def _reduce_sorted_keys_vals(keys_sorted, vals_sorted, out_dtype):
    if keys_sorted.numel() == 0:
        return keys_sorted, vals_sorted
    uniq_keys, counts = torch.unique_consecutive(keys_sorted, return_counts=True)
    if uniq_keys.numel() == keys_sorted.numel():
        return uniq_keys, vals_sorted.to(out_dtype)
    acc_dtype = torch.float64 if out_dtype == torch.float32 else vals_sorted.dtype
    vals_acc = vals_sorted.to(acc_dtype)
    prefix = torch.cumsum(vals_acc, dim=0)
    end_idx = torch.cumsum(counts.to(torch.int64), dim=0) - 1
    seg_end = prefix[end_idx]
    seg_begin = torch.zeros_like(seg_end)
    if seg_end.numel() > 1:
        seg_begin[1:] = prefix[end_idx[:-1]]
    uniq_vals = (seg_end - seg_begin).to(out_dtype)
    return uniq_keys, uniq_vals


def _sort_reduce_pairs(keys, vals, out_dtype):
    if keys.numel() == 0:
        return keys, vals
    order = torch.argsort(keys)
    keys_sorted = keys[order]
    vals_sorted = vals[order]
    return _reduce_sorted_keys_vals(keys_sorted, vals_sorted, out_dtype)


def _spgemm_count_phase(prepared, profile=False):
    n_rows = prepared.n_rows
    device = prepared.a_data.device
    row_nnz_c = torch.zeros(n_rows, dtype=torch.int64, device=device)
    bucket_ms = {bucket_id: None for bucket_id in _SPGEMM_BUCKET_ORDER}
    long_row_sliced = 0
    chunks_by_bucket = _ensure_count_chunks(prepared)
    for bucket_id in _SPGEMM_BUCKET_ORDER:
        chunks = chunks_by_bucket.get(bucket_id, [])
        for row_ids in chunks:
            if profile:
                torch.cuda.synchronize()
                t_bucket0 = time.perf_counter()
            if bucket_id != _SPGEMM_BUCKET_LONG:
                keys, _ = _expand_rows_contrib(prepared, row_ids, need_values=False)
                if keys.numel() > 0:
                    uniq_keys = torch.unique(keys, sorted=True)
                    uniq_rows = torch.div(
                        uniq_keys,
                        max(1, prepared.n_cols),
                        rounding_mode="floor",
                    )
                    rows_unique, counts = torch.unique_consecutive(
                        uniq_rows, return_counts=True
                    )
                    row_nnz_c[rows_unique] = counts.to(torch.int64)
            else:
                for row in row_ids:
                    slices = _iter_row_a_slices(
                        prepared,
                        int(row),
                        max_expanded=prepared.long_row_slice_expanded,
                    )
                    if len(slices) > 1:
                        long_row_sliced += 1
                    keys_parts = []
                    for a_start, a_end in slices:
                        keys, _ = _expand_single_row_slice_contrib(
                            prepared,
                            int(row),
                            a_start,
                            a_end,
                            need_values=False,
                        )
                        if keys.numel() == 0:
                            continue
                        keys_parts.append(torch.unique(keys, sorted=True))
                    if not keys_parts:
                        row_nnz_c[int(row)] = 0
                        continue
                    uniq_row_keys = torch.unique(torch.cat(keys_parts), sorted=True)
                    row_nnz_c[int(row)] = int(uniq_row_keys.numel())
            if profile:
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t_bucket0) * 1000.0
                bucket_ms[bucket_id] = (
                    elapsed if bucket_ms[bucket_id] is None else bucket_ms[bucket_id] + elapsed
                )
    meta = {
        "bucket_count_ms_short": bucket_ms[_SPGEMM_BUCKET_SHORT],
        "bucket_count_ms_medium": bucket_ms[_SPGEMM_BUCKET_MEDIUM],
        "bucket_count_ms_long": bucket_ms[_SPGEMM_BUCKET_LONG],
        "long_row_sliced_count_count": int(long_row_sliced),
    }
    return row_nnz_c, meta


def _spgemm_fill_phase(prepared, c_indptr, out_data=None, out_indices=None, profile=False):
    nnz_c = int(c_indptr[-1].item())
    device = prepared.a_data.device
    c_data = (
        out_data
        if out_data is not None
        else torch.empty(nnz_c, dtype=prepared.a_data.dtype, device=device)
    )
    c_indices = (
        out_indices
        if out_indices is not None
        else torch.empty(nnz_c, dtype=torch.int32, device=device)
    )
    bucket_ms = {bucket_id: None for bucket_id in _SPGEMM_BUCKET_ORDER}
    long_row_sliced = 0
    if nnz_c == 0:
        meta = {
            "bucket_fill_ms_short": bucket_ms[_SPGEMM_BUCKET_SHORT],
            "bucket_fill_ms_medium": bucket_ms[_SPGEMM_BUCKET_MEDIUM],
            "bucket_fill_ms_long": bucket_ms[_SPGEMM_BUCKET_LONG],
            "long_row_sliced_count_fill": int(long_row_sliced),
        }
        return c_data, c_indices, meta

    chunks_by_bucket = _ensure_fill_chunks(prepared)
    for bucket_id in _SPGEMM_BUCKET_ORDER:
        chunks = chunks_by_bucket.get(bucket_id, [])
        for row_ids in chunks:
            if profile:
                torch.cuda.synchronize()
                t_bucket0 = time.perf_counter()
            if bucket_id != _SPGEMM_BUCKET_LONG:
                keys, vals = _expand_rows_contrib(prepared, row_ids, need_values=True)
                if keys.numel() > 0:
                    uniq_keys, uniq_vals = _sort_reduce_pairs(
                        keys,
                        vals,
                        out_dtype=prepared.a_data.dtype,
                    )
                    uniq_rows = torch.div(
                        uniq_keys,
                        max(1, prepared.n_cols),
                        rounding_mode="floor",
                    )
                    uniq_cols = (uniq_keys - uniq_rows * max(1, prepared.n_cols)).to(torch.int32)
                    _, row_counts = torch.unique_consecutive(uniq_rows, return_counts=True)
                    row_offsets = torch.cumsum(row_counts.to(torch.int64), dim=0) - row_counts.to(torch.int64)
                    local_pos = (
                        torch.arange(uniq_keys.numel(), device=device, dtype=torch.int64)
                        - torch.repeat_interleave(row_offsets, row_counts)
                    )
                    dst = c_indptr[uniq_rows] + local_pos
                    c_indices[dst] = uniq_cols
                    c_data[dst] = uniq_vals
            else:
                for row in row_ids:
                    row = int(row)
                    slices = _iter_row_a_slices(
                        prepared,
                        row,
                        max_expanded=prepared.long_row_slice_expanded,
                    )
                    if len(slices) > 1:
                        long_row_sliced += 1
                    key_parts = []
                    val_parts = []
                    for a_start, a_end in slices:
                        keys, vals = _expand_single_row_slice_contrib(
                            prepared,
                            row,
                            a_start,
                            a_end,
                            need_values=True,
                        )
                        if keys.numel() == 0:
                            continue
                        uniq_k, uniq_v = _sort_reduce_pairs(
                            keys,
                            vals,
                            out_dtype=prepared.a_data.dtype,
                        )
                        key_parts.append(uniq_k)
                        val_parts.append(uniq_v)
                    row_start = int(c_indptr[row].item())
                    row_end = int(c_indptr[row + 1].item())
                    row_nnz = row_end - row_start
                    if row_nnz == 0:
                        continue
                    if not key_parts:
                        raise RuntimeError(f"row {row} expected nnz={row_nnz} but got empty fill")
                    row_keys = torch.cat(key_parts)
                    row_vals = torch.cat(val_parts)
                    row_keys, row_vals = _sort_reduce_pairs(
                        row_keys,
                        row_vals,
                        out_dtype=prepared.a_data.dtype,
                    )
                    if row_keys.numel() != row_nnz:
                        raise RuntimeError(
                            f"row {row} fill nnz mismatch: expected {row_nnz}, got {row_keys.numel()}"
                        )
                    row_cols = (row_keys - row * max(1, prepared.n_cols)).to(torch.int32)
                    c_indices[row_start:row_end] = row_cols
                    c_data[row_start:row_end] = row_vals
            if profile:
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t_bucket0) * 1000.0
                bucket_ms[bucket_id] = (
                    elapsed if bucket_ms[bucket_id] is None else bucket_ms[bucket_id] + elapsed
                )

    meta = {
        "bucket_fill_ms_short": bucket_ms[_SPGEMM_BUCKET_SHORT],
        "bucket_fill_ms_medium": bucket_ms[_SPGEMM_BUCKET_MEDIUM],
        "bucket_fill_ms_long": bucket_ms[_SPGEMM_BUCKET_LONG],
        "long_row_sliced_count_fill": int(long_row_sliced),
    }
    return c_data, c_indices, meta


def _run_spgemm_prepared(prepared, out=None, profile=False, measure_stage=False):
    _ensure_row_work(prepared)
    if out is not None:
        if not isinstance(out, (tuple, list)) or len(out) != 3:
            raise TypeError("out must be a tuple/list of (data, indices, indptr)")
        out_data, out_indices, out_indptr = out
        if not out_data.is_cuda or not out_indices.is_cuda or not out_indptr.is_cuda:
            raise ValueError("out data/indices/indptr must be CUDA tensors")
        if out_data.device != prepared.a_data.device or out_indices.device != prepared.a_data.device or out_indptr.device != prepared.a_data.device:
            raise ValueError("out data/indices/indptr must be on the same CUDA device as computed C")
        if out_indptr.shape != (prepared.n_rows + 1,) or out_indptr.dtype != torch.int64:
            raise ValueError("out indptr shape/dtype must match computed C indptr")
    else:
        out_data = out_indices = out_indptr = None

    if measure_stage:
        torch.cuda.synchronize()
        t_count0 = time.perf_counter()
    row_nnz_c, count_meta = _spgemm_count_phase(prepared, profile=profile)
    if measure_stage:
        torch.cuda.synchronize()
        count_ms = (time.perf_counter() - t_count0) * 1000.0
    else:
        count_ms = None
    _clear_count_schedule(prepared)

    c_indptr = out_indptr
    if c_indptr is None:
        c_indptr = torch.empty(prepared.n_rows + 1, dtype=torch.int64, device=prepared.a_data.device)
    c_indptr[0] = 0
    if prepared.n_rows > 0:
        c_indptr[1:] = torch.cumsum(row_nnz_c, dim=0)
    nnz_c = int(c_indptr[-1].item()) if c_indptr.numel() > 0 else 0

    if out_data is not None:
        if out_data.shape != (nnz_c,) or out_data.dtype != prepared.a_data.dtype:
            raise ValueError("out data shape/dtype must match computed C data")
        if out_indices.shape != (nnz_c,) or out_indices.dtype != torch.int32:
            raise ValueError("out indices shape/dtype must match computed C indices")

    if measure_stage:
        torch.cuda.synchronize()
        t_fill0 = time.perf_counter()
    c_data, c_indices, fill_meta = _spgemm_fill_phase(
        prepared,
        c_indptr,
        out_data=out_data,
        out_indices=out_indices,
        profile=profile,
    )
    if measure_stage:
        torch.cuda.synchronize()
        fill_ms = (time.perf_counter() - t_fill0) * 1000.0
    else:
        fill_ms = None
    _clear_fill_schedule(prepared)

    def _sum_bucket_ms(count_key, fill_key):
        count_val = count_meta[count_key]
        fill_val = fill_meta[fill_key]
        if count_val is None or fill_val is None:
            return None
        return float(count_val + fill_val)

    return c_data, c_indices, c_indptr, {
        "count_ms": count_ms,
        "fill_ms": fill_ms,
        "bucket_ms_short": _sum_bucket_ms("bucket_count_ms_short", "bucket_fill_ms_short"),
        "bucket_ms_medium": _sum_bucket_ms("bucket_count_ms_medium", "bucket_fill_ms_medium"),
        "bucket_ms_long": _sum_bucket_ms("bucket_count_ms_long", "bucket_fill_ms_long"),
        "bucket_count_ms_short": count_meta["bucket_count_ms_short"],
        "bucket_count_ms_medium": count_meta["bucket_count_ms_medium"],
        "bucket_count_ms_long": count_meta["bucket_count_ms_long"],
        "bucket_fill_ms_short": fill_meta["bucket_fill_ms_short"],
        "bucket_fill_ms_medium": fill_meta["bucket_fill_ms_medium"],
        "bucket_fill_ms_long": fill_meta["bucket_fill_ms_long"],
        "bucket_nrows_short": int(torch.count_nonzero(prepared.row_bucket == _SPGEMM_BUCKET_SHORT).item()),
        "bucket_nrows_medium": int(torch.count_nonzero(prepared.row_bucket == _SPGEMM_BUCKET_MEDIUM).item()),
        "bucket_nrows_long": int(torch.count_nonzero(prepared.row_bucket == _SPGEMM_BUCKET_LONG).item()),
        "long_row_sliced_count": int(
            max(
                count_meta["long_row_sliced_count_count"],
                fill_meta["long_row_sliced_count_fill"],
            )
        ),
    }


def flagsparse_spgemm_csr(
    a_data=None,
    a_indices=None,
    a_indptr=None,
    a_shape=None,
    b_data=None,
    b_indices=None,
    b_indptr=None,
    b_shape=None,
    prepared=None,
    out=None,
    return_time=False,
    return_meta=False,
):
    """CSR SpGEMM: C = A @ B with CSR output (Triton-only main path)."""
    prepare_ms = 0.0
    internal_prepared = prepared is None
    if prepared is None:
        if any(
            x is None
            for x in (
                a_data,
                a_indices,
                a_indptr,
                a_shape,
                b_data,
                b_indices,
                b_indptr,
                b_shape,
            )
        ):
            raise ValueError(
                "A/B CSR tensors and shapes are required when prepared is not provided"
            )
        if return_meta:
            torch.cuda.synchronize()
            t_prepare0 = time.perf_counter()
        prepared = prepare_spgemm_csr(
            a_data,
            a_indices,
            a_indptr,
            a_shape,
            b_data,
            b_indices,
            b_indptr,
            b_shape,
        )
        if return_meta:
            torch.cuda.synchronize()
            prepare_ms = (time.perf_counter() - t_prepare0) * 1000.0
    elif not isinstance(prepared, SpGEMMPrepared):
        raise TypeError("prepared must be a SpGEMMPrepared instance")

    elapsed_ms = None
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    c_data, c_indices, c_indptr, stage_meta = _run_spgemm_prepared(
        prepared,
        out=out,
        profile=bool(return_meta),
        measure_stage=bool(return_meta),
    )
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if internal_prepared:
        _clear_runtime_schedules(prepared)

    result = (c_data, c_indices, c_indptr, (prepared.n_rows, prepared.n_cols))
    if return_time and return_meta:
        meta = {
            "prepare_ms": prepare_ms,
            "count_ms": stage_meta["count_ms"],
            "fill_ms": stage_meta["fill_ms"],
            "triton_ms": elapsed_ms,
            "hash_capacity_hint": prepared.hash_capacity_hint,
            "bucket_ms_short": stage_meta["bucket_ms_short"],
            "bucket_ms_medium": stage_meta["bucket_ms_medium"],
            "bucket_ms_long": stage_meta["bucket_ms_long"],
            "bucket_count_ms_short": stage_meta["bucket_count_ms_short"],
            "bucket_count_ms_medium": stage_meta["bucket_count_ms_medium"],
            "bucket_count_ms_long": stage_meta["bucket_count_ms_long"],
            "bucket_fill_ms_short": stage_meta["bucket_fill_ms_short"],
            "bucket_fill_ms_medium": stage_meta["bucket_fill_ms_medium"],
            "bucket_fill_ms_long": stage_meta["bucket_fill_ms_long"],
            "bucket_nrows_short": stage_meta["bucket_nrows_short"],
            "bucket_nrows_medium": stage_meta["bucket_nrows_medium"],
            "bucket_nrows_long": stage_meta["bucket_nrows_long"],
            "long_row_sliced_count": stage_meta["long_row_sliced_count"],
        }
        return result, elapsed_ms, meta
    if return_time:
        return result, elapsed_ms
    if return_meta:
        meta = {
            "prepare_ms": prepare_ms,
            "count_ms": stage_meta["count_ms"],
            "fill_ms": stage_meta["fill_ms"],
            "hash_capacity_hint": prepared.hash_capacity_hint,
            "bucket_ms_short": stage_meta["bucket_ms_short"],
            "bucket_ms_medium": stage_meta["bucket_ms_medium"],
            "bucket_ms_long": stage_meta["bucket_ms_long"],
            "bucket_count_ms_short": stage_meta["bucket_count_ms_short"],
            "bucket_count_ms_medium": stage_meta["bucket_count_ms_medium"],
            "bucket_count_ms_long": stage_meta["bucket_count_ms_long"],
            "bucket_fill_ms_short": stage_meta["bucket_fill_ms_short"],
            "bucket_fill_ms_medium": stage_meta["bucket_fill_ms_medium"],
            "bucket_fill_ms_long": stage_meta["bucket_fill_ms_long"],
            "bucket_nrows_short": stage_meta["bucket_nrows_short"],
            "bucket_nrows_medium": stage_meta["bucket_nrows_medium"],
            "bucket_nrows_long": stage_meta["bucket_nrows_long"],
            "long_row_sliced_count": stage_meta["long_row_sliced_count"],
        }
        return result, meta
    return result


def _csr_to_sorted_pairs(data, indices, indptr, n_cols):
    n_rows = int(indptr.numel()) - 1
    row_counts = indptr[1:] - indptr[:-1]
    rows = torch.repeat_interleave(
        torch.arange(n_rows, device=data.device, dtype=torch.int64),
        row_counts,
    )
    cols = indices.to(torch.int64)
    keys = rows * max(1, int(n_cols)) + cols
    if keys.numel() == 0:
        return keys, data
    order = torch.argsort(keys)
    return keys[order], data[order]


def _spgemm_pairwise_summary(candidate, reference, value_dtype):
    c_data, c_indices, c_indptr, c_shape = candidate
    r_data, r_indices, r_indptr, r_shape = reference
    if c_shape != r_shape:
        return {
            "match": False,
            "max_abs_error": float("inf"),
            "max_relative_error": float("inf"),
            "status": f"shape mismatch {c_shape} vs {r_shape}",
        }
    c_keys, c_vals = _csr_to_sorted_pairs(c_data, c_indices, c_indptr, c_shape[1])
    r_keys, r_vals = _csr_to_sorted_pairs(r_data, r_indices, r_indptr, r_shape[1])
    if c_keys.numel() != r_keys.numel():
        return {
            "match": False,
            "max_abs_error": float("inf"),
            "max_relative_error": float("inf"),
            "status": f"nnz mismatch {c_keys.numel()} vs {r_keys.numel()}",
        }
    if c_keys.numel() > 0 and not torch.equal(c_keys, r_keys):
        return {
            "match": False,
            "max_abs_error": float("inf"),
            "max_relative_error": float("inf"),
            "status": "sparsity pattern mismatch",
        }
    if c_vals.numel() == 0:
        return {
            "match": True,
            "max_abs_error": 0.0,
            "max_relative_error": 0.0,
            "status": "ok",
        }
    abs_diff = torch.abs(c_vals - r_vals)
    max_abs = float(torch.max(abs_diff).item())
    ref_max = float(torch.max(torch.abs(r_vals)).item())
    max_rel = 0.0 if ref_max == 0.0 else max_abs / ref_max
    atol, rtol = _tolerance_for_dtype(value_dtype)
    match = bool(torch.allclose(c_vals, r_vals, atol=atol, rtol=rtol))
    return {
        "match": match,
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
        "status": "ok" if match else "value mismatch",
    }


def _to_torch_csr(data, indices, indptr, shape):
    return torch.sparse_csr_tensor(
        indptr.to(torch.int64),
        indices.to(torch.int64),
        data,
        size=shape,
        device=data.device,
    )


def _torch_sparse_to_csr(tensor):
    if tensor.layout == torch.sparse_csr:
        indptr = tensor.crow_indices().to(torch.int64).contiguous()
        indices = tensor.col_indices().to(torch.int32).contiguous()
        data = tensor.values().contiguous()
        shape = (int(tensor.shape[0]), int(tensor.shape[1]))
        return data, indices, indptr, shape
    if tensor.layout == torch.sparse_coo:
        t = tensor.coalesce()
        rows = t.indices()[0].to(torch.int64)
        cols = t.indices()[1].to(torch.int64)
        vals = t.values()
        n_rows, n_cols = int(t.shape[0]), int(t.shape[1])
        if rows.numel() == 0:
            return (
                torch.empty(0, dtype=vals.dtype, device=vals.device),
                torch.empty(0, dtype=torch.int32, device=vals.device),
                torch.zeros(n_rows + 1, dtype=torch.int64, device=vals.device),
                (n_rows, n_cols),
            )
        key = rows * max(1, n_cols) + cols
        order = torch.argsort(key)
        rows = rows[order]
        cols = cols[order]
        vals = vals[order]
        row_counts = torch.bincount(rows, minlength=n_rows)
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=vals.device)
        indptr[1:] = torch.cumsum(row_counts, dim=0)
        return vals, cols.to(torch.int32), indptr, (n_rows, n_cols)
    raise TypeError(f"Unsupported sparse layout: {tensor.layout}")


def _spgemm_csr_reference(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
    return_metadata=False,
):
    a_t = _to_torch_csr(a_data, a_indices, a_indptr, a_shape)
    b_t = _to_torch_csr(b_data, b_indices, b_indptr, b_shape)
    fallback_reason = None
    pytorch_sparse_format = "CSR"
    try:
        reference_sparse = torch.sparse.mm(a_t, b_t)
    except Exception as exc:
        fallback_reason = f"CSR fallback: {exc}"
        pytorch_sparse_format = "COO"
        reference_sparse = torch.sparse.mm(
            a_t.to_sparse_coo().coalesce(),
            b_t.to_sparse_coo().coalesce(),
        )
    reference = _torch_sparse_to_csr(reference_sparse)
    metadata = {
        "backend": "torch",
        "fallback_reason": fallback_reason,
        "pytorch_sparse_format": pytorch_sparse_format,
    }
    if return_metadata:
        return reference, metadata
    return reference


def _hipsparse_spgemm_algorithm():
    return _hipsparse_lookup(
        "hipsparseSpGEMMAlg_t",
        ("HIPSPARSE_SPGEMM_DEFAULT", "HIPSPARSE_SPGEMM_ALG_DEFAULT"),
    )


def _hipsparse_spgemm_csr_skip_reason(
    value_dtype,
    a_indices_dtype,
    a_indptr_dtype,
    b_indices_dtype=None,
    b_indptr_dtype=None,
):
    b_indices_dtype = a_indices_dtype if b_indices_dtype is None else b_indices_dtype
    b_indptr_dtype = a_indptr_dtype if b_indptr_dtype is None else b_indptr_dtype
    if not _is_rocm_runtime():
        return "hipSPARSE CSR SpGEMM reference requires a ROCm runtime"
    unavailable_reason = _hipsparse_unavailable_reason()
    if unavailable_reason is not None:
        return unavailable_reason
    required_symbols = (
        "hipsparseCreate",
        "hipsparseDestroy",
        "hipsparseCreateCsr",
        "hipsparseDestroySpMat",
        "hipsparseSpGEMM_createDescr",
        "hipsparseSpGEMM_destroyDescr",
        "hipsparseSpGEMM_workEstimation",
        "hipsparseSpGEMM_compute",
        "hipsparseSpGEMM_copy",
        "hipsparseSpMatGetSize",
        "hipsparseCsrSetPointers",
    )
    for symbol in required_symbols:
        if not hasattr(hipsparse, symbol):
            return f"hipSPARSE CSR SpGEMM direct API is unavailable: missing {symbol}"
    if value_dtype not in SUPPORTED_SPGEMM_VALUE_DTYPES:
        return f"hipSPARSE CSR SpGEMM has no supported value dtype mapping for {value_dtype}"
    try:
        _ = _hipsparse_value_type(value_dtype)
        _ = _hipsparse_scalar(value_dtype, 1.0, 0.0)
        _ = _hipsparse_scalar(value_dtype, 0.0, 0.0)
        _ = _hipsparse_index_type(
            a_indptr_dtype, "hipSPARSE CSR SpGEMM A row offsets"
        )
        _ = _hipsparse_index_type(
            a_indices_dtype, "hipSPARSE CSR SpGEMM A column indices"
        )
        _ = _hipsparse_index_type(
            b_indptr_dtype, "hipSPARSE CSR SpGEMM B row offsets"
        )
        _ = _hipsparse_index_type(
            b_indices_dtype, "hipSPARSE CSR SpGEMM B column indices"
        )
        _ = _hipsparse_lookup(
            "hipsparseOperation_t", ("HIPSPARSE_OPERATION_NON_TRANSPOSE",)
        )
        _ = _hipsparse_lookup(
            "hipsparseIndexBase_t", ("HIPSPARSE_INDEX_BASE_ZERO",)
        )
        _ = _hipsparse_spgemm_algorithm()
    except Exception as exc:
        return str(exc)
    return None


def _hipsparse_spgemm_create_descr():
    result = hipsparse.hipsparseSpGEMM_createDescr()
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            "hipsparseSpGEMM_createDescr wrapper returned an unexpected result"
        )
    status, descr = result
    _hip_check_result(status, "hipsparseSpGEMM_createDescr")
    return descr


def _hipsparse_create_csr_descriptor(
    spmat_ref,
    n_rows,
    n_cols,
    nnz,
    indptr,
    indices,
    data,
    row_index_type,
    col_index_type,
    index_base,
    value_type,
):
    _hip_check_result(
        hipsparse.hipsparseCreateCsr(
            spmat_ref,
            n_rows,
            n_cols,
            nnz,
            HipPointer.fromObj(indptr.data_ptr()),
            HipPointer.fromObj(indices.data_ptr()),
            HipPointer.fromObj(data.data_ptr()),
            row_index_type,
            col_index_type,
            index_base,
            value_type,
        ),
        "hipsparseCreateCsr",
    )


def _hipsparse_spgemm_work_estimation(
    handle,
    op_enum,
    alpha,
    mat_a,
    mat_b,
    beta,
    mat_c,
    value_type,
    alg,
    descr,
    workspace,
):
    size_out = ctypes.c_size_t()
    _hip_check_result(
        hipsparse.hipsparseSpGEMM_workEstimation(
            handle,
            op_enum,
            op_enum,
            alpha,
            mat_a,
            mat_b,
            beta,
            mat_c,
            value_type,
            alg,
            descr,
            size_out,
            workspace,
        ),
        "hipsparseSpGEMM_workEstimation",
    )
    return int(size_out.value)


def _hipsparse_spgemm_compute(
    handle,
    op_enum,
    alpha,
    mat_a,
    mat_b,
    beta,
    mat_c,
    value_type,
    alg,
    descr,
    workspace,
):
    size_out = ctypes.c_size_t()
    _hip_check_result(
        hipsparse.hipsparseSpGEMM_compute(
            handle,
            op_enum,
            op_enum,
            alpha,
            mat_a,
            mat_b,
            beta,
            mat_c,
            value_type,
            alg,
            descr,
            size_out,
            workspace,
        ),
        "hipsparseSpGEMM_compute",
    )
    return int(size_out.value)


def _prepare_spgemm_csr_ref_hipsparse(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
):
    skip_reason = _hipsparse_spgemm_csr_skip_reason(
        a_data.dtype,
        a_indices.dtype,
        a_indptr.dtype,
        b_indices.dtype,
        b_indptr.dtype,
    )
    if skip_reason is not None:
        raise RuntimeError(skip_reason)
    if not all(
        torch.is_tensor(t)
        for t in (a_data, a_indices, a_indptr, b_data, b_indices, b_indptr)
    ):
        raise TypeError("all CSR inputs must be torch.Tensor")
    if not all(t.is_cuda for t in (a_data, a_indices, a_indptr, b_data, b_indices, b_indptr)):
        raise ValueError("all CSR inputs must be CUDA tensors")
    if not all(
        t.device == a_data.device
        for t in (a_indices, a_indptr, b_data, b_indices, b_indptr)
    ):
        raise ValueError("all CSR inputs must be on the same CUDA device")
    if a_data.dtype != b_data.dtype:
        raise TypeError("A and B value dtypes must match for direct hipSPARSE SpGEMM")
    if a_data.ndim != 1 or a_indices.ndim != 1 or a_indptr.ndim != 1:
        raise ValueError("A CSR tensors must be 1D")
    if b_data.ndim != 1 or b_indices.ndim != 1 or b_indptr.ndim != 1:
        raise ValueError("B CSR tensors must be 1D")
    if a_data.numel() != a_indices.numel():
        raise ValueError("A data and indices must have the same length")
    if b_data.numel() != b_indices.numel():
        raise ValueError("B data and indices must have the same length")

    m, k_a = int(a_shape[0]), int(a_shape[1])
    k_b, n = int(b_shape[0]), int(b_shape[1])
    if k_a != k_b:
        raise ValueError(f"inner dimensions must match, got {k_a} and {k_b}")
    if a_indptr.numel() != m + 1:
        raise ValueError(f"A indptr length must be n_rows+1={m + 1}")
    if b_indptr.numel() != k_b + 1:
        raise ValueError(f"B indptr length must be n_rows+1={k_b + 1}")

    a_data = a_data.contiguous()
    a_indices = a_indices.contiguous()
    a_indptr = a_indptr.contiguous()
    b_data = b_data.contiguous()
    b_indices = b_indices.contiguous()
    b_indptr = b_indptr.contiguous()

    c_indptr = torch.zeros(m + 1, dtype=a_indptr.dtype, device=a_data.device)
    # Some hip-python wrappers still validate pointer objects even when nnz == 0.
    c_indices_placeholder = torch.empty(1, dtype=a_indices.dtype, device=a_data.device)
    c_data_placeholder = torch.empty(1, dtype=a_data.dtype, device=a_data.device)

    value_type = _hipsparse_value_type(a_data.dtype)
    alpha = _hipsparse_scalar(a_data.dtype, 1.0, 0.0)
    beta = _hipsparse_scalar(a_data.dtype, 0.0, 0.0)
    a_row_index_type = _hipsparse_index_type(
        a_indptr.dtype, "hipSPARSE CSR SpGEMM A row offsets"
    )
    a_col_index_type = _hipsparse_index_type(
        a_indices.dtype, "hipSPARSE CSR SpGEMM A column indices"
    )
    b_row_index_type = _hipsparse_index_type(
        b_indptr.dtype, "hipSPARSE CSR SpGEMM B row offsets"
    )
    b_col_index_type = _hipsparse_index_type(
        b_indices.dtype, "hipSPARSE CSR SpGEMM B column indices"
    )
    c_col_index_type = _hipsparse_index_type(
        a_indices.dtype, "hipSPARSE CSR SpGEMM C column indices"
    )
    op_enum = _hipsparse_lookup(
        "hipsparseOperation_t", ("HIPSPARSE_OPERATION_NON_TRANSPOSE",)
    )
    index_base = _hipsparse_lookup(
        "hipsparseIndexBase_t", ("HIPSPARSE_INDEX_BASE_ZERO",)
    )
    alg = _hipsparse_spgemm_algorithm()

    handle = None
    mat_a = None
    mat_b = None
    mat_c = None
    descr = None
    buffer1 = 0
    buffer2 = 0
    buffer1_allocated = False
    buffer2_allocated = False
    try:
        handle = _hip_check_result(hipsparse.hipsparseCreate(), "hipsparseCreate")
        ptr_type = type(handle)

        mat_a = ptr_type()
        mat_b = ptr_type()
        mat_c = ptr_type()
        _hipsparse_create_csr_descriptor(
            mat_a.createRef(),
            m,
            k_a,
            int(a_data.numel()),
            a_indptr,
            a_indices,
            a_data,
            a_row_index_type,
            a_col_index_type,
            index_base,
            value_type,
        )
        _hipsparse_create_csr_descriptor(
            mat_b.createRef(),
            k_b,
            n,
            int(b_data.numel()),
            b_indptr,
            b_indices,
            b_data,
            b_row_index_type,
            b_col_index_type,
            index_base,
            value_type,
        )
        _hipsparse_create_csr_descriptor(
            mat_c.createRef(),
            m,
            n,
            0,
            c_indptr,
            c_indices_placeholder,
            c_data_placeholder,
            a_row_index_type,
            c_col_index_type,
            index_base,
            value_type,
        )
        descr = _hipsparse_spgemm_create_descr()

        buffer1_size = _hipsparse_spgemm_work_estimation(
            handle,
            op_enum,
            alpha,
            mat_a,
            mat_b,
            beta,
            mat_c,
            value_type,
            alg,
            descr,
            0,
        )
        if buffer1_size > 0:
            buffer1 = _hip_check_result(hip.hipMalloc(buffer1_size), "hipMalloc")
            buffer1_allocated = True
        _ = _hipsparse_spgemm_work_estimation(
            handle,
            op_enum,
            alpha,
            mat_a,
            mat_b,
            beta,
            mat_c,
            value_type,
            alg,
            descr,
            buffer1,
        )

        buffer2_size = _hipsparse_spgemm_compute(
            handle,
            op_enum,
            alpha,
            mat_a,
            mat_b,
            beta,
            mat_c,
            value_type,
            alg,
            descr,
            0,
        )
        if buffer2_size > 0:
            buffer2 = _hip_check_result(hip.hipMalloc(buffer2_size), "hipMalloc")
            buffer2_allocated = True
        _ = _hipsparse_spgemm_compute(
            handle,
            op_enum,
            alpha,
            mat_a,
            mat_b,
            beta,
            mat_c,
            value_type,
            alg,
            descr,
            buffer2,
        )

        rows_out = ctypes.c_int64()
        cols_out = ctypes.c_int64()
        nnz_out = ctypes.c_int64()
        _hip_check_result(
            hipsparse.hipsparseSpMatGetSize(mat_c, rows_out, cols_out, nnz_out),
            "hipsparseSpMatGetSize",
        )
        nnz_c = int(nnz_out.value)
        c_indices = torch.empty(nnz_c, dtype=a_indices.dtype, device=a_data.device)
        c_data = torch.empty(nnz_c, dtype=a_data.dtype, device=a_data.device)
        _hip_check_result(
            hipsparse.hipsparseCsrSetPointers(
                mat_c,
                HipPointer.fromObj(c_indptr.data_ptr()),
                HipPointer.fromObj(c_indices.data_ptr()),
                HipPointer.fromObj(c_data.data_ptr()),
            ),
            "hipsparseCsrSetPointers",
        )
        return {
            "backend": "hipsparse",
            "buffer1_size": buffer1_size,
            "buffer2_size": buffer2_size,
            "compute_size_out": ctypes.c_size_t(buffer2_size),
            "handle": handle,
            "mat_a": mat_a,
            "mat_b": mat_b,
            "mat_c": mat_c,
            "descr": descr,
            "buffer1": buffer1,
            "buffer2": buffer2,
            "buffer1_allocated": buffer1_allocated,
            "buffer2_allocated": buffer2_allocated,
            "op_enum": op_enum,
            "alpha": alpha,
            "beta": beta,
            "value_type": value_type,
            "alg": alg,
            "result": (c_data, c_indices, c_indptr, (m, n)),
        }
    finally:
        if handle is None and descr is not None:
            try:
                _hip_check_result(
                    hipsparse.hipsparseSpGEMM_destroyDescr(descr),
                    "hipsparseSpGEMM_destroyDescr",
                )
            except Exception:
                pass
        for name, mat in (
            ("hipsparseDestroySpMat(C)", mat_c),
            ("hipsparseDestroySpMat(B)", mat_b),
            ("hipsparseDestroySpMat(A)", mat_a),
        ):
            if handle is None and mat is not None:
                try:
                    _hip_check_result(hipsparse.hipsparseDestroySpMat(mat), name)
                except Exception:
                    pass
        if handle is None and buffer2_allocated:
            try:
                _hip_check_result(hip.hipFree(buffer2), "hipFree(buffer2)")
            except Exception:
                pass
        if handle is None and buffer1_allocated:
            try:
                _hip_check_result(hip.hipFree(buffer1), "hipFree(buffer1)")
            except Exception:
                pass
 

def _run_spgemm_csr_ref_hipsparse_prepared(state):
    _hip_check_result(
        hipsparse.hipsparseSpGEMM_compute(
            state["handle"],
            state["op_enum"],
            state["op_enum"],
            state["alpha"],
            state["mat_a"],
            state["mat_b"],
            state["beta"],
            state["mat_c"],
            state["value_type"],
            state["alg"],
            state["descr"],
            state["compute_size_out"],
            state["buffer2"],
        ),
        "hipsparseSpGEMM_compute",
    )
    _hip_check_result(
        hipsparse.hipsparseSpGEMM_copy(
            state["handle"],
            state["op_enum"],
            state["op_enum"],
            state["alpha"],
            state["mat_a"],
            state["mat_b"],
            state["beta"],
            state["mat_c"],
            state["value_type"],
            state["alg"],
            state["descr"],
        ),
        "hipsparseSpGEMM_copy",
    )
    return state["result"]


def _destroy_spgemm_csr_ref_hipsparse_prepared(state):
    descr = state.get("descr")
    mat_c = state.get("mat_c")
    mat_b = state.get("mat_b")
    mat_a = state.get("mat_a")
    buffer2_allocated = bool(state.get("buffer2_allocated"))
    buffer1_allocated = bool(state.get("buffer1_allocated"))
    buffer2 = state.get("buffer2", 0)
    buffer1 = state.get("buffer1", 0)
    handle = state.get("handle")
    if descr is not None:
        try:
            _hip_check_result(
                hipsparse.hipsparseSpGEMM_destroyDescr(descr),
                "hipsparseSpGEMM_destroyDescr",
            )
        except Exception:
            pass
    for name, mat in (
        ("hipsparseDestroySpMat(C)", mat_c),
        ("hipsparseDestroySpMat(B)", mat_b),
        ("hipsparseDestroySpMat(A)", mat_a),
    ):
        if mat is not None:
            try:
                _hip_check_result(hipsparse.hipsparseDestroySpMat(mat), name)
            except Exception:
                pass
    if buffer2_allocated:
        try:
            _hip_check_result(hip.hipFree(buffer2), "hipFree(buffer2)")
        except Exception:
            pass
    if buffer1_allocated:
        try:
            _hip_check_result(hip.hipFree(buffer1), "hipFree(buffer1)")
        except Exception:
            pass
    if handle is not None:
        try:
            _hip_check_result(hipsparse.hipsparseDestroy(handle), "hipsparseDestroy")
        except Exception:
            pass


def _spgemm_csr_ref_hipsparse(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
    return_metadata=False,
):
    state = _prepare_spgemm_csr_ref_hipsparse(
        a_data,
        a_indices,
        a_indptr,
        a_shape,
        b_data,
        b_indices,
        b_indptr,
        b_shape,
    )
    try:
        result = _run_spgemm_csr_ref_hipsparse_prepared(state)
        metadata = {
            "backend": "hipsparse",
            "buffer1_size": int(state.get("buffer1_size", 0)),
            "buffer2_size": int(state.get("buffer2_size", 0)),
        }
        if return_metadata:
            return result, metadata
        return result
    finally:
        _destroy_spgemm_csr_ref_hipsparse_prepared(state)


def _spgemm_csr_sparse_ref_backend(
    value_dtype=None,
    a_indices_dtype=None,
    a_indptr_dtype=None,
    b_indices_dtype=None,
    b_indptr_dtype=None,
):
    if _is_rocm_runtime():
        reason = _hipsparse_spgemm_csr_skip_reason(
            value_dtype,
            a_indices_dtype,
            a_indptr_dtype,
            b_indices_dtype,
            b_indptr_dtype,
        )
        if reason is None:
            return "hipsparse", None
        return None, reason
    if cp is None or cpx_sparse is None:
        return None, "CuPy/cuSPARSE is not available"
    return "cupy_cusparse", None


def _benchmark_spgemm_csr_sparse_ref(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
    warmup,
    iters,
    value_dtype,
):
    backend, reason = _spgemm_csr_sparse_ref_backend(
        a_data.dtype,
        a_indices.dtype,
        a_indptr.dtype,
        b_indices.dtype,
        b_indptr.dtype,
    )
    result = {
        "backend": backend,
        "values": None,
        "ms": None,
        "reason": reason,
    }
    if backend is None:
        return result

    if backend == "hipsparse":
        values, ms = _common_mod._benchmark_prepared_cuda_op(
            lambda: _prepare_spgemm_csr_ref_hipsparse(
                a_data,
                a_indices,
                a_indptr,
                a_shape,
                b_data,
                b_indices,
                b_indptr,
                b_shape,
            ),
            _run_spgemm_csr_ref_hipsparse_prepared,
            _destroy_spgemm_csr_ref_hipsparse_prepared,
            warmup=warmup,
            iters=iters,
        )
        result["values"] = values
        result["ms"] = ms
        result["reason"] = None
        return result

    a_cp = cpx_sparse.csr_matrix(
        (
            _cupy_from_torch(a_data),
            _cupy_from_torch(a_indices.to(torch.int64)),
            _cupy_from_torch(a_indptr.to(torch.int64)),
        ),
        shape=a_shape,
    )
    b_cp = cpx_sparse.csr_matrix(
        (
            _cupy_from_torch(b_data),
            _cupy_from_torch(b_indices.to(torch.int64)),
            _cupy_from_torch(b_indptr.to(torch.int64)),
        ),
        shape=b_shape,
    )
    c_cp, ms = _benchmark_cuda_op(lambda: a_cp @ b_cp, warmup=warmup, iters=iters)
    c_coo = c_cp.tocoo()
    rows = _torch_from_cupy(c_coo.row).to(torch.int64)
    cols = _torch_from_cupy(c_coo.col).to(torch.int64)
    vals = _torch_from_cupy(c_coo.data).to(value_dtype)
    c_t = torch.sparse_coo_tensor(
        torch.stack([rows, cols]),
        vals,
        (int(a_shape[0]), int(b_shape[1])),
        device=a_data.device,
    ).coalesce()
    result["values"] = _torch_sparse_to_csr(c_t)
    result["ms"] = ms
    result["reason"] = None
    return result


def benchmark_spgemm_case(
    n_rows=1024,
    n_inner=1024,
    n_cols=1024,
    nnz_a=16384,
    nnz_b=16384,
    value_dtype=torch.float32,
    warmup=10,
    iters=30,
    run_cusparse=True,
):
    """Benchmark CSR SpGEMM and compare with torch/cuSPARSE baselines."""
    if value_dtype not in SUPPORTED_SPGEMM_VALUE_DTYPES:
        raise TypeError("value_dtype must be torch.float32 or torch.float64")
    device = torch.device("cuda")
    a_data, a_indices, a_indptr = _build_random_csr(
        n_rows, n_inner, nnz_a, value_dtype, torch.int32, device
    )
    b_data, b_indices, b_indptr = _build_random_csr(
        n_inner, n_cols, nnz_b, value_dtype, torch.int32, device
    )

    prepared = prepare_spgemm_csr(
        a_data, a_indices, a_indptr, (n_rows, n_inner),
        b_data, b_indices, b_indptr, (n_inner, n_cols),
    )
    op = lambda: flagsparse_spgemm_csr(prepared=prepared, return_time=False)
    triton_result, triton_ms = _benchmark_cuda_op(op, warmup=warmup, iters=iters)

    pytorch_result, ref_meta = _spgemm_csr_reference(
        a_data,
        a_indices,
        a_indptr,
        (n_rows, n_inner),
        b_data,
        b_indices,
        b_indptr,
        (n_inner, n_cols),
        return_metadata=True,
    )
    a_t = _to_torch_csr(a_data, a_indices, a_indptr, (n_rows, n_inner))
    b_t = _to_torch_csr(b_data, b_indices, b_indptr, (n_inner, n_cols))

    pytorch_reason = ref_meta["fallback_reason"]
    pytorch_ms = None
    try:
        if ref_meta["pytorch_sparse_format"] == "CSR":
            torch_op = lambda: torch.sparse.mm(a_t, b_t)
        else:
            a_coo = a_t.to_sparse_coo().coalesce()
            b_coo = b_t.to_sparse_coo().coalesce()
            torch_op = lambda: torch.sparse.mm(a_coo, b_coo)
        pytorch_sparse, pytorch_ms = _benchmark_cuda_op(torch_op, warmup=warmup, iters=iters)
        pytorch_result = _torch_sparse_to_csr(pytorch_sparse)
    except Exception as exc:
        pytorch_reason = (
            str(exc) if pytorch_reason is None else f"{pytorch_reason}; timing: {exc}"
        )

    triton_summary = _spgemm_pairwise_summary(triton_result, pytorch_result, value_dtype)

    cusparse_ms = None
    cusparse_reason = None
    cusparse_match = None
    sparse_ref_backend = None
    if run_cusparse:
        try:
            sparse_ref = _benchmark_spgemm_csr_sparse_ref(
                a_data,
                a_indices,
                a_indptr,
                (n_rows, n_inner),
                b_data,
                b_indices,
                b_indptr,
                (n_inner, n_cols),
                warmup=warmup,
                iters=iters,
                value_dtype=value_dtype,
            )
            sparse_ref_backend = sparse_ref["backend"]
            if sparse_ref_backend is not None:
                cusparse_ms = sparse_ref["ms"]
                cusparse_match = _spgemm_pairwise_summary(
                    triton_result,
                    sparse_ref["values"],
                    value_dtype,
                )["match"]
            else:
                cusparse_reason = sparse_ref["reason"]
        except Exception as exc:
            cusparse_reason = str(exc)

    return {
        "parameters": {
            "n_rows": n_rows,
            "n_inner": n_inner,
            "n_cols": n_cols,
            "nnz_a": nnz_a,
            "nnz_b": nnz_b,
            "value_dtype": str(value_dtype),
            "warmup": warmup,
            "iters": iters,
        },
        "performance": {
            "triton_ms": triton_ms,
            "pytorch_ms": pytorch_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_pytorch": (pytorch_ms / triton_ms if (pytorch_ms and triton_ms > 0) else None),
            "triton_speedup_vs_cusparse": (cusparse_ms / triton_ms if (cusparse_ms and triton_ms > 0) else None),
        },
        "verification": {
            "triton_match_pytorch": triton_summary["match"],
            "triton_max_abs_error": triton_summary["max_abs_error"],
            "triton_max_relative_error": triton_summary["max_relative_error"],
            "cusparse_match_pytorch": cusparse_match,
        },
        "backend_status": {
            "pytorch_unavailable_reason": pytorch_reason,
            "pytorch_sparse_format": ref_meta["pytorch_sparse_format"],
            "cusparse_unavailable_reason": cusparse_reason,
            "sparse_ref_backend": sparse_ref_backend,
        },
        "samples": {
            "triton": triton_result,
            "pytorch": pytorch_result,
        },
    }
