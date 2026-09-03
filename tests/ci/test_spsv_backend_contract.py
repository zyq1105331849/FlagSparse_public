# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Static contracts for the shared CUDA/ROCm SpSV implementation."""

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_PATH = PROJECT_ROOT / "src" / "flagsparse" / "sparse_operations" / "_common.py"
COMMON_SOURCE = COMMON_PATH.read_text(encoding="utf-8")
COMMON_TREE = ast.parse(COMMON_SOURCE)
SPSV_PATH = PROJECT_ROOT / "src" / "flagsparse" / "sparse_operations" / "spsv.py"
SPSV_SOURCE = SPSV_PATH.read_text(encoding="utf-8")
SPSV_TREE = ast.parse(SPSV_SOURCE)
SPSV_BENCHMARK_PATH = PROJECT_ROOT / "tests" / "test_spsv.py"
SPSV_BENCHMARK_SOURCE = SPSV_BENCHMARK_PATH.read_text(encoding="utf-8")
SPSV_BENCHMARK_TREE = ast.parse(SPSV_BENCHMARK_SOURCE)
SPSV_SELL_BENCHMARK_SOURCE = (
    PROJECT_ROOT / "tests" / "test_spsv_sell.py"
).read_text(encoding="utf-8")
SPSV_SELL_BENCHMARK_TREE = ast.parse(SPSV_SELL_BENCHMARK_SOURCE)
SPSV_ACCURACY_SOURCE = (
    PROJECT_ROOT / "tests" / "pytest" / "test_spsv_csr_accuracy.py"
).read_text(encoding="utf-8")
SPSV_SELL_ACCURACY_SOURCE = (
    PROJECT_ROOT / "tests" / "pytest" / "test_spsv_sell_accuracy.py"
).read_text(encoding="utf-8")
SPSM_PATH = PROJECT_ROOT / "src" / "flagsparse" / "sparse_operations" / "spsm.py"
SPSM_SOURCE = SPSM_PATH.read_text(encoding="utf-8")
SPSM_TREE = ast.parse(SPSM_SOURCE)
SPSM_BENCHMARK_PATH = PROJECT_ROOT / "tests" / "test_spsm.py"
SPSM_BENCHMARK_SOURCE = SPSM_BENCHMARK_PATH.read_text(encoding="utf-8")
SPSM_BENCHMARK_TREE = ast.parse(SPSM_BENCHMARK_SOURCE)


def _function_source(name):
    for node in SPSV_TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(SPSV_SOURCE, node)
    raise AssertionError(f"function {name!r} not found")


def _common_function_source(name):
    for node in COMMON_TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(COMMON_SOURCE, node)
    raise AssertionError(f"common function {name!r} not found")


def _load_csr_descriptor_helper(mock_hipsparse):
    namespace = {
        "hipsparse": mock_hipsparse,
        "_hip_check_result": lambda result, _name: (
            result[1] if isinstance(result, tuple) and len(result) == 2 else None
        ),
    }
    exec(_common_function_source("_hipsparse_create_csr_descriptor"), namespace)
    return namespace["_hipsparse_create_csr_descriptor"]


class _MockHipPointer:
    def createRef(self):
        return ("descriptor_ref", self)


def _call_mock_csr_descriptor(helper):
    descriptor = _MockHipPointer()
    result = helper(
        descriptor.createRef(),
        2,
        2,
        2,
        _MockHipPointer(),
        _MockHipPointer(),
        _MockHipPointer(),
        "row_type",
        "col_type",
        "zero",
        "float32",
    )
    return descriptor, result


def test_hipsparse_csr_descriptor_uses_explicit_output_wrapper():
    class ExplicitOutputWrapper:
        calls = []

        @classmethod
        def hipsparseCreateCsr(cls, *args):
            cls.calls.append(args)
            assert len(args) == 11
            assert args[0][0] == "descriptor_ref"
            return (0,)

    helper = _load_csr_descriptor_helper(ExplicitOutputWrapper)
    descriptor, result = _call_mock_csr_descriptor(helper)
    assert isinstance(descriptor, _MockHipPointer)
    assert result is None
    assert [len(args) for args in ExplicitOutputWrapper.calls] == [11]


def _benchmark_function_source(name):
    for node in SPSV_BENCHMARK_TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(SPSV_BENCHMARK_SOURCE, node)
    raise AssertionError(f"benchmark function {name!r} not found")


def _spsm_function_source(name):
    for node in SPSM_TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(SPSM_SOURCE, node)
    raise AssertionError(f"SpSM function {name!r} not found")


def _spsm_benchmark_function_source(name):
    for node in SPSM_BENCHMARK_TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(SPSM_BENCHMARK_SOURCE, node)
    raise AssertionError(f"SpSM benchmark function {name!r} not found")


def test_shared_cw_kernels_use_launch_time_serial_policy():
    for name in ("_spsv_csr_cw_kernel", "_spsv_csr_cw_kernel_complex"):
        source = _function_source(name)
        assert "SERIAL_EXECUTION" in source
        assert "_is_rocm_runtime" not in source


def test_shared_transpose_kernels_use_launch_time_serial_policy():
    for name in (
        "_spsv_csr_transpose_cw_kernel",
        "_spsv_csr_transpose_cw_kernel_complex",
    ):
        source = _function_source(name)
        assert "SERIAL_EXECUTION" in source
        assert "_is_rocm_runtime" not in source


def test_rocm_nontrans_cw_uses_cu_capped_persistent_workers():
    for name in (
        "_triton_spsv_csr_cw_vector",
        "_triton_spsv_csr_cw_vector_complex",
    ):
        source = _function_source(name)
        assert "persistent_parallel = is_rocm and SPSV_ROCM_ENABLE_PERSISTENT_PARALLEL" in source
        assert "_spsv_alg4_worker_count(n_rows, b_vec.device, True)" in source
        assert "serial_execution = is_rocm and not persistent_parallel" in source
        assert "SERIAL_EXECUTION=serial_execution" in source

    # TRANS/CONJ was not part of the CSR-NON performance change.
    for name in (
        "_triton_spsv_csr_transpose_cw_vector",
        "_triton_spsv_csr_transpose_cw_vector_complex",
    ):
        source = _function_source(name)
        assert "serial_execution = _is_rocm_runtime()" in source
        assert "if serial_execution:\n        worker_count = 1" in source


def test_rocm_alg3_directly_reuses_cuda_alg8_nnz_balance_kernel():
    roc = _function_source("_triton_spsv_csr_n_lo_roc_vector")
    level = _function_source("_triton_spsv_csr_n_lo_cw_levelschd_vector")
    nnz = _function_source("_triton_spsv_csr_n_lo_nnz_balance_vector")
    nnz_complex = _function_source(
        "_triton_spsv_csr_n_lo_nnz_balance_vector_complex"
    )

    # CUDA's legacy ALG3 stays on its original unscheduled kernel. Only the
    # shared DCU ALG2 wrapper retains the explicit per-level launch path.
    assert "LEVEL_SCHEDULED=False" in roc
    assert "LEVEL_SCHEDULED=True" not in roc
    assert "for start, end in zip(bounds, bounds[1:])" in level
    assert "LEVEL_SCHEDULED=True" in level
    assert "LEVEL_SCHEDULED=False" in level

    # DCU ALG3 is NNZ-balanced itself: it must not redirect to old ALG3/ALG4.
    assert "_triton_spsv_csr_n_lo_smblk_vector(" not in nnz
    assert "_triton_spsv_csr_n_lo_roc_vector(" not in nnz
    assert "_spsv_csr_nnz_balance_kernel[grid]" in nnz
    assert "_spsv_csr_nnz_balance_kernel_complex[grid]" in nnz_complex
    assert "persistent_kernel" not in nnz
    assert "persistent_kernel" not in nnz_complex
    for source in (nnz, nnz_complex):
        assert "_prepare_spsv_nnz_balance_runtime_buffers(" in source
        assert "_spsv_nnz_balance_launch_config(nnz, b_vec.device)" in source
        assert "grid = (worker_count,)" in source
        assert "PERSISTENT=is_rocm" in source
        assert "NUM_WORKERS=worker_count" in source
        assert "BLOCK_NNZ=block_nnz" in source
        assert "num_warps=num_warps" in source

    runtime_buffers = _function_source(
        "_prepare_spsv_nnz_balance_runtime_buffers"
    )
    assert "x = torch.zeros_like(b_vec)" in runtime_buffers
    assert "tmp_sum.zero_()" in runtime_buffers
    assert "ready.zero_()" in runtime_buffers
    assert "indegree.copy_(indegree_init)" in runtime_buffers

    launch_config = _function_source("_spsv_nnz_balance_launch_config")
    assert "if not is_rocm:" in launch_config
    assert "return False, 1, int(nnz), 1" in launch_config
    assert "block_nnz = SPSV_ROCM_ALG3_BLOCK_NNZ" in launch_config
    assert "multi_processor_count" in launch_config
    assert "worker_cap = cu_count * SPSV_ROCM_ALG3_WORKGROUPS_PER_CU" in launch_config
    assert "worker_count = min(triton.cdiv(nnz, block_nnz), max(1, worker_cap))" in launch_config

    for name in (
        "_spsv_csr_nnz_balance_kernel",
        "_spsv_csr_nnz_balance_kernel_complex",
    ):
        kernel = _function_source(name)
        assert "PERSISTENT: tl.constexpr" in kernel
        assert "NUM_WORKERS: tl.constexpr" in kernel
        assert "BLOCK_NNZ: tl.constexpr" in kernel
        assert "offsets = tl.arange(0, BLOCK_NNZ)" in kernel
        assert "tl.program_id(0) * BLOCK_NNZ + offsets" in kernel
        assert "while tl.sum(active.to(tl.int32), axis=0) > 0" in kernel
        assert "val_id += BLOCK_NNZ * NUM_WORKERS" in kernel
        assert 'sem="acquire"' in kernel
        assert 'sem="release"' in kernel
        assert 'scope="gpu"' in kernel
        assert "if PERSISTENT:" in kernel
        assert "tl.debug_barrier()" not in kernel
        assert "x_ptr + col" in kernel or "x_ri_ptr + col" in kernel
        assert "tl.atomic_add(" in kernel

    real_kernel = _function_source("_spsv_csr_nnz_balance_kernel")
    complex_kernel = _function_source("_spsv_csr_nnz_balance_kernel_complex")
    assert "tl.store(x_ptr + row" not in real_kernel
    assert "tl.store(x_ri_ptr + row" not in complex_kernel

    assert '"FLAGSPARSE_SPSV_ROCM_ALG3_BLOCK_NNZ", "256"' in SPSV_SOURCE
    assert '"FLAGSPARSE_SPSV_ROCM_ALG3_WORKGROUPS_PER_CU", "4"' in SPSV_SOURCE
    assert "1 <= SPSV_ROCM_ALG3_WORKGROUPS_PER_CU <= 8" in SPSV_SOURCE

    # Lower metadata has one backend-independent full-overwrite contract. The
    # only analysis launch difference is the native wave/warp width.
    metadata = _function_source("_build_spsv_nnz_balance_metadata")
    assert "WARP_SIZE=64 if _is_rocm_runtime() else 32" in metadata
    assert "FILL_ALL_ROWS" not in metadata
    assert "allocator = torch.empty if is_rocm else torch.zeros" not in metadata
    assert "indegree32 = torch.empty" in metadata
    assert "row_idx32 = torch.empty" in metadata
    assert '"launch_order32": torch.empty(0' in metadata
    assert '"kernel_indices32": indices32' in metadata
    assert "torch.arange(indices32.numel()" not in metadata
    assert "if indices64.is_cuda:" not in metadata

    prepare = _function_source("_prepare_spsv_csr_system")
    assert 'nnz_meta["kernel_indices32"]' in prepare
    assert '"kernel_indices32": kernel_indices32' in prepare
    assert "currently supports lower only" in prepare

    preprocess = _function_source("_spsv_nnz_balance_preprocess_kernel")
    assert "LOWER: tl.constexpr" not in preprocess
    assert "FILL_ALL_ROWS: tl.constexpr" not in preprocess
    assert "BUILD_UPPER_ORDER: tl.constexpr" not in preprocess
    assert "launch_order_ptr" not in preprocess
    assert "tl.store(row_idx_ptr + ptr, row, mask=active)" in preprocess
    assert "active = ptr < end" in preprocess

    assert "if LOWER:" in _function_source(
        "_spsv_csr_nnz_balance_kernel"
    )
    assert "if LOWER:" in _function_source(
        "_spsv_csr_nnz_balance_kernel_complex"
    )

    executor = _function_source("_execute_spsv_csr_plan")
    assert "rocm_serial_nontrans" not in executor


def test_rocm_alg3_uses_native_compute_dtype_for_atomic_accumulation():
    layout = _function_source("_build_spsv_workspace_layout")
    real = _function_source("_triton_spsv_csr_n_lo_nnz_balance_vector")
    complex_source = _function_source(
        "_triton_spsv_csr_n_lo_nnz_balance_vector_complex"
    )
    runtime_buffers = _function_source(
        "_prepare_spsv_nnz_balance_runtime_buffers"
    )

    assert '_workspace_entry("tmp_sum", n_rows, value_dtype)' in layout
    for source in (real, complex_source):
        assert "acc_dtype = data.dtype" in source
        assert "_prepare_spsv_nnz_balance_runtime_buffers(" in source
    assert "tmp_sum.dtype != b_vec.dtype" in runtime_buffers
    assert "tmp_sum_in is not None" not in runtime_buffers
    assert "ready_in is not None" not in runtime_buffers
    assert "indegree_in is not None" not in runtime_buffers


def test_rocm_alg3_initializes_mutable_state_once_with_framework_kernels():
    real = _function_source("_triton_spsv_csr_n_lo_nnz_balance_vector")
    complex_source = _function_source(
        "_triton_spsv_csr_n_lo_nnz_balance_vector_complex"
    )
    runtime_buffers = _function_source(
        "_prepare_spsv_nnz_balance_runtime_buffers"
    )
    executor = _function_source("_execute_spsv_csr_plan")

    assert "def _spsv_nnz_balance_init_kernel(" not in SPSV_SOURCE
    for source in (real, complex_source):
        assert "_prepare_spsv_nnz_balance_runtime_buffers(" in source
    assert "x = torch.zeros_like(b_vec)" in runtime_buffers
    assert "tmp_sum.zero_()" in runtime_buffers
    assert "ready.zero_()" in runtime_buffers
    assert "indegree.copy_(indegree_init)" in runtime_buffers
    assert "tmp_sum_buf.zero_()" not in executor
    assert "indegree_buf.copy_(nnz_balance_indegree32)" not in executor


def test_rocm_topology_analysis_uses_cu_capped_persistent_grid():
    dispatcher = _function_source("_build_spsv_level_schedule_metadata")
    builder = _function_source("_build_spsv_level_schedule_metadata_rocm_gpu")
    kernel = _function_source("_spsv_levelschd_analysis_persistent_kernel")

    assert "if _is_rocm_runtime():" in dispatcher
    assert "_build_spsv_level_schedule_metadata_rocm_gpu(" in dispatcher
    assert "if SPSV_ROCM_ENABLE_PERSISTENT_PARALLEL:" in builder
    assert "_spsv_alg4_worker_count(n_blocks, device, True)" in builder
    assert "_spsv_levelschd_analysis_persistent_kernel[(worker_count,)]" in builder
    assert "_spsv_levelschd_analysis_serial_kernel[(1,)]" in builder
    assert "torch.argsort" in builder
    assert "block_counter_ptr" in kernel
    assert 'sem="acquire"' in kernel
    assert 'sem="release"' in kernel


def test_rocm_alg3_builds_nnz_metadata_without_level_fallback():
    prepare = _function_source("_prepare_spsv_csr_system")
    nnz_branch = prepare.split('elif effective_route == "csr_nnz_balance":', 1)[1]
    nnz_branch = nnz_branch.split(
        "        else:\n            if not _supports_spsv_advanced_nontrans_routes", 1
    )[0]
    assert "_build_spsv_nnz_balance_metadata(" in nnz_branch
    assert "_build_spsv_level_schedule_metadata(" not in nnz_branch
    assert 'default_solve_kind = "csr_nnz_balance"' in nnz_branch


def test_backend_specific_alg_numbers_remove_old_dcu_alg3_alg4_alg8():
    assert '3: "csr_roc"' in SPSV_BENCHMARK_SOURCE
    assert '4: "csr_smblk"' in SPSV_BENCHMARK_SOURCE
    assert '8: "csr_nnz_balance"' in SPSV_BENCHMARK_SOURCE
    rocm_map = SPSV_BENCHMARK_SOURCE.split(
        "ROCM_SPSV_ALG_NUM_TO_SOLVE_KIND = {", 1
    )[1].split("}", 1)[0]
    assert '3: "csr_nnz_balance"' in rocm_map
    assert '4:' not in rocm_map
    assert '8:' not in rocm_map

    normalize = _function_source("_normalize_requested_spsv_route")
    assert '"alg3": "csr_nnz_balance" if is_rocm else "csr_roc"' in normalize
    for token in ('"csr_roc"', '"roc"', '"csr_smblk"', '"smblk"', '"alg4"', '"alg8"'):
        assert token in normalize.split("aliases =", 1)[0]


def test_cuda_level_analysis_does_not_copy_level_bounds_to_host():
    builder = _function_source("_build_spsv_level_schedule_metadata_lower_gpu")
    assert '"level_ptr_host": None' in builder
    assert '.to("cpu")' not in builder


def test_cuda_keeps_requested_parallel_routes():
    source = _function_source("_execute_spsv_csr_plan")
    assert "elif solve_kind == \"csr_roc\"" in source
    assert "elif solve_kind == \"csr_smblk\"" in source
    assert "elif solve_kind == \"csr_cw_levelschd\"" in source
    assert "elif solve_kind == \"csr_nnz_balance\"" in source
    assert "and _is_rocm_runtime()\n        and worker_count_use == 1" in source

    for name in (
        "_triton_spsv_csr_n_lo_roc_vector",
        "_triton_spsv_csr_n_lo_roc_vector_complex",
        "_triton_spsv_csr_n_lo_cw_levelschd_vector",
        "_triton_spsv_csr_n_lo_cw_levelschd_vector_complex",
    ):
        assert "LEVEL_SCHEDULED=False" in _function_source(name)


def test_rocm_auto_can_use_safe_advanced_route_by_default():
    assert '"FLAGSPARSE_SPSV_ROCM_ENABLE_ADVANCED_AUTO", "1"' in SPSV_SOURCE


def test_rocm_vendor_reference_includes_analysis_and_solve_per_round():
    source = _benchmark_function_source("_cupy_spsolve_csr_with_op")
    assert 'fs_spsv_impl._spsv_csr_sparse_ref_backend(' in source
    assert 'if vendor_backend == "hipsparse":' in source
    assert "fresh_each_iter=True" in source
    assert "time.perf_counter()" in source
    assert "deviceSynchronize()" in source

    vendor_source = _function_source("_benchmark_spsv_csr_sparse_ref")
    timed_loop = vendor_source.split("for _ in range(iters):", 1)[1]
    assert "state = _prepare_spsv_csr_ref_hipsparse(" in timed_loop
    assert "values = _run_spsv_csr_ref_hipsparse_prepared(state)" in timed_loop
    assert "_destroy_spsv_csr_ref_hipsparse_prepared(state)" in timed_loop
    assert "_reanalyze_spsv_csr_ref_hipsparse_prepared(state)" not in timed_loop


def test_rocm_vendor_dispatch_matches_spmv_spmm_without_cross_backend_fallback():
    source = _benchmark_function_source("_cupy_spsolve_csr_with_op")
    rocm_branch, cuda_cupy_path = source.split(
        "if cp is None or cpx_sparse is None or cpx_spsolve_triangular is None:",
        1,
    )
    assert 'sparse_ref.get("backend") != "hipsparse"' in rocm_branch
    assert "ROCm/DCU vendor dispatch did not select hipSPARSE" in rocm_branch
    assert '"hipSPARSE direct API"' in rocm_branch
    assert "cpx_spsolve_triangular(" not in rocm_branch
    assert "cpx_spsolve_triangular(" in cuda_cupy_path
    assert '"cuSPARSE via CuPy spsolve_triangular"' in cuda_cupy_path


def test_spsv_output_uses_runtime_vendor_name_and_total_time_only():
    for name in (
        "run_spsv_synthetic_all",
        "run_all_supported_spsv_csr_csv",
        "run_all_dtypes_spsv_coo_csv",
    ):
        source = _benchmark_function_source(name)
        assert "_vendor_backend_name()" in source
        assert "_vendor_short_name()" in source
        assert "FS.an" not in source
        assert "FS.sol" not in source
        assert "spdS" not in source
    selector = _function_source("_spsv_csr_sparse_ref_backend")
    assert "if _is_rocm_runtime():" in selector
    assert 'return "hipsparse", None' in selector
    assert 'return "cupy_cusparse", None' in selector

    csv_fields = _benchmark_function_source("_spsv_csv_fieldnames")
    assert "backend_name = _vendor_backend_name()" in csv_fields
    assert 'f"{backend_name}_ms"' in csv_fields
    assert 'f"{backend_name}_route"' in csv_fields
    assert 'f"FlagSparse_vs_{backend_name}_speedup"' in csv_fields
    assert "_backend_error_key()" in csv_fields
    assert '"err_ref"' not in csv_fields
    assert '"err_res"' not in csv_fields
    assert 'f"{backend_name}_reason"' in csv_fields
    backend_error_key = _benchmark_function_source("_backend_error_key")
    assert 'return "err_hip" if' in backend_error_key
    assert 'else "err_cu"' in backend_error_key
    assert '"err_vendor"' not in backend_error_key
    for generic_name in (
        '"vendor_backend"',
        '"vendor_route"',
        '"FlagSparse_vs_vendor_speedup"',
        '"err_vendor"',
        '"vendor_reason"',
    ):
        assert generic_name not in csv_fields
    assert '"cuSPARSE_ms"' not in csv_fields
    assert '"hipSPARSE_ms"' not in csv_fields
    for name in ("_finalize_csv_row", "_finalize_csv_row_csr_full"):
        source = _benchmark_function_source(name)
        assert 'f"{vendor_backend}_ms"' in source
        assert 'f"{vendor_backend}_route"' in source
        assert 'f"FlagSparse_vs_{vendor_backend}_speedup"' in source
        assert "backend_error_key = _backend_error_key()" in source
        assert "backend_error_key: err_vendor" in source
        assert '"err_ref"' not in source
        assert '"err_res"' not in source
        assert '"_err_res": err_res' in source
        assert 'f"{vendor_backend}_reason"' in source
        for generic_name in (
            '"vendor_backend"',
            '"vendor_route"',
            '"FlagSparse_vs_vendor_speedup"',
            '"err_vendor"',
            '"vendor_reason"',
        ):
            assert generic_name not in source
    for name in (
        "run_all_supported_spsv_csr_csv",
        "run_all_dtypes_spsv_coo_csv",
    ):
        source = _benchmark_function_source(name)
        assert "fieldnames = _spsv_csv_fieldnames()" in source


def test_spsv_csv_error_columns_are_minimal_and_residual_stays_diagnostic():
    csv_fields_node = next(
        node
        for node in SPSV_SELL_BENCHMARK_TREE.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "CSV_FIELDS"
            for target in node.targets
        )
    )
    sell_fields = ast.literal_eval(csv_fields_node.value)
    assert [field for field in sell_fields if field.startswith("err_")] == [
        "err_pt",
        "err_cu",
    ]
    assert not any(field.startswith("rel_") for field in sell_fields)
    assert '"_err_res": err_res' in SPSV_SELL_BENCHMARK_SOURCE


def test_spsm_vendor_dispatch_matches_spmv_spmm_selector_shape():
    selector = _spsm_function_source("_spsm_csr_sparse_ref_backend")
    assert "if _is_rocm_runtime():" in selector
    assert 'return "hipsparse", None' in selector
    assert 'return "native_cusparse", None' in selector

    benchmark = _spsm_benchmark_function_source("_benchmark_cusparse_reference")
    assert "fs_spsm_impl._spsm_csr_sparse_ref_backend(" in benchmark
    assert 'if vendor_backend == "hipsparse":' in benchmark
    assert 'if vendor_backend != "native_cusparse":' in benchmark
    assert "fs_spsm_impl._is_rocm_runtime()" not in benchmark
    assert '"hipSPARSE csrsm2 direct API"' in benchmark
    assert '"native cuSPARSE SpSM API"' in benchmark
    assert 'col if fmt == "csr" else row' not in benchmark


def test_spsm_output_uses_runtime_vendor_name():
    source = SPSM_BENCHMARK_SOURCE
    assert '"vendor_backend": vendor_backend' in source
    assert '"hipSPARSE_ms": vendor_ms if vendor_backend == "hipSPARSE"' in source
    assert '"cuSPARSE_ms": vendor_ms if vendor_backend == "cuSPARSE"' in source
    assert '"FlagSparse_vs_vendor_speedup"' in source
    assert '"FlagSparse_vs_cuSPARSE_speedup"' not in source
    assert '"err_cu"' not in source
    assert '"cusparse_reason"' not in source


def test_flagsparse_and_vendor_use_full_spsv_rounds_for_total_speedup():
    rounds = _benchmark_function_source("_benchmark_flagsparse_spsv_full_rounds")
    assert "reset_call()" in rounds
    assert "state = analyze_call()" in rounds
    assert "x = solve_call(state)" in rounds
    assert "total_times.append" in rounds

    for name in (
        "_benchmark_flagsparse_spsv_csr_split",
        "_benchmark_flagsparse_spsv_coo_split",
    ):
        source = _benchmark_function_source(name)
        assert "_benchmark_flagsparse_spsv_full_rounds(" in source

    assert "_amortized_total_ms" not in SPSV_BENCHMARK_SOURCE


def test_spsv_default_rounds_match_spsm():
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in SPSV_BENCHMARK_TREE.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"WARMUP", "ITERS"}
    }
    assert assignments == {"WARMUP": 10, "ITERS": 20}


def test_spsv_float32_tolerances_are_consistent():
    assert "return 1e-6, 1e-5" in SPSV_BENCHMARK_SOURCE
    assert "return 1e-6, 1e-5" in SPSV_SELL_BENCHMARK_SOURCE
    # Accuracy tests unpack (rtol, atol), unlike the benchmark helpers.
    assert "return 1e-5, 1e-6" in SPSV_ACCURACY_SOURCE
    assert "atol, rtol = 1e-6, 1e-5" in SPSV_SELL_ACCURACY_SOURCE
