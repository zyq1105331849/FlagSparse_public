# DCU（ROCm / hipSPARSE）测试指南

本文针对 FlagSparse 的 **DCU 后端分发**改动。CUDA 侧已在 NVIDIA 机器上验证通过，
**hipSPARSE 那条分支在合并时一行都没有执行过**——它只做了导入、语法和分发路径检查。
本文的目的就是把这部分跑起来并确认。

---

## 0. 一句话背景

Triton 内核在两个后端是同一份代码，**只有"厂商参考实现/基线"按后端分发**：

| 运行时 | 判定 | 厂商库 | Python 绑定 |
| --- | --- | --- | --- |
| NVIDIA CUDA | `torch.version.hip is None` | cuSPARSE | CuPy |
| DCU / ROCm | `torch.version.hip is not None` | hipSPARSE | `hip-python` |

所以在 DCU 上出问题，**优先怀疑参考/基线路径，而不是 Triton 内核**。
基线拿不到不会让测试报错，只会让某列变成 `N/A` 并在 `reason` 字段里写原因。

---

## 1. 环境准备

```bash
# 1) 确认 torch 是 ROCm 版本 —— 这是后端分发的唯一判据
python -c "import torch; print('hip=', torch.version.hip, '| cuda=', torch.version.cuda)"
# 期望：hip= 6.x.xxxxx  | cuda= None
```

`torch.version.hip` 为 `None` 时，**所有 hipSPARSE 分支都不会被走到**，
测试会静默回到 CUDA/torch 路径——这时你测的根本不是 DCU 代码。

```bash
# 2) 安装 hip-python（版本需与 ROCm 大版本匹配）
pip install hip-python

# 3) 确认能导入
python -c "from hip import hip, hipsparse; from hip._util.types import Pointer; print('ok')"
```

> `hip-python` 是**可选**依赖。装不上时框架会回落到 `torch.sparse` 参考实现并写明原因，
> 不会崩——但那样就失去了 DCU 验证的意义。

---

## 2. ⚠️ 最容易踩的坑：跑到了旧的已安装包

`import flagsparse` **可能解析到 site-packages 里的旧副本**，而不是你改的 `src/`。
这在 CUDA 机器上真实发生过：新加的符号找不到，表现是**基线列静默变 `N/A`，不报错**。

```bash
# 每次开工先确认
python -c "import flagsparse; print(flagsparse.__file__)"
# 必须指向 <仓库>/src/flagsparse/__init__.py
```

若指向 `/usr/.../site-packages/`，二选一：

```bash
export PYTHONPATH=$PWD/src      # 推荐，最省事
# 或
pip install -e . --no-deps --no-build-isolation
```

仓库内的 `tests/*.py` 已全部自带 `sys.path` 插入，但**你自己写的临时脚本没有**，
所以养成 `PYTHONPATH=src` 的习惯最稳妥。

---

## 3. 第一步永远是 diagnose（不要直接跑基准）

hipSPARSE 调用出错时的典型表现是**卡住**而不是抛异常。
`tests/diagnose_hipsparse_ref.py` 会逐阶段打印，最后一行就是卡住的位置。

```bash
# 3.1 先探环境（不碰任何算子）
python tests/diagnose_hipsparse_ref.py --op env
```

期望看到 `hipSPARSE available=True`、`hipSPARSE unavailable reason=None`。
若显示 `requires a ROCm runtime`，回到第 1 步。

```bash
# 3.2 再单独探 HIP 事件计时链（基准计时依赖它）
python tests/diagnose_hipsparse_ref.py --timing-only

# 3.3 然后一次一个算子，不要一上来就 --op all
python tests/diagnose_hipsparse_ref.py --op spmv-csr
python tests/diagnose_hipsparse_ref.py --op spmv-coo
python tests/diagnose_hipsparse_ref.py --op spmm-csr
python tests/diagnose_hipsparse_ref.py --op spmm-coo
python tests/diagnose_hipsparse_ref.py --op gather
python tests/diagnose_hipsparse_ref.py --op scatter

# 全部单点通过后再跑
python tests/diagnose_hipsparse_ref.py --op all
```

> 注意是**连字符** `spmv-csr`，不是下划线。

---

## 4. 确认分发确实选中了 hipSPARSE

这是最关键的一步——**跑通不等于走了 DCU 路径**。

```bash
PYTHONPATH=src python - <<'EOF'
import torch
from flagsparse.sparse_operations import _common as c
from flagsparse.sparse_operations import spmm_csr, spmm_coo, spgemm_csr, gather_scatter

print("ROCm      :", c._is_rocm_runtime())
print("hipSPARSE :", c._is_hipsparse_available(), "|", c._hipsparse_unavailable_reason())
i32 = torch.int32
print("spmv_csr :", c._spmv_csr_sparse_ref_backend(torch.float32, i32))
print("spmv_coo :", c._spmv_coo_sparse_ref_backend(torch.float32, i32))
print("spmm_csr :", spmm_csr._spmm_csr_sparse_ref_backend(torch.float32, i32, i32))
print("spmm_coo :", spmm_coo._spmm_coo_sparse_ref_backend(torch.float32, i32))
print("spgemm   :", spgemm_csr._spgemm_csr_sparse_ref_backend(torch.float32, i32, i32, i32, i32))
print("gather   :", gather_scatter._gather_scatter_sparse_ref_backend(torch.float32, i32, "gather"))
print("scatter  :", gather_scatter._gather_scatter_sparse_ref_backend(torch.float32, i32, "scatter"))
EOF
```

每行应为 `('hipsparse', None)`。返回 `(None, '<原因>')` 说明该算子在这台机器上没有厂商基线，
原因字符串会直接告诉你缺哪个符号。

各算子所需的 hipSPARSE 符号（缺任何一个都会被跳过并写明）：

- **通用**：`hipsparseCreate` / `hipsparseDestroy`
- **SpMV**：`hipsparseCreateCsr`、`hipsparseCreateCoo`、`hipsparseCreateDnVec`、
  `hipsparseSpMV`、`hipsparseSpMV_bufferSize`
- **SpMM**：`hipsparseCreateDnMat`、`hipsparseSpMM`、`hipsparseSpMM_bufferSize`、
  `hipsparseSpMM_preprocess`
- **SpGEMM**：`hipsparseSpGEMM_createDescr`、`hipsparseSpGEMM_workEstimation`、
  `hipsparseSpGEMM_compute`、`hipsparseSpGEMM_copy`、`hipsparseSpMatGetSize`、
  `hipsparseCsrSetPointers`
- **gather/scatter**：`hipsparseCreateSpVec`、`hipsparseCreateDnVec`、
  `hipsparseGather` / `hipsparseScatter`

---

## 4.5 算子覆盖范围（哪些有 hipSPARSE 基线，哪些没有）

**有 hipSPARSE 厂商基线的算子（9 个）**——DCU 分支实现的全部内容，已 1:1 合入
（96 个 hip 相关函数逐一比对，零遗漏）：

| 算子 | hipSPARSE 入口 | 备注 |
| --- | --- | --- |
| SpMV CSR | `hipsparseSpMV` | 支持 non/trans/conj |
| SpMV COO | `hipsparseSpMV` | 支持 non/trans/conj |
| SpMM CSR | `hipsparseSpMM` | **仅 op=non**，见第 7 节 |
| SpMM COO | `hipsparseSpMM` | 全 op（op 调用前已物化） |
| SpGEMM CSR | `hipsparseSpGEMM_*` | 两阶段 workEstimation/compute |
| gather | `hipsparseGather` | SpVec 原生 gather |
| scatter | `hipsparseScatter` | SpVec 原生 scatter |
| **SpSV CSR** | `hipsparseSpSV_*` | 三角求解，支持 non/trans/conj |
| **SpSM CSR** | `hipsparseXcsrsm2_*` | 三角矩阵-稠密矩阵求解 |

**间接获得基线（1 个）**：`spmm_csr_opt_alg2` 本身没有 hipSPARSE 代码，
但其基准已改为委托 `_benchmark_spmm_csr_sparse_ref`，因此在 DCU 上会复用 SpMM CSR 的
hipSPARSE 基线。

**两个公开 API 也已按后端分发**：`cusparse_spmv_gather` / `cusparse_spmv_scatter`
（名字里的 `cusparse` 是历史命名，未改）在 ROCm 上会走 hipSPARSE 原生 SpVec，
在 CUDA 上仍是原来的 selector-matrix + cuSPARSE SpMV。

> DCU 分支原本是把这两个函数**整体替换**成 hipSPARSE-only、并在非 ROCm 上直接 `raise`。
> 那样会删掉 CUDA 实现，所以本次改为分发而非替换。若不加这一层，DCU 上这两个 API
> 不会报错，但会静默退化到 `torch.sparse` 回落——**能跑，但测出来的不是厂商库性能**。

**DCU 上仍无厂商基线的算子**——DCU 分支本身就没有做，本次合并没有凭空补：

| 算子 | 基线依赖 | 在 DCU 上的预期表现 |
| --- | --- | --- |
| SDDMM CSR | `torch.sparse.sampled_addmm` | **不依赖 CuPy**，ROCm 版 PyTorch 若支持则可正常出数，需实测 |


| SpMV CSC | CuPy（5 处） | 基线列 `N/A` |
| SpMV BSR | CuPy（12 处） | 基线列 `N/A` |
| SpMM BSR | CuPy（8 处） | 基线列 `N/A` |
| alpha_spmm_alg1 | CuPy（2 处） | 基线列 `N/A` |

> 其中 **SpMV CSC / SpMV BSR / SpMM BSR 三个模块在 DCU 分支上根本不存在**
> （是上游后来新增的），所以 DCU 分支不可能有它们的 hipSPARSE 实现。

**重要**：上表"无厂商基线"**不代表算子在 DCU 上不能用**。Triton 内核照常运行，
正确性仍由 `torch.sparse` 参考校验；只是少了一个厂商性能对照列。
这些算子在 DCU 上应当 **PASS 但基线列为 `N/A`**——若出现 FAIL，那是内核问题，与基线无关。

---

## 4.55 SpMV 在 DCU 上走的是不同内核（rowpar）

SpMV CSR 是**唯一按后端切换内核实现**的算子（其余算子两个后端共用内核体）：

| 运行时 | 默认内核 | 策略 |
| --- | --- | --- |
| CUDA | `_spmv_csr_segbin_kernel` | 按 nnz 均匀切块 + 二分定位行 + `associative_scan` |
| **DCU/ROCm** | `_spmv_csr_real_kernel` | **一行一 program + 行内分段循环**（DCU 分支调优版） |

选择在 `_spmv_csr_default_backend()`，可用环境变量强制以做 A/B：

```bash
FLAGSPARSE_SPMV_CSR_KERNEL=segbin   # 强制 CUDA 版内核
FLAGSPARSE_SPMV_CSR_KERNEL=rowpar   # 强制 DCU 版内核
```

**两条路都是后端中立的通用 Triton 代码**，在两个后端都能跑——切换只是默认值不同。
segbin 未在 DCU 上调过参（`BLOCK` 固定 256），rowpar 未在 CUDA 上调过参。
建议在 DCU 上两个都跑一遍再定：

```bash
FLAGSPARSE_SPMV_CSR_KERNEL=rowpar python tests/test_spmv.py <dir/> --csv-csr rowpar.csv
FLAGSPARSE_SPMV_CSR_KERNEL=segbin python tests/test_spmv.py <dir/> --csv-csr segbin.csv
```

`use_opt=True` 的 bucket 路径另有一套设备属性调优（`_spmv_opt_bucket_configs` /
`_clip_spmv_opt_launch_spec`）：HIP 上换用 `_SPMV_OPT_BUCKET_CONFIGS_HIP*` 分档，
并把 `num_warps` 上限压到 8、`block_size` 上限压到 512。CUDA 上实测为恒等变换。

---

## 4.6 内核层：内核体同源，仅启动参数按后端特化

**除 SpMV 外**（见 4.55），Triton 内核体在两个后端是同一份代码，没有合并 DCU 版内核体；
启动参数另按后端特化，见下。
这个结论是逐函数比对验证过的，不是推断：

- DCU 分支相对其 merge-base 新增的 9 个内核（alg1 / alg2 系列），
  **全部已通过上游进入本仓库**；
- 其中 6 个与 DCU 版仅有 ruff 换行差异，语义一致；
- 另外 3 个（`_spmm_csr_alg2_{segmented,row,batched}_rows_kernel`）
  **本仓库是更新的版本**——多了 `ACCURACY: tl.constexpr` 参数，并把标量
  `tl.static_range` 循环向量化为 2D 张量加载（上游 `66f30a6` 引入）。
  DCU 分支是旧版（该文件 1149 行 vs 本仓库 1389 行）。**合 DCU 的内核是倒退。**

### wavefront=64 与内核启动调优（2026-08 已部分解决）

> **本节已更新。** 早期版本称"DCU 分支没有任何内核适配"，那是基于 2026-07 快照的结论。
> DCU 分支在 `dcu_tuning` 系列提交中补上了 **SpMM 的启动参数特化**，已合入本仓库：
> `_spmm_is_hip_device()` 守卫的 11 处分支，在 DCU 上把 `num_warps` 上限压到 **8**
> （CUDA 是 16）、线程上限压到 **512**、tile 宽度压到 `block_n ≤ 64`、
> `num_stages` 固定为 **1**。这些分支在 CUDA 上全部是恒等变换（已逐项验证）。

**wavefront=64 的适配情况按算子而异。** AMD 的 wavefront 通常是 **64**，NVIDIA warp 是 32。
**SpSV 已适配**（DCU ALG3 复用 CUDA ALG8 的 NNZ-balanced 算法，见下方）；
**SpMM/alg1 尚未适配**。下表按路径区分：

| 路径 | `warp_size` 来源 | wavefront=64 时 |
| --- | --- | --- |
| `spmm_csr_opt_alg2` | `getattr(props, "warp_size", 32)`，**读设备属性** | 自动按 64 算 block 尺寸；能工作，但这套启发式从未在 64 宽硬件上调过参 |
| `alpha_spmm_alg1`<br>`spmm_csr` 的 alg1 路径 | `_select_alpha_spmm_alg1_warp_and_factor()`，**按 `n_dense_cols` 硬编码返回 32/16/8/4** | ⚠️ **上限就是 32**，与设备无关——内核里 `lane_offsets = tl.arange(0, WARP_SIZE)` 最多只用 32 条 lane，**64 宽硬件上有一半 lane 闲置** |

先确认这台机器报告的值：

```bash
PYTHONPATH=src python -c "
import torch
p = torch.cuda.get_device_properties(0)
print('warp_size =', getattr(p, 'warp_size', '<无该属性>'))
print('max_threads_per_mp =', p.max_threads_per_multi_processor)
"
```

- 报告 **64** → `alg2` 会自适应，但 **alg1 系列仍卡在 32**，这是已知的性能上限。
- 属性**缺失** → `alg2` 回落到默认 32，在 64 宽硬件上就是错配。

这属于"**待调优**"而非"合并遗漏"——DCU 分支本身也没解决。
DCU 上若 alg1 相关算子性能明显偏低，先查这里，不要怀疑是合并出了问题。

### SpSV 在 DCU 上的共享执行策略（待实机验收）

SpSV 现在按与 SpMM/SpMV 相同的原则组织：尽量复用算法和 kernel 主体，但允许 ROCm
为 wavefront、前向推进和 launch 方式保留少量专用实现，不要求两个后端逐行完全相同。

| 行为 | DCU/ROCm | CUDA |
| --- | --- | --- |
| NON_TRANS lower 默认路由 | ALG3 `csr_nnz_balance`；环境变量可退回 ALG1 | 保持原 AUTO 选择 |
| CW kernel | 同一个 `_spsv_csr_cw_kernel`，CU-capped persistent launch | 同一个 `_spsv_csr_cw_kernel` |
| CW worker 数 | 不超过 CU 数 | 按矩阵并行度选择 |
| TRANS/CONJ kernel | 共享 TRANS kernel，串行 | 共享 TRANS kernel，多 worker |
| 拓扑分析 | ALG2 使用 CU-capped persistent 分析 | 原 GPU level-analysis |
| ALG2 | 逐 level 的标量 row kernel | 原 level-scheduled kernel |
| ALG3 | `csr_nnz_balance`：复用 CUDA ALG8 的数学 kernel，DCU 使用不超过 CU 数的常驻 workgroup | 原 32-lane ROC kernel |
| ALG4 | 不提供（显式请求会报 CUDA-only） | 原 `csr_smblk` |
| ALG8 | 不提供；其算法已作为 DCU ALG3 | 原 `csr_nnz_balance` |

`SERIAL_EXECUTION` 是 launch-time constexpr：DCU 路径按三角顺序逐行执行，kernel 编译时
会消掉 ready-flag 轮询和依赖计数；CUDA 路径保留原来的 acquire/release 并行同步。
这种差异属于启动/同步策略，不是两套数学算法。

DCU ALG3 的 analysis 与 CUDA ALG8 一样生成 `csr_row_idx` 和 `in_degree`，但预处理按
DCU 的 64-lane wavefront 配置（CUDA 保持 32）。solve 复用同一套 ALG8 数学 kernel；
CUDA 仍编译为一-NNZ-一-program、`num_warps=1`。DCU 则把 grid 限制为不超过 CU 数，
每个常驻 workgroup 默认用 256 个 NNZ lane，并按递增 NNZ 块继续取任务；依赖通过
GPU-scope acquire/release 发布。这既避免海量等待 program 把生产者堵在调度队列中，
又恢复上一版的向量吞吐。`FLAGSPARSE_SPSV_ROCM_ALG3_BLOCK_NNZ=1|64|128|256`
可用于实机 A/B，默认 256；复验通过前仍不应把“无 hang”标记为已验收。

当前性能版还保持生产者/消费者对 `x` 的 GPU-scope 原子发布协议，但不再把
float32/complex64 的 `tmp_sum` 提升到双精度；此前双精度也出现过相同的大误差，说明
根因是可见性而不是舍入精度。DCU 下三角路径直接使用 identity `val_id`，不再分配或
读取 nnz 长度的 `launch_order`。`x/tmp_sum/ready/indegree` 使用框架原生的
fill/copy kernel 初始化一次；实测它比合并的 Triton 初始化 kernel 更快，尤其是复数类型。
CUDA 继续使用原有显式 launch order 和初始化路径。
NNZ-balanced solve 使用 GPU-scope release/acquire 原子链发布 `tmp_sum`、
`indegree`、`x` 和 `ready`；不再使用 256-thread workgroup barrier，避免四个
wavefront 因无关依赖互相等待。CU 数量封顶保持不变，因此不会恢复到全网格
自旋等待的旧调度方式。

可用 `FLAGSPARSE_SPSV_ROCM_ENABLE_ADVANCED_AUTO=0` 强制 AUTO 使用 ALG1。
但 0.8 加速比必须以 gfx936 和正式矩阵集的 CSV 为准，静态检查不能代替性能验收。

---

## 5. 正确性套件

```bash
PYTHONPATH=src python -m pytest tests/pytest -q
```

**已知的既有失败（与 DCU 无关，CUDA 上同样失败）**，看到这三个不用查：

- `test_spsv_sell_accuracy.py::test_spsv_sell_non_unit_rejects_malformed_structure[duplicate_diagonal]`
  —— 确定性失败，已在改动前的干净树上复现
- `test_spmv_coo_accuracy.py::test_spmv_coo_tocsr_matches_torch[...]`
- `test_spmv_csc_accuracy.py::test_spmv_csc_matches_dense_reference[...]`
  —— 后两个是容差抖动：**每次失败的 dtype 参数都不一样**（一次 `complex64`、一次 `float32`），
  且单独跑就过。参数固定不变的失败才是真问题。

CUDA 基准线：`1613 passed / 3 failed`。

```bash
# 策略/契约类测试（不需要 GPU，秒级）
python -m pytest tests/ci -q     # 期望 39 passed / 3 skipped
```

---

## 6. 逐算子基准

`tests/data/` 里自带三个矩阵，先用最小的过一遍：

```bash
export PYTHONPATH=$PWD/src
M=tests/data/trdheim.mtx

python tests/test_spmv.py      $M --warmup 2 --iters 5
python tests/test_spmv_opt.py  $M --warmup 2 --iters 5
python tests/test_spmm.py      $M --warmup 2 --iters 5
python tests/test_spmm_opt.py  $M --warmup 2 --iters 5
python tests/test_spgemm.py    $M --warmup 2 --iters 5
python tests/test_spmm_coo.py  $M --warmup 2 --iters 5

python tests/test_spmv_coo.py --synthetic --dtypes float32 --ops non --warmup 2 --iters 5
python tests/test_gather.py   --value-dtypes float32 --warmup 3 --iters 10
python tests/test_scatter.py  --value-dtypes float32 --warmup 3 --iters 10
```

SpSV 先逐算法跑 CSR/COO，既能定位 hang，也能比较 allinone 风格的多路径：

```bash
mkdir -p results
for ALG in 1 2 3 4 8; do
  python tests/test_spsv.py "$M" --csv-csr "results/spsv_csr_alg${ALG}.csv" \
    --ops NON --value-dtypes float --index-dtypes int32 --alg-num "$ALG" \
    --warmup 2 --iters 10
  python tests/test_spsv.py "$M" --csv-coo "results/spsv_coo_alg${ALG}.csv" \
    --ops NON --value-dtypes float --index-dtypes int32 --alg-num "$ALG" \
    --warmup 2 --iters 10
done
```

先看 `FlagSparse_ms` 是否稳定且程序能退出，再看
`FlagSparse_vs_vendor_speedup`；二者都按每轮完整的 analysis/preparation+solve 统计，
不会只摊销 FlagSparse 的 analysis。`vendor_backend` 在 DCU 上为 `hipSPARSE`，在 CUDA
上为 `cuSPARSE`。正式验收时应使用老师规定的迭代数和矩阵集，不能只用
`trdheim.mtx` 推断全量是否达到 0.8。

**怎么看结果**：SpSV 会按运行时自动显示 `HIP.ms` 或 `CU.ms`；CSV 分别写入
`hipSPARSE_ms` 或 `cuSPARSE_ms`，不再用固定的 CUDA 名称表示 DCU 基线。

- 有数值 → hipSPARSE 基线跑通了 ✅
- `N/A` → 基线没拿到，去 `vendor_reason` 字段看测试封装的具体原因

带 `--no-cusparse` 的 harness（`test_spmv`、`test_spmm`、`test_spgemm`、`test_gather`、
`test_scatter`、`test_spmm_coo`）可以用该参数先把基线关掉，单独确认 Triton 内核本身没问题：

```bash
python tests/test_spmv.py $M --no-cusparse --warmup 2 --iters 5
```

---

## 7. 已知限制（不是 bug，不用查）

- **hipSPARSE 的 SpMM 入口只支持非转置**。因此 CSR SpMM 的 `op=trans` / `op=conj`
  会直接跳过并给出：
  `hipSPARSE CSR SpMM reference covers op=non only; trans/conj skipped`
  这是有意为之——拿一个形状不同的运算去对比毫无意义。
  **COO SpMM 不受影响**，因为 op 在调用前已被物化。
- **fp16 / bf16 没有厂商基线**。CuPy 和 hipSPARSE 的稀疏矩阵都不支持这两种 dtype，
  两个后端上这一列都是 `N/A`，回落到 `torch.sparse` 参考。这在 CUDA 上就是如此。
- **`test_spmm` 的 CU 列会在阈值附近 FAIL/PASS 乱跳**。CUDA 上改动前的误差跨度是
  0.84~2.07（判定阈值 1.0），这是既有的 fp32 与厂商库逐元素比较的容差问题，不是后端引入的。
- **SpGEMM 在超大矩阵上会触发 rocSPARSE 的显存非法访问（VMFault）**。2026-08 实测，
  `mip1.mtx`（66463×66463，nnz≈1035 万，A_EQUALS_B 自乘）跑到参考实现阶段时进程被
  `SIGABRT` 打死（returncode `-6`）：
  ```
  Invalid address access: 0x7f341574d000, Error code: 3.
  >>>>>>>> KERNEL VMFault !!!! <<<<<<
  kernel name: _ZL23csrgemm_fill_wf_per_rowILj256ELj16ELj32ELj137EiiffE...
  ```
  故障内核 `csrgemm_fill_wf_per_row` 是 **rocSPARSE 内部的 SpGEMM 填充内核，不是
  FlagSparse 的 Triton 内核**，所以不在本仓库的修复范围内，排查时不要往 Triton 侧找。
  GPU 页错误是 `SIGABRT`，Python 的 `except BaseException` 拦不住，进程直接消失。
  已做的规避：`tests/test_spgemm.py` 改成**逐矩阵 flush + fsync 写 CSV**，崩溃前已完成的
  矩阵结果得以保留；同时写 `<csv>.inflight.json` 记录正在处理的项，正常跑完才删除，
  崩溃后该文件的 `last_completed` 的**下一个**矩阵即为触发者。
  配合 `run_flagsparse_pytest.py` 里状态不被部分产物覆盖的修复，这类崩溃现在表现为
  「部分性能数据 + `Failed` 状态 + 指名凶手」，而不是「整轮数据全丢」。
- **SpSM 在 DCU 上仍有 GPU 内核死锁风险**（2026-08 实测，gfx936）。Python 层正常返回，
  hang 在 `torch.cuda.synchronize()`，16×16 的矩阵跑 15 分钟也不结束。根因是内核里
  跨 program 的裸自旋等待：消费者 program 占住 CU，生产者排不进去，flag 永远不会
  被置位。这是内核层问题，与 hipSPARSE 参考层无关。
- **SpSV CSR/COO 已加入 DCU 安全并行路径，等待 DCU 实机复验。** DCU 公开 ALG1/2/3；
  ALG3 是复用 CUDA ALG8 数学逻辑的 `csr_nnz_balance`，ALG4/ALG8 仅保留在 CUDA。
  DCU analysis 采用 64-lane wavefront，solve 复用 CUDA ALG8 数学 kernel，并把常驻
  worker 数限制在 CU 数以内，以减少全量 NNZ program 的调度停滞风险。
  `FLAGSPARSE_SPSV_ROCM_ENABLE_ADVANCED_AUTO=0` 可强制 AUTO 回退 ALG1，默认值为 1。
  复验通过前不要把“无 hang”或加速比标记为已验收；
  建议先用小矩阵逐文件运行 `test_spsv_csr_accuracy.py` 和
  `test_spsv_coo_accuracy.py`，再用正式矩阵集测 ALG1/2/3 与 hipSPARSE 的比值。

---

## 8. 统一运行器

单点都通过后，用统一运行器跑全量：

```bash
PYTHONPATH=src python run_flagsparse_pytest.py --list-ops          # 先看有哪些算子
PYTHONPATH=src python run_flagsparse_pytest.py --ops spmv_csr --phase both
PYTHONPATH=src python run_flagsparse_pytest.py --mode quick --phase accuracy
```

常用参数：`--ops`（逗号分隔）、`--phase {accuracy,performance,both}`、
`--mode {quick,normal}`、`--gpus`、`--results-dir`、`--timeout`。

---

## 9. 排查速查表

| 现象 | 首先检查 |
| --- | --- |
| 基线列全是 `N/A` | `flagsparse.__file__` 是否指向 `src/`（见第 2 节）；再看 `reason` 字段 |
| `requires a ROCm runtime` | `torch.version.hip` 是否为 `None`——装的可能是 CUDA 版 torch |
| `No module named 'hip'` | `pip install hip-python` |
| `... is unavailable: missing hipsparseXxx` | hip-python 版本与 ROCm 不匹配，换匹配版本 |
| **程序卡住不动** | `diagnose_hipsparse_ref.py --op <算子>`，最后打印的阶段就是卡点 |
| 计时数字异常大 | 先跑 `--timing-only` 确认 HIP 事件链正常 |
| 显存持续增长 / 崩溃 | 怀疑 `_prepare_*_ref_hipsparse` 的描述符释放；这部分**从未在真机跑过**，是最高风险区 |
| alg1 系列算子性能偏低 | wavefront=64 但 alg1 硬编码 32 lane，见第 4.6 节——是已知调优项，不是合并问题 |
| SpMV 结果或性能异常 | 先用 `FLAGSPARSE_SPMV_CSR_KERNEL=segbin` A/B，确认是内核选择问题还是别的（见 4.55） |
| gather/scatter 基线比预期慢 | 确认走的是 hipSPARSE 而非 `torch.sparse` 回落（见第 4 节分发探测） |
| SpSV/SpSM 基线列为 `N/A` | 查 `_hipsparse_spsv_skip_reason` / `_hipsparse_csrsm2_skip_reason` 的返回原因；csrsm2 是旧版 API，部分 ROCm 可能未导出 |

---

## 10. 风险提示：哪些代码是真的没跑过

合并是在 CUDA 机器上做的，以下内容**只经过静态检查**，请重点观察：

1. `_prepare_* / _run_* / _destroy_*_ref_hipsparse` 里的 **ctypes 指针与描述符生命周期**
   ——悬垂指针、内存泄漏、重复释放这类问题只会在真机暴露。
2. 移植时做的**文本级改名**（原分支的 `_common_mod.X` 改成了直接 `X`）。
3. **hipSPARSE 枚举查找**是否与你这台机器的 ROCm 版本对得上
   （`_hipsparse_lookup` / `_hip_lookup` 会在找不到时给出明确报错，不会静默）。

建议：先用 `diagnose_hipsparse_ref.py` 单点确认，再跑小矩阵，最后才上全量和大矩阵。

**反过来说，以下几项已经排除，不必怀疑：**

- **Triton 内核**——两个后端同一份代码，未合并任何 DCU 版内核，且本仓库版本比 DCU 分支更新（见 4.6）。
- **CUDA 侧回归**——合并全程在 CUDA 机器上验证：`tests/ci` 39 passed，
  `tests/pytest` 1613 passed / 3 failed（均为既有失败），十个 harness 全通，
  并覆盖了 SpMV 六种 dtype、SpMV/SpMM 的 non/trans/conj、gather/scatter 各 40 例。
- **hipSPARSE 参考层的完整性**——DCU 分支的 96 个 hip 相关函数已逐一比对，零遗漏
  （2026-08 第二轮合并后；含 SpSV 16 个、SpSM 10 个、HIP 事件/流 5 个、SpMM 启动特化）。
