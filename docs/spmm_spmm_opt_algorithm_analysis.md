# FlagSparse OF SpMM / SpMM-opt 算法解析

## 1. 文档目标

本文面向 `flagsparse_of` 当前代码，完整梳理下面三类 SpMM 路线：

- `flagsparse_spmm_csr`
- `flagsparse_spmm_csr_opt`
- `flagsparse_spmm_coo`

目标不是抄源码，而是把下面几件事讲清楚：

- 每条路线在算什么
- 为什么这样分派
- 什么矩阵会更适合哪条路线
- 当前实现的性能与数值稳定性取舍
- 诊断路径和公开路径的真实差异

本文只基于静态代码阅读与仓库内现有诊断 CSV 做分析，不声称运行了新的 CUDA/Triton correctness、benchmark 或 `.mtx` 批量实验。

## 2. 代码入口与支持边界

### 2.1 主要源码位置

- `src/flagsparse/sparse_operations/spmm_csr.py`
- `src/flagsparse/sparse_operations/spmm_coo.py`
- `src/flagsparse/__init__.py`
- `src/flagsparse/sparse_operations/__init__.py`

### 2.2 公开 API

| API | 格式 | 当前公开定位 | dtype 支持 |
| --- | --- | --- | --- |
| `flagsparse_spmm_csr` | CSR | 基础版 SpMM | `float16/bfloat16/float32/float64/complex64/complex128` |
| `prepare_spmm_csr_opt` | CSR | `spmm_opt` 预处理缓存 | 跟输入稀疏矩阵一致 |
| `flagsparse_spmm_csr_opt` | CSR | 优化版 SpMM，面向重复调用 | `float32/float64` |
| `flagsparse_spmm_coo` | COO | 原生 COO SpMM，默认走 `rowrun` | `float16/bfloat16/float32/float64/complex64/complex128` |

### 2.3 内部 diagnose / experimental 路线

这些函数存在于代码中，但不是正式公开 API：

- `_triton_spmm_csr_impl_opt_prepared_stable`
- `_flagsparse_spmm_csr_opt_stable_for_diagnose`
- `_triton_spmm_csr_impl_opt_prepared_candidate`
- `_flagsparse_spmm_csr_opt_candidate_for_diagnose`
- `spmm_coo` 的内部 `atomic` route

### 2.4 总体对比

| 路线 | 输入组织 | 并行粒度 | 累加方式 | 适用矩阵画像 | 主要风险 |
| --- | --- | --- | --- | --- | --- |
| CSR base | `indptr/indices/data`，行天然分组 | 一个 program 负责一行和一个 dense tile | 普通顺序求和，实数路径按 dtype 选 `fp32/fp64` 累加，复数同理 | 行长度相对均匀，或先要一个简单稳定的通用基线 | 超长行拖慢单 program；只按行做负载均衡 |
| CSR opt batched | CSR + row bucket | 一个 program 串行处理多条短行 | 普通顺序求和 | 大量短行、平均行长低 | 行间工作量仍可能不均；对 `dense_cols` 不敏感 |
| CSR opt vector | CSR + row bucket | 一行一个 program | 普通顺序求和 | 中长行，单行工作量足够大 | 极长行会拖垮单 program |
| CSR opt split | CSR + long-row part metadata | 一个 part 一个 program，再二次 reduce | 先局部求和再归约 | 少量超长行、长尾矩阵 | workspace 额外访存与二次 reduce 开销 |
| COO rowrun | canonical COO，同行连续 | 一段 row-run 一个 program | 普通顺序求和 | 已排序/可排序、重复项需要先 coalesce、想避免原子冲突 | 预处理要排序/合并；不适合直接吃原始乱序数据 |
| COO atomic | canonical/非 canonical 都能表达 | 一个非零 x 一个 dense 列 | `atomic_add` | 通用 fallback / 对照路线 | 原子冲突、顺序不确定、性能和数值稳定性波动更大 |

## 3. SpMM 的统一问题模型

SpMM 要计算：

```text
C = A @ B
```

其中：

- `A` 是稀疏矩阵
- `B` 是稠密矩阵，形状是 `(K, N)`
- `C` 是输出稠密矩阵，形状是 `(M, N)`

对任意输出元素：

```text
C[i, j] = sum(A[i, k] * B[k, j]) over all nonzero k in row i
```

核心难点不在公式，而在于：

- 稀疏矩阵每行非零数差异非常大
- 不同存储格式对访问局部性影响很大
- `N` 的大小会影响一次 dense tile 该做多宽
- 求和顺序会直接影响浮点误差

因此当前实现的本质，是在不同输入分布下寻找更好的三件事：

- 谁来负责一行
- 一次负责多少列
- 长行如何拆，短行如何并

## 4. CSR 基础版 SpMM：`flagsparse_spmm_csr`

### 4.1 设计定位

`flagsparse_spmm_csr` 的注释写得很明确：它是 AlphaSparse CSR ALG1 (`csrspmm_rb_sr`) 的 Triton 映射版本，面向：

- CSR 输入
- 非转置 `A @ B`
- `B/C` 按 row-major 使用

这里可以把它理解为“简单、清晰、通用的行驱动基线”。

### 4.2 输入组织

CSR 三元组：

- `indptr[row]` 到 `indptr[row + 1]` 给出该行非零区间
- `indices[idx]` 给出每个非零的列号
- `data[idx]` 给出每个非零值

这天然适合“按行处理”。某一行所有非零在内存中连续，因此基础版直接沿行扫描。

### 4.3 调度启发式

`_select_spmm_alg1_warp_and_factor(n_dense_cols)` 会按 `B.shape[1]` 选择一个粗粒度宽度：

| `n_dense_cols` 条件 | `warp_size` | `factor` | 默认 `block_n` |
| --- | ---: | ---: | ---: |
| `> 64` | 32 | 4 | 128 |
| `> 32` | 32 | 2 | 64 |
| `> 16` | 32 | 1 | 32 |
| `> 8` | 16 | 1 | 16 |
| `> 4` | 8 | 1 | 8 |
| `<= 4` | 4 | 1 | 4 |

`_resolve_spmm_alg1_launch_config(...)` 进一步给出：

- `block_n = warp_size * factor`
- `block_nnz = warp_size`

这意味着基础版的默认策略是：

- 输出列 tile 宽度主要由 `dense_cols` 决定
- 稀疏维度上的一次 chunk 长度主要取决于 `warp_size`

### 4.3.1 `block_n` 和 `block_nnz` 不是一回事

这两个名字很像，但它们切的是两个不同维度：

- `block_n`：切的是输出矩阵 `C` 的列维，一次处理多少个 `dense cols`
- `block_nnz`：切的是某一行稀疏非零的扫描块，一次拿多少个 `nnz` 进入内层循环

从 kernel 视角可以把一次计算想成：

```text
固定 row
固定 dense tile: C[row, offs_n : offs_n + block_n]
沿这一行的 nnz 做归约，但按 block_nnz 分 chunk 扫描
```

所以：

- `block_n` 决定 `offs_n = pid_n * BLOCK_N + arange(0, BLOCK_N)` 这一块输出列的宽度
- `block_nnz` 决定 `for chunk_start in range(0, row_nnz, BLOCK_NNZ)` 一次扫多少个非零

它们不要求相等，也通常不会相等。

当前代码里它们的来源也不同：

- 基础版默认 `block_n = warp_size * factor`，`block_nnz = warp_size`
- `spmm_opt` 中 `block_n` 只由 `n_dense_cols` 决定，而 `block_nnz` 由 bucket 决定

因此它们是两个完全不同的调优旋钮：

- `block_n` 偏向 dense 维吞吐和 tile 宽度
- `block_nnz` 偏向 sparse 维扫描粒度、循环展开和寄存器压力

### 4.3.2 什么叫“切块”，到底在切什么

先把 SpMM 想成一个三维问题：

```text
输出位置：C[row, n]
row 维：哪一行
n 维：输出矩阵的列，也就是 dense cols
k 维：归约维，也就是这行里所有 nnz 对应的列索引
```

当前 CSR kernel 固定住 `row` 之后，还剩两个方向要处理：

- `n` 方向：一次算多少列输出
- `k/nnz` 方向：一次扫这一行里的多少个非零

于是就有了两种“切块”：

```text
block_n   -> 切 n 方向
block_nnz -> 切 nnz / k 方向
```

可以把它画成下面这样：

```text
对某一条固定 row 来说：

输出列方向（dense cols）:
0         4         8         12
|----tile0----|----tile1----|--tile2--|
<---block_n--->

这条 row 的 nnz 列表:
[nnz0 nnz1 nnz2 nnz3 nnz4 nnz5 nnz6 nnz7 nnz8 nnz9]
|------chunk0------|------chunk1------|--chunk2--|
<----block_nnz----->
```

这就是为什么要同时有两个 block 参数：

- 一个控制“横着一次算多少列”
- 一个控制“顺着这行一次扫多少个非零”

### 4.3.3 `block_n` 的直观例子

假设：

- `n_dense_cols = 10`
- `block_n = 4`

那么输出列不会一次把 10 列全算完，而是切成：

```text
tile0: cols [0, 1, 2, 3]
tile1: cols [4, 5, 6, 7]
tile2: cols [8, 9]
```

也就是说，某个 program 只负责其中一个 tile，例如：

```text
program(row=7, tile=1)
```

它只算：

```text
C[7, 4], C[7, 5], C[7, 6], C[7, 7]
```

而不会碰这一行的其他输出列。

为什么要这么切：

- 一次把所有 `dense cols` 都放进寄存器太重
- `B[col, :]` 也不需要一次读完整行
- Triton 更适合做固定宽度的小 tile

所以 `block_n` 本质上是在控制“一个 program 的输出工作面有多宽”。

### 4.3.4 `block_nnz` 的直观例子

假设某一行有：

```text
row_nnz = 10
block_nnz = 4
```

那么不会一次把 10 个非零全塞进内层展开，而是分成：

```text
chunk0: nnz[0:4]
chunk1: nnz[4:8]
chunk2: nnz[8:10]
```

对应到代码就是：

```text
for chunk_start in [0, 4, 8]:
    处理这一小段 nnz
```

为什么要这么切：

- 一次展开太多 `nnz`，寄存器压力会很高
- 对短行、中行、长行，合适的循环粒度不同
- 分 chunk 后 kernel 模板更稳定，也更容易让编译器展开

所以 `block_nnz` 本质上是在控制“一个 program 在归约维上每口吃多大”。

### 4.3.5 两个 block 怎么一起工作

把上面两个例子合起来看。

假设：

- 当前正在处理 `row = 3`
- `n_dense_cols = 10`
- `block_n = 4`
- `row_nnz = 10`
- `block_nnz = 4`

那么一个 program 可能是：

```text
program(row=3, tile=1)
```

它负责：

- 输出列：`[4, 5, 6, 7]`
- 但要把 row 3 的全部 10 个非零都归约进去

它的执行过程是：

```text
acc[4 cols] = 0

扫 chunk0: nnz[0:4]
    把这 4 个 nnz 对应的 A[row, col] * B[col, 4:8] 加进 acc

扫 chunk1: nnz[4:8]
    再加进 acc

扫 chunk2: nnz[8:10]
    再加进 acc

最后把 acc 写回 C[row, 4:8]
```

这一步如果没想清楚，最容易误以为：

- `block_n` 和 `block_nnz` 是同一种块

其实不是。一个 program 的工作始终是：

- 横向只管一个输出列 tile
- 纵向要把这一行该 tile 的所有 nnz 都归约完

### 4.3.6 为什么 `n_dense_cols` 影响 `block_n`，而 `row_nnz` 影响 `block_nnz`

因为它们控制的是两个不同方向。

`block_n` 跟 `n_dense_cols` 强相关：

- `dense cols` 很少时，tile 太宽没有意义
- `dense cols` 很多时，tile 太窄会让 program 数量太多

`block_nnz` 跟行长度类型强相关：

- 短行没必要把 chunk 设太大
- 中长行更适合较大的 chunk
- 超长行甚至不能只靠 chunk，还要进一步 `split`

所以当前设计是合理分工：

- `block_n` 用来适配 dense 维
- `block_nnz` 用来适配 sparse 归约维

### 4.4 基础 kernel 在做什么

#### 实数路径：`_spmm_csr_real_kernel`

逻辑可以压缩成下面的伪代码：

```text
program(row, dense_tile):
    acc[BLOCK_N] = 0
    start = indptr[row]
    end = indptr[row + 1]
    for each sparse chunk in this row:
        for each nnz in the chunk:
            col = indices[idx]
            a = data[idx]
            acc += a * B[col, dense_tile]
    store acc -> C[row, dense_tile]
```

真正的并行粒度是：

- `program_id(0)` 对应一行
- `program_id(1)` 对应输出矩阵的一个 dense 列块

所以一个 program 的职责非常明确：

- 只负责一行
- 只负责这行输出中的一段列

#### 复数路径：`_spmm_csr_complex_kernel`

复数版做的不是新算法，只是把：

- `A = a_re + i a_im`
- `B = b_re + i b_im`

拆成实部和虚部分开累加：

```text
real += a_re * b_re - a_im * b_im
imag += a_re * b_im + a_im * b_re
```

这也是为什么 `mhd1280b.mtx` 这类 complex Hermitian 示例更适合作为“输入语义和 complex 路径”的说明样本，而不是 `spmm_opt` 的代表矩阵。

### 4.5 手工例子 1：CSR base 如何扫一行

设：

```text
A =
[ 2 0 3 0 ]
[ 0 4 0 5 ]
[ 6 0 0 7 ]

CSR:
indptr  = [0, 2, 4, 6]
indices = [0, 2, 1, 3, 0, 3]
data    = [2, 3, 4, 5, 6, 7]
```

再设：

```text
B =
[ b00 b01 b02 b03 ]
[ b10 b11 b12 b13 ]
[ b20 b21 b22 b23 ]
[ b30 b31 b32 b33 ]
```

假设 `BLOCK_N = 2`，某个 program 负责：

- `row = 1`
- dense tile 是列 `0:2`

那么它会读：

- `start = indptr[1] = 2`
- `end = indptr[2] = 4`

对应本行两项：

- `(col=1, val=4)`
- `(col=3, val=5)`

局部累加就是：

```text
acc[0] = 4 * b10 + 5 * b30
acc[1] = 4 * b11 + 5 * b31
```

再写回：

```text
C[1, 0:2] = acc
```

这就是基础版的核心：行扫描很直接，但所有该行工作都绑在一个 program 上。

### 4.6 为什么这种路线适合作为基线

优点：

- 思路简单，维护成本低
- CSR 行连续访问自然
- 对稀疏行的解释直观
- 支持范围最广，连复数都支持

缺点：

- 行长度不均时，一个 program 的工作量差异会非常大
- 如果矩阵有“极少数超长行”，这些行会成为拖尾
- 没有把短行合并，也没有把长行拆开

### 4.7 数值特征

- `float16/bfloat16` 最终会在更高精度上做一层累加后再写回
- `float32` 基础路径用 `tl.float32` 累加
- `float64` 用 `tl.float64` 累加
- complex 路径按对应实部/虚部精度累加
- 没有补偿求和，没有长行 special handling

因此基础版的数值行为可理解为：

- 通用
- 明确
- 但对长行、大量消去、超宽动态范围，不会主动做额外稳定化

## 5. CSR 优化版 SpMM：`flagsparse_spmm_csr_opt`

### 5.1 设计定位

`spmm_opt` 的核心不是“某一个更快的 kernel”，而是：

- 先预处理
- 再按行长度分桶
- 为不同 bucket 选择不同 kernel
- 对极长行单独切分

所以它本质上是一个小型 dispatch framework。

### 5.2 `prepare_spmm_csr_opt` 缓存了什么

`PreparedCsrSpmmOpt` 中重要字段如下：

| 字段 | 含义 |
| --- | --- |
| `data` / `kernel_indices` / `kernel_indptr` | 预处理后的 CSR 数据 |
| `shape` / `n_rows` / `n_cols` | 稀疏矩阵维度 |
| `row_lengths` | 每行的 `nnz` |
| `max_row_nnz` | 最大行长度 |
| `row_buckets` | 公开 `opt` 路线用到的分桶结果 |
| `long_part_rows` / `long_part_starts` / `long_part_ends` | 长行被切成若干 part 后的元数据 |
| `long_row_ids` / `long_row_part_ptr` | 哪些原始行被 split，以及每行对应哪些 part |
| `supports_opt` | 是否支持 `opt`，当前只对 `float32/float64` 为真 |
| `long_row_fallback_only` | 当前代码里保留的状态位，但没有真正驱动调度策略 |

这个对象的真正价值是：同一个稀疏矩阵如果会重复乘很多不同 `B`，那么分桶和长行拆分元数据只需要做一次。

### 5.3 `spmm_opt` 的公开分桶表

`_SPMM_OPT_BUCKET_SPECS` 给出的公开路线是：

| 行长度范围 | kind | `batch_rows` | `block_nnz` | 解释 |
| --- | --- | ---: | ---: | --- |
| `<= 32` | `batched` | 8 | 32 | 很短的行，合并多行一起吃 |
| `33 ~ 128` | `batched` | 4 | 64 | 仍然偏短，但每行更重，批量缩小 |
| `129 ~ 512` | `vector` | 1 | 128 | 中行，改成一行一个 program |
| `513 ~ 2048` | `vector` | 1 | 128 | 长行，仍按一行一个 program |
| `> 2048` | `split` | 1 | 256 | 超长行，必须拆 part |

### 5.4 `spmm_opt` 的 dense 维宽度选择

`_select_spmm_opt_block_n(n_dense_cols)` 的规则是：

| `n_dense_cols` | `block_n` |
| --- | ---: |
| `<= 8` | 8 |
| `<= 16` | 16 |
| `<= 32` | 32 |
| `<= 64` | 64 |
| `> 64` | 128 |

注意这里和基础版不同：

- 基础版用 `warp_size * factor`
- `spmm_opt` 直接用更离散、固定的 `8/16/32/64/128`

这说明 `spmm_opt` 更强调 bucket 化后的稳定模板，而不是继续沿用基础版的 ALG1 宽度推断逻辑。

### 5.5 为什么短行适合 `batched`

如果一行只有 4 到 10 个非零，那么“一行一个 program”会出现两个问题：

- program 数量太多，调度开销高
- 每个 program 真的做的乘加很少，吞吐利用率不高

这时让一个 program 串行处理多条短行，通常更划算。当前实现里：

- `<=32` 用 `BATCH=8`
- `33~128` 用 `BATCH=4`

原因很直观：

- 行越短，越能塞更多行
- 行越长，每个 program 内部串行负担越重，因此批量必须缩小

### 5.6 手工例子 2：batched 为什么适合短行

设 8 条行，每行都只有 6 个非零，`dense_cols = 32`。

如果用 vector 路线：

- 8 条行需要 8 个 program
- 每个 program 只做很少的 sparse 访问与乘加

如果用 batched 路线：

- 一个 program 可以串行处理其中 8 条行
- 每次 dense tile 都能复用类似的执行模板
- 更接近“把很多零碎小活打包做”

从 CPU 类比也好理解：8 个特别短的小循环，合并起来通常比拆成 8 个完全独立的任务更划算。

### 5.7 `batched` kernel 在做什么

`_spmm_csr_batched_rows_f32_kernel` / `_f64_kernel` 的模式是：

```text
program(batch_id, dense_tile):
    for batch_idx in 0..BATCH-1:
        row = rows[batch_id * BATCH + batch_idx]
        acc = 0
        scan this row
        write C[row, dense_tile]
```

关键点：

- program 并没有“并行处理 8 条短行”
- 它是在一个 program 内顺序处理多条短行

所以这里的优化重点不是“单行更快”，而是“单位 launch 能覆盖更多短行”。

### 5.8 为什么中长行适合 `vector`

当行长度到 129 以上后：

- 单行已经有足够 sparse work
- 再把多行塞进一个 program，会让单个 program 太重
- 行内访问本身就足以填满计算

因此 `spmm_opt` 公开路径在 `129~2048` 范围内改用 `vector`，即：

- 一行对应一个 program
- 但不再是基础版那种“纯通用基线”，而是已经建立在分桶之后

### 5.9 手工例子 3：vector 为什么适合中长行

设某一行有 220 个非零，另有 `dense_cols = 64`。

如果仍用 `batched`：

- 一个 program 要在内部处理多条 200+ 非零的行
- 内部串行工作量过大
- 单个 program latency 上升，容易拖尾

改成 vector：

- 一个 program 专心处理这一条行
- 稀疏访问与 dense tile 累加都围绕这条行展开
- 调度更符合“每条行已经够大”的事实

### 5.10 `vector` kernel 在做什么

`_spmm_csr_vector_rows_f32_kernel` / `_f64_kernel` 与基础版很像，区别是：

- 它不是对“所有行统一跑”
- 而是只对某个 bucket 中的 `rows` 跑

这使得它具备两个实际优势：

- 只在相似长度的行之间调度
- `BLOCK_NNZ` 可以按 bucket 固定成更贴近该类行的值

### 5.11 为什么超长行必须 `split`

如果某一行 `nnz > 2048`，继续让一个 program 从头扫到尾，会有明显问题：

- 单个 program 工作量过大
- 长行会成为整个 kernel 的拖尾
- 其他较短行早就结束，只剩长行在跑

所以公开 `opt` 路线把这类行拆成多个 part：

- 每个 part 负责该行的一段非零
- 每个 part 先算自己的部分和
- 再把所有 part 的结果 reduce 回原始行

### 5.12 长行元数据怎么建

`_build_spmm_opt_split_metadata(...)` 的核心逻辑是：

- 找到每一条超长行在 CSR 中的 `[start, end)`
- 用 `part_block_nnz = 256` 切成多个区间
- 生成：
  - `long_part_rows`
  - `long_part_starts`
  - `long_part_ends`
  - `long_row_part_ptr`

本质上它把：

```text
row 17: [0, 256), [256, 512), [512, 768), ...
```

这样的结构显式存下来，后续 kernel 就能按 part 并行。

### 5.13 手工例子 4：split 如何工作

假设某一行有 3000 个非零，`part_block_nnz = 256`，那么这一行会被切成大约 12 个 part：

```text
part0: [0, 256)
part1: [256, 512)
...
part11: [2816, 3000)
```

第一阶段：

- 每个 part 独立算出一个长度为 `BLOCK_N` 的局部输出向量
- 写到 `workspace[part_id, :]`

第二阶段：

- reduce kernel 按原始行收集该行全部 part
- 对 `workspace` 中属于同一行的 part 求和
- 写回 `C[row, :]`

这就把“一个超长行必须由一个 program 从头扫到尾”变成了“多个 part 分担，再二次归约”。

### 5.14 `split` kernel 在做什么

#### 第一阶段：`_spmm_csr_split_part_*_kernel`

每个 program 负责：

- 一个 part
- 一个 dense tile

逻辑是：

```text
program(part_id, dense_tile):
    acc = 0
    for idx in [part_start, part_end):
        acc += data[idx] * B[col[idx], dense_tile]
    store workspace[part_id, dense_tile] = acc
```

#### 第二阶段：`_spmm_csr_split_reduce_*_kernel`

每个 program 负责：

- 一条原始长行
- 一个 dense tile

逻辑是：

```text
program(long_row_id, dense_tile):
    acc = 0
    for each part belonging to this row:
        acc += workspace[part_id, dense_tile]
    store C[row, dense_tile] = acc
```

### 5.15 公开 `spmm_opt` 的真实优缺点

优点：

- 短行、中行、长行分别处理，比基础版更有针对性
- `prepare` 非常适合“同一稀疏矩阵反复乘不同 B”
- 能主动处理超长行，不再把它们完全绑死在一个 program 上

缺点：

- 分桶只看 `row_nnz`，不看 `dense_cols`、矩阵偏斜程度和 value 分布
- 长行 `split` 引入 `workspace + reduce` 额外访存
- 预处理元数据生成里有 CPU `tolist()` 和 Python 循环

### 5.16 数值特征

公开 `spmm_opt` 路线的数值行为基本是：

- `float32` bucket 路线用 `tl.float32` 累加
- `float64` bucket 路线用 `tl.float64` 累加
- 没有补偿求和
- `split` 路线是“先 part 局部和，再全行 reduce”

因此相较基础版，它的主要提升方向是性能，不是精度。

### 5.16.1 `spmm_opt` 里的“归约”其实有三层

这里容易混淆。`spmm_opt` 里并不是所有路线都用了同一种归约方式。

#### 第一层：单行内部的普通乘加归约

这是 `batched` 和 `vector` 共有的基本模式：

```text
acc = 0
for chunk_start in 0..row_nnz step block_nnz:
    for kk in 0..block_nnz-1:
        idx = start + chunk_start + kk
        acc += data[idx] * B[col[idx], dense_tile]
```

也就是说，对一条行而言，公开 `spmm_opt` 主路径并没有树形归约，也没有补偿求和；它就是把该行所有 `nnz` 顺序累加到一个 `acc[BLOCK_N]` 向量里。

#### 第二层：长行 `split` 的二阶段归约

只有 `split` 路线存在显式的“局部归约 + 全局归约”：

```text
part kernel:
    acc_part = sum over nnz in this part
    workspace[part_id, :] = acc_part

reduce kernel:
    acc_row = sum over all parts of the same original row
    C[row, :] = acc_row
```

这是真正意义上的二次归约。第一阶段对 part 内部归约，第二阶段对多个 part 的局部结果再次归约。

#### 第三层：candidate segmented 的程序内分段归并

`candidate segmented` 不是 `split`。它没有 `workspace`，也没有跨 program reduce，而是：

- 一条行先逻辑切成多个 segment
- 每个 segment 得到自己的 `acc0/acc1/...`
- 最后在同一个 program 内做树形合并

所以：

- `split` 是跨 program 的二阶段归约
- `segmented` 是单 program 内的多段归并

### 5.16.2 公开 `spmm_opt` 主路径如何累加

#### `batched`

`batched` 的“优化”在于一个 program 内顺序处理多条短行，而不是改变单条行的归约公式。对每一条行，它仍然是：

```text
acc_row = sum(A[row, col] * B[col, dense_tile])
```

#### `vector`

`vector` 也是一样的单行归约，只是 program 不再包多条行，而是一行一个 program。

#### `split`

`split` 才是主路径里唯一改变归约拓扑的路线。它把：

```text
一条超长行的一次大归约
```

拆成：

```text
多个 part 的局部归约 + 一个 reduce kernel 的全行归约
```

所以如果你问“`spmm_opt` 真正特别的归约算法是什么”，最核心的答案就是：

- 对短/中行：仍是普通顺序累加，只是调度形态变了
- 对超长行：改成了 `split-part -> workspace -> reduce` 的二阶段归约

### 5.16.3 用一个完整例子看 `spmm_opt split` 如何归约

假设有一条超长行：

```text
row = 17
row_nnz = 10
part_block_nnz = 4
block_n = 4
n_dense_cols = 6
```

为了方便说明，我们把 `part_block_nnz` 缩小成 4。真实代码里公开 `split` 用的是 256。

这条行会先被切成 3 个 part：

```text
part0: nnz[0:4]
part1: nnz[4:8]
part2: nnz[8:10]
```

输出列又会按 `block_n = 4` 切成两个 tile：

```text
tile0: cols [0,1,2,3]
tile1: cols [4,5]
```

于是实际执行会变成这样：

#### 第一步：part kernel 先算局部和

例如：

```text
program(part0, tile0) -> 算 row17 的 nnz[0:4] 对 cols[0:3] 的贡献
program(part1, tile0) -> 算 row17 的 nnz[4:8] 对 cols[0:3] 的贡献
program(part2, tile0) -> 算 row17 的 nnz[8:10] 对 cols[0:3] 的贡献
```

它们分别写出：

```text
workspace[part0, 0:4] = partial_sum0
workspace[part1, 0:4] = partial_sum1
workspace[part2, 0:4] = partial_sum2
```

对于 `tile1` 也同理：

```text
workspace[part0, 4:6]
workspace[part1, 4:6]
workspace[part2, 4:6]
```

#### 第二步：reduce kernel 再把同一行的 part 加起来

接着：

```text
program(row17, tile0):
    C[17, 0:4] =
        workspace[part0, 0:4] +
        workspace[part1, 0:4] +
        workspace[part2, 0:4]

program(row17, tile1):
    C[17, 4:6] =
        workspace[part0, 4:6] +
        workspace[part1, 4:6] +
        workspace[part2, 4:6]
```

可以把它画成一张小图：

```text
原始一条超长行 row17
    |
    +--> part0 ----\
    +--> part1 -----+--> workspace ----> reduce by row17 ----> C[row17, :]
    +--> part2 ----/
```

所以 `split` 的本质不是“公式变了”，而是：

- 先把一条超长行切成多个局部归约任务
- 再把这些局部归约结果合并回原行

### 5.16.4 candidate segmented 又是怎么“归并”的

这条路线更像“程序内部的多段 reduction”。

假设：

- 一条行有 12 个 nnz
- `SEGMENTS = 3`

那么这条行会被逻辑分成 3 段，例如：

```text
seg0: nnz[0:4]
seg1: nnz[4:8]
seg2: nnz[8:12]
```

但它们不是不同 program，也不会写 workspace，而是同一个 program 内部维护：

```text
acc0
acc1
acc2
```

最后在程序内部做：

```text
acc = acc0 + acc1 + acc2
```

所以：

- `split` 是“先分任务，再跨 program 合并”
- `segmented` 是“同一个 program 内部分段，再在寄存器里合并”

## 6. `spmm_opt` 的 diagnose / experimental 路线

这一部分非常重要，因为当前代码里“定义了哪些 kernel”和“实际最终走到哪些 kernel”并不完全一致。

## 6.1 stable 路线存在的意义

`stable` 路线的目标是：

- 用更高精度累加
- 或用补偿求和
- 作为 diagnose/对照路径观察误差来源

它不是公开性能主路线，而更像“高稳定性参考实现”。

### 6.2 stable 路线的 kernel 特征

#### `stable_batched_*`

- 定义了 batched 版本
- 使用补偿求和变量 `comp`

#### `stable_vector_*`

- `float32` 路线直接升到 `tl.float64` 累加，再转回 `float32`
- `float64` 路线维持 `tl.float64`

#### `stable_split_part_*` / `stable_split_reduce_*`

- part 与 reduce 阶段都在更高精度上做

### 6.3 当前 stable 路线的真实行为

这点必须明确写清楚：

- 虽然 `stable_batched_*` kernel 已定义
- 但 `_run_spmm_opt_bucket_stable(...)` 中，当 bucket 是 `batched` 时，会强制把它当成 `vector` 来跑

也就是说：

- 短行稳定路径当前不是走 `stable_batched_*`
- 而是改走“row-per-program 的 stable vector 风格”

这是一个非常关键的“定义存在，但公开 diagnose 真实执行不同”的例子。

### 6.4 candidate 路线存在的意义

`candidate` 路线的目标更像：

- 尝试更细粒度的 bucket
- 对不同 row length 区间做更激进的特化
- 作为实验性调度候选，对比 legacy opt 路线

### 6.5 candidate 的分桶表

#### `float32`

| 标签 | kind | 行长度范围 | `block_nnz` | `segments` |
| --- | --- | --- | ---: | ---: |
| `short_16` | `micro` | `0~16` | 16 | - |
| `short_32` | `micro` | `17~32` | 32 | - |
| `short_64` | `segmented` | `33~64` | 64 | 2 |
| `short_128` | `segmented` | `65~128` | 128 | 4 |
| `vector_192` | `segmented` | `129~192` | 64 | 2 |
| `vector_256` | `segmented` | `193~256` | 64 | 4 |
| `vector_512` | `segmented` | `257~512` | 64 | 4 |
| `vector_1024` | `segmented` | `513~1024` | 32 | 8 |
| `vector_2048` | `segmented` | `1025~2048` | 32 | 8 |
| `split` | `split` | `>2048` | 256 | - |

#### `float64`

`float64` 的 candidate 路线更保守，短行一开始就是 `segmented`，没有 `micro`。

### 6.6 candidate 的 micro / segmented 是什么

#### `micro`

`_spmm_csr_candidate_short_micro_kernel` 直接把一个短行的全部非零 gather 成小矩阵，再：

```text
prod = a_vals[:, None] * b_vals
acc = sum(prod, axis=0)
```

它本质上是：

- 非常短的行
- 用更“微核化”的方式一次吃掉

#### `segmented`

`_spmm_csr_candidate_segmented_rows_kernel` 把一行逻辑上分成多个 segment：

- 2 段
- 4 段
- 最多 8 段

每段分别累加，再做树式合并。

这和 `split` 不同：

- `segmented` 仍是一个 program 内部逻辑分段
- `split` 是真的把一行拆成多个 part，由不同 program 处理并写 workspace

### 6.7 手工例子 5：candidate segmented 与 split 的区别

假设某一行有 800 个非零。

如果走 `candidate segmented`：

- 一个 program 仍负责这行
- 它把 800 个非零逻辑切成若干 segment
- 在 program 内部做多段累加后合并

如果走 `split`：

- 多个 program 分别负责不同区间
- 结果先落到 `workspace`
- 再由额外 reduce kernel 汇总

所以：

- `segmented` 是“单 program 内部分治”
- `split` 是“跨 program 的真正拆分”

### 6.8 candidate 路线的真实行为差异

当前代码里有一个非常关键的行为：

`_candidate_bucket_uses_stable_short_kernel(bucket)` 规定：

- 只要 `bucket.max_row_nnz <= 256`
- 就优先走 `stable_short` kernel

这意味着：

- `short_16`
- `short_32`
- `short_64`
- `short_128`
- `vector_192`
- `vector_256`

这些 bucket 虽然在“分桶定义”上有 `micro` 或 `segmented` 设计，
但真实执行时，会被 `stable_short_f32/f64` 覆盖。

换句话说：

- 代码里确实定义了更细的 `micro` / `segmented` 思路
- 但当前 diagnose candidate 路线对 `<=256 nnz/row` 的 bucket，真实优先级是 `stable_short`

这也是本文必须强调“设计意图不等于当前真实执行行为”的第二个典型例子。

### 6.9 当前代码中的实验分支脱节点

至少有三处值得明确记录：

1. `stable_batched_*` kernel 已定义，但 stable 短行路线实际改走 vector 风格。
2. candidate 的 `micro/segmented` 设计在 `<=256 nnz/row` 时大范围被 `stable_short` 覆盖。
3. `_SPMM_OPT_LONG_ROW_THRESHOLD` 被定义，但没有真正作为公开 dispatch 关键条件使用。

这些都说明：

- `spmm_opt` 的实验设计曾经更丰富
- 但当前落地代码里，部分实验路径并没有以“名字所暗示的方式”真实执行

## 7. COO SpMM：`flagsparse_spmm_coo`

### 7.1 设计定位

COO 路线不是简单把 CSR 换个存储格式，而是先把原始 COO 变成更适合计算的 canonical 形式，再选择 route：

- `rowrun`
- `atomic`

公开 `flagsparse_spmm_coo` 默认走 `rowrun`。

### 7.2 canonical 预处理在做什么

`_prepare_spmm_coo_canonical_prepared(...)` 做了三件核心事情：

1. `compute_dtype` 提升
   - `float16/bfloat16 -> float32`
   - `float32 -> float64`
   - `complex64 -> complex128`

2. `coalesce`
   - 相同 `(row, col)` 的重复项先合并求和

3. 按 `(row, col)` 词典序排序

这三步非常关键，因为原始 COO 常见问题正是：

- 重复坐标
- 乱序
- 行段不连续

### 7.3 为什么 `rowrun` 强依赖 canonical sort/coalesce

`rowrun` 的基本思想是：

- 同一行的非零如果排在一起
- 那么整行就可以像 CSR 一样“作为一个连续段处理”

为了做到这一点，需要先构造：

```text
seg_starts = [0, break1, break2, ..., nnz]
```

其中每个 segment 对应排序后 COO 中的一段同行非零。

如果不先排序：

- 同一行的非零散落在全数组不同位置
- `rowrun` 就没法自然形成连续段

如果不先 coalesce：

- 重复项会使“同一行同一列的多次写入”逻辑变复杂

所以 `rowrun` 的先决条件正是 canonical COO。

### 7.4 `rowrun` kernel 在做什么

`_spmm_coo_rowrun_real_kernel` / `_complex_kernel` 的思路几乎就是“COO 版按行扫描”：

```text
program(seg_id, dense_tile):
    start = seg_starts[seg_id]
    end = seg_starts[seg_id + 1]
    row = row[start]
    acc = 0
    for idx in [start, end):
        acc += data[idx] * B[col[idx], dense_tile]
    store C[row, dense_tile] = acc
```

可以看到：

- segment 就是一条逻辑行
- 它本质上把 canonical COO 重新变成“可按行遍历的形式”

### 7.5 手工例子 6：COO rowrun 如何形成 `seg_starts`

假设 canonical COO 为：

```text
row = [0, 0, 0, 2, 2, 5]
col = [1, 3, 4, 0, 7, 2]
val = [...]
```

那么行边界很清楚：

- row 0 对应 `[0, 3)`
- row 2 对应 `[3, 5)`
- row 5 对应 `[5, 6)`

所以：

```text
seg_starts = [0, 3, 5, 6]
```

然后：

- `seg 0` 负责 row 0
- `seg 1` 负责 row 2
- `seg 2` 负责 row 5

这就是 `rowrun` 的本质。

### 7.6 `atomic` route 在做什么

`_spmm_coo_atomic_real_kernel` / `_complex_kernel` 的粒度非常直接：

- `program_id(0)` 是某个非零 `idx`
- `program_id(1)` 是某个输出列 `dense_col`

也就是：

```text
one nonzero x one dense column -> one atomic contribution
```

伪代码：

```text
idx = program_id(0)
j = program_id(1)
row = row[idx]
col = col[idx]
atomic_add(C[row, j], data[idx] * B[col, j])
```

它的优点是通用，几乎不需要先把同行整理成连续段；缺点是原子冲突与顺序问题明显。

### 7.7 为什么 `atomic` 更通用，但通常更难稳定

`atomic` 的优点：

- 概念简单
- 不强依赖同行连续
- 适合作为对照或 fallback

缺点：

- 多个非零可能同时写同一个 `C[row, j]`
- 写入顺序不固定
- 浮点加法不结合，因此顺序波动会导致结果轻微变化
- 热点行/热点列会导致原子冲突，性能下降

### 7.8 数值特征

`COO rowrun`：

- 仍是普通顺序求和
- 但因为同行先聚到一段内，执行顺序更可解释

`COO atomic`：

- 原子加的先后顺序不稳定
- 更容易出现可重复性差和误差漂移

因此在精度语义上：

- `rowrun` 更适合作为默认公开路径
- `atomic` 更适合作为内部对照路线

## 8. 真实 `.mtx` 输入例子

## 8.1 `test.mtx`：极小型 Matrix Market 示例

文件：

- `E:/codex_project/alphasparse_triton/alphasparse/matrix_test/test.mtx`

头部显示它是：

- `MatrixMarket matrix coordinate complex Hermitian`

尺寸很小：

- `4 x 4`
- `8` 个条目

这个文件特别适合用来解释两件事：

1. Matrix Market 装载到 `CSR/COO` 时，如何把 `(row, col, val)` 转成稀疏格式
2. complex / Hermitian 输入对 SpMM 的意义

它不适合作为 `spmm_opt` 代表矩阵，因为：

- 太小
- 行分布过于简单
- 没有短行/中行/超长行混合的调度意义

## 8.2 `mhd1280b.mtx`：真实 complex Hermitian 样本

文件：

- `E:/codex_project/alphasparse_triton/alphasparse/matrix_test/mhd1280b.mtx`

头部显示：

- `1280 x 1280`
- `12029` 条目
- complex Hermitian
- 电磁学问题矩阵

这个样本更适合说明：

- Matrix Market complex 数据如何读入
- `spmm_csr` / `spmm_coo` 的 complex 路径如何解释
- 为什么 `spmm_opt` 当前不支持 complex：因为 `opt` 只支持 `float32/float64`

## 9. 真实诊断矩阵案例

下面引用仓库已有诊断：

- `spmm_data/diag_all_native-f32_a/spmm_diag_summary.csv`
- `spmm_data/diag_all_native-f32_a/diag_buckets/*.buckets.csv`

注意：这里分析的是“已有结果”，不是本文新运行得到的结果。

## 9.1 案例 A：`amazon0601.mtx`，典型短行主导矩阵

摘要数据：

- `n_rows = 403394`
- `nnz = 3387388`
- `avg_nnz_per_row ≈ 8.40`

bucket 统计显示：

- 全部 `403394` 行都落在 `short_16`
- `row_nnz_max = 10`

这说明它是非常典型的“短行主导”矩阵。对这种矩阵：

- 基础版 CSR 可以算，但并不利用“行特别短”的事实
- `spmm_opt batched` 非常自然，因为它可以把大量小行打包

为什么这类矩阵适合 batched：

- 一行只有个位数到十个左右非零
- 如果一行一个 program，大量 program 都在做很小的工作
- 用 `batched` 可以显著减少这种碎片化

数值现象上：

- `legacy_worst_native_ratio ≈ 0.00623`
- `candidate_worst_native_ratio ≈ 0.00829`

误差都不大，说明此类短行矩阵的主要矛盾更偏向性能而不是精度。

## 9.2 案例 B：`mip1.mtx`，典型长尾混合矩阵

摘要数据：

- `n_rows = 66463`
- `nnz = 10352819`
- `avg_nnz_per_row ≈ 155.77`

但这个均值非常有迷惑性，因为 bucket 分布非常长尾：

- `short_16`: 21223 行
- `short_32`: 7821 行
- `short_64`: 6229 行
- `short_128`: 12058 行
- `vector_192`: 1699 行
- `vector_256`: 7474 行
- `vector_512`: 1360 行
- `vector_1024`: 8527 行
- `vector_2048`: 37 行
- `split`: 35 行

这正是 `spmm_opt` 设计最想处理的矩阵类型：

- 有大量短行，需要 batched
- 有不少中长行，需要 vector
- 还有极少数特别长的尾部行，需要 split

也就是说，`mip1.mtx` 非常适合拿来解释“为什么不能只用一个 kernel”。

如果全部用基础版：

- `short_16` 那部分会太碎
- `split` 那 35 行会拖尾

如果用 `spmm_opt`：

- 短行不再逐条独立调度
- 中长行单独成桶
- 超长尾部行被拆 part

这里 `split` bucket 的 `row_nnz_mean ≈ 4728.63`，最大达到 `66395`，这正说明：

- 没有 split 不现实
- 单行极端长尾会严重影响纯 row-per-program 路线

## 9.3 案例 C：`c8_mat11.mtx`，高密长行且误差敏感

摘要数据：

- `n_rows = 4562`
- `nnz = 2462970`
- `avg_nnz_per_row ≈ 539.89`

bucket 分布覆盖：

- `short_16`
- `short_32`
- `short_64`
- `short_128`
- `vector_192`
- `vector_256`
- `vector_512`
- `vector_1024`
- `vector_2048`

其中尤其值得注意：

- `vector_1024` 有 `2166` 行
- `vector_2048` 有 `349` 行
- `short_128` 甚至出现 `legacy_bad_rows_native_gt_1 = 1`

摘要误差上：

- `legacy_worst_native_ratio ≈ 1.6575`
- `candidate_worst_native_ratio ≈ 1.4703`

这说明它已经不是“只是快慢问题”，而是明显暴露了数值稳定性风险。

为什么这类矩阵容易出问题：

- 行很长，单行求和项多
- 大量乘加导致舍入误差累积
- 若存在正负抵消，普通顺序求和更容易放大误差

因此它非常适合作为：

- 为什么要有 `stable` 路线
- 为什么要区分 native reference 与 high precision reference

的代表矩阵。

## 9.4 案例 D：`engine.mtx`，中长行集中且误差尾部明显

摘要数据：

- `n_rows = 143571`
- `nnz = 4706073`
- `avg_nnz_per_row ≈ 32.78`

bucket 数据显示：

- `short_32` 行数极多：`106869`
- `short_64`：`25968`
- `short_128`：`10161`
- 还有少量 `vector_192`

这说明它总体上还是短到中等长度主导，但已经不是 `amazon0601` 那种极短图结构型矩阵，而是更偏工程型、混合型。

误差上：

- `legacy_worst_native_ratio ≈ 0.2650`
- `candidate_worst_native_ratio ≈ 0.2232`

虽然没有 `c8_mat11` 那么极端，但已经足够说明：

- 即便 bucket 还没进入真正的 `split`
- 中长行与更复杂数值分布也会让误差尾部变厚

## 10. 为什么这些路线对不同矩阵有效

可以把当前实现理解成下面这张“矩阵画像 -> 路线”映射表：

| 矩阵画像 | 典型特征 | 最自然路线 |
| --- | --- | --- |
| 图结构短行矩阵 | 平均行长很低，几乎都在 `<=16` | `spmm_opt batched` |
| 中等有限元/工程矩阵 | 行长分布集中在 `32~512` | `spmm_opt batched + vector` |
| 长尾型矩阵 | 大多数普通行 + 少数超长尾部行 | `spmm_opt split` 必不可少 |
| 需要 complex 支持 | `complex64/complex128` | `spmm_csr` 或 `spmm_coo`，不能用 `spmm_opt` |
| 原始 COO 输入且重复项多 | 需要先合并排序 | `spmm_coo rowrun` 更自然 |
| 想要通用 COO 对照 | 不介意原子冲突 | `spmm_coo atomic` |

## 11. 关键代码行为总结

## 11.1 公开路径

- `flagsparse_spmm_csr`
  - 通用基线
  - 支持最广
  - 行驱动

- `flagsparse_spmm_csr_opt`
  - 只支持 `float32/float64`
  - 先 `prepare`
  - 再 `batched / vector / split`

- `flagsparse_spmm_coo`
  - 默认 `rowrun`
  - 先 canonical 化 COO

## 11.2 diagnose / internal 路径

- `stable`
  - 目标是更稳，不是更快
  - 短行当前真实走 vector 风格而非 stable batched

- `candidate`
  - 目标是更细粒度试验
  - 但 `<=256 nnz/row` 的许多 bucket 会优先落到 `stable_short`

## 11.3 已定义但当前未完整发挥的语义

- `stable_batched_*` 定义了，但 diagnose 稳定短行不真正依赖它
- `micro` / `segmented` 某些 candidate 设计存在，但实际常被 `stable_short` 覆盖
- `_SPMM_OPT_LONG_ROW_THRESHOLD` 存在，但不是公开 dispatch 核心条件

## 12. 优化建议

## 12.1 性能方向

### 建议 1：分桶阈值不要只看 `row_nnz`

当前 `spmm_opt` 分桶完全由行长度驱动，这有明显局限：

- `dense_cols = 8` 和 `dense_cols = 128` 对最优 kernel 的要求不同
- 同样是 `row_nnz = 80`，若矩阵极端偏斜或缓存局部性不同，最优路线也可能不同

建议：

- 让分桶阈值至少联合考虑 `row_nnz` 与 `n_dense_cols`
- 对极端偏斜矩阵引入“top-k 长行占比”或 `row_nnz` 分位数特征

### 建议 2：优化长行拆分元数据构造

`_build_spmm_opt_split_metadata(...)` 当前依赖：

- `long_rows.to(...).cpu().tolist()`
- `kernel_indptr.to(...).cpu().tolist()`
- Python 循环生成 part

这意味着：

- prepare 阶段存在显著 Python/CPU 开销
- 对大矩阵 repeated prepare 不友好

建议：

- 尽量用张量运算构造 part offsets
- 至少把 CPU 循环收缩到更小的数据子集

### 建议 3：降低 split 路线的 workspace 成本

当前 `split` 路线必然经历：

- part kernel 写 `workspace`
- reduce kernel 再读回来

问题是：

- 多一次 global write
- 多一次 global read
- 额外 workspace 分配

建议：

- 研究 fused reduce 或层次化归约
- 至少尝试减少 workspace 生命周期与中间张量写回量

### 建议 4：整理 candidate / stable 的实验分支

当前问题不只是代码多，而是语义不透明：

- 函数名暗示一种路线
- 真实执行却可能被别的分支覆盖

建议：

- 若某实验路线不再需要，直接删掉
- 若仍要保留，明确改名或重新组织 dispatch 逻辑
- 避免“定义存在但不真正生效”的维护陷阱

### 建议 5：把 repeated `B` 调用场景写进用户文档

当前 `prepare_spmm_csr_opt` 的价值很明确，但公开说明还不够突出。

建议：

- 明确告诉使用者：固定 `A`、多次更换 `B` 时一定要复用 `prepared`
- 这是当前最直接、最稳妥的性能收益点

## 12.2 精确方向

### 建议 1：显式暴露稳定模式

现在稳定路径只存在 diagnose 内部入口，对正式用户不可见。

建议：

- 考虑显式增加 stable 模式开关
- 让用户能在吞吐与精度间自己做取舍

### 建议 2：对长行和高误差 bucket 采用更强累加策略

对于 `float32`：

- 长行
- 高消去误差行
- `vector_1024 / vector_2048 / split` 这类 bucket

可以考虑：

- 更广泛的 `fp64` 累加
- 补偿求和
- 或按误差风险动态切换稳定 kernel

### 建议 3：明确区分两类误差视角

当前诊断脚本已经区分：

- native reference
- high precision reference

这是很好的做法，建议在更正式的对外说明中也保留这种区分。

原因是：

- native reference 适合看“同 dtype 下的工程误差”
- high precision reference 适合看“真实数值偏差”

若混在一起，很容易误判某条路线到底是“精度真的差”，还是“只是跟同 dtype 参考的顺序不同”。

### 建议 4：对 `COO atomic` 明确标注顺序不稳定

`COO atomic` 不是错，而是语义上更接近：

- 通用
- 但不保证固定求和顺序

建议：

- 文档中明确写出其可重复性与误差风险
- 避免用户误把它看成和 `rowrun` 同级的默认稳定路径

## 13. 结论

`flagsparse_of` 当前的 SpMM 体系可以概括成三层：

1. `spmm_csr`：通用、清晰、支持面广的 CSR 基线。
2. `spmm_csr_opt`：围绕行长度分桶、针对短中长行分别优化的性能主线。
3. `spmm_coo`：先 canonical 化，再在 `rowrun` 和 `atomic` 间选择的 COO 路线。

如果只抓一句话：

- `CSR base` 的关键词是“单行扫描”
- `CSR opt` 的关键词是“分桶 + 短行合并 + 长行拆分”
- `COO rowrun` 的关键词是“排序后按行段处理”
- `COO atomic` 的关键词是“每个非零直接原子写回”

而当前代码最值得额外注意的地方是：

- diagnose / candidate / stable 路线里，函数命名、分桶定义和真实执行路径并不总是一一对应
- 这既是分析理解时的重点，也是后续性能与工程化整理时最值得下手的地方

## 14. 参考来源

### 14.1 源码

- `E:/codex_project/alphasparse_triton/flagsparse_of/src/flagsparse/sparse_operations/spmm_csr.py`
- `E:/codex_project/alphasparse_triton/flagsparse_of/src/flagsparse/sparse_operations/spmm_coo.py`
- `E:/codex_project/alphasparse_triton/flagsparse_of/src/flagsparse/__init__.py`
- `E:/codex_project/alphasparse_triton/flagsparse_of/src/flagsparse/sparse_operations/__init__.py`

### 14.2 测试与诊断

- `E:/codex_project/alphasparse_triton/flagsparse_of/tests/test_spmm.py`
- `E:/codex_project/alphasparse_triton/flagsparse_of/tests/test_spmm_opt.py`
- `E:/codex_project/alphasparse_triton/flagsparse_of/tests/diagnose_spmm_opt.py`
- `E:/codex_project/alphasparse_triton/flagsparse_of/tests/pytest/test_spmm_csr_accuracy.py`

### 14.3 真实矩阵与现有诊断结果

- `E:/codex_project/alphasparse_triton/alphasparse/matrix_test/test.mtx`
- `E:/codex_project/alphasparse_triton/alphasparse/matrix_test/mhd1280b.mtx`
- `E:/codex_project/alphasparse_triton/spmm_data/diag_all_native-f32_a/spmm_diag_summary.csv`
- `E:/codex_project/alphasparse_triton/spmm_data/diag_all_native-f32_a/diag_buckets/amazon0601.buckets.csv`
- `E:/codex_project/alphasparse_triton/spmm_data/diag_all_native-f32_a/diag_buckets/mip1.buckets.csv`
- `E:/codex_project/alphasparse_triton/spmm_data/diag_all_native-f32_a/diag_buckets/c8_mat11.buckets.csv`
- `E:/codex_project/alphasparse_triton/spmm_data/diag_all_native-f32_a/diag_buckets/engine.buckets.csv`
