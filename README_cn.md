# FlagSparse

GPU 稀疏运算库（SpMV、gather、scatter、多种稀疏格式）。

## 安装

```bash
pip install . --no-deps --no-build-isolation
```

离线时可加 `--no-build-isolation` 避免拉取构建依赖。

运行时依赖（按需安装）：

```bash
pip install torch triton cupy-cuda12x
```

## 目录说明

- `src/flagsparse/` — 核心包
- `tests/` — pytest 测试
- `benchmark/` — 性能基准
