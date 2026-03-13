# FlagSparse

GPU sparse operations package (SpMV, gather, scatter, sparse formats).

## Install

```bash
pip install . --no-deps --no-build-isolation
```

Use `--no-build-isolation` to avoid downloading build deps when offline.

Runtime dependencies (install when needed):

```bash
pip install torch triton cupy-cuda12x
```

## Layout

- `src/flagsparse/` — core package
- `tests/` — pytest tests
- `benchmark/` — performance benchmarks
