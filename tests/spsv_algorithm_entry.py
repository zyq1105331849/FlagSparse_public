"""Small entrypoint helper for per-algorithm SpSV scripts."""

import sys
from pathlib import Path


def run_spsv_algorithm(alg_num):
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    if "--alg-num" not in sys.argv and "--alg_num" not in sys.argv:
        sys.argv[1:1] = ["--alg-num", str(int(alg_num))]

    from tests.spsv_common import main

    main()
