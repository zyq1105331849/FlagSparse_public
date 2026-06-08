"""Validate and record the ROCm runtime expected by DCU GPU CI jobs."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-rocm",
        action="store_true",
        help="Fail when torch cannot see a ROCm/DCU device through PyTorch.",
    )
    parser.add_argument(
        "--min-device-count",
        type=int,
        default=1,
        help="Minimum DCU device count required when --require-rocm is set.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional JSON path for environment metadata.",
    )
    return parser.parse_args()


def _run_probe(command: List[str]) -> Dict[str, Any]:
    executable = command[0]
    from shutil import which

    resolved = which(executable)
    if not resolved:
        return {"available": False, "error": f"{executable} not found"}

    proc = subprocess.run(
        [resolved, *command[1:]],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "available": proc.returncode == 0,
        "return_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _rocm_smi() -> Dict[str, Any]:
    return _run_probe(["rocm-smi", "--showproductname", "--showdriverversion"])


def _torch_rocm_metadata() -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    try:
        import torch
    except Exception as exc:
        metadata["torch_import_error"] = repr(exc)
        metadata["rocm_available"] = False
        metadata["rocm_device_count"] = 0
        return metadata

    metadata["torch_version"] = getattr(torch, "__version__", "unknown")
    metadata["torch_hip_version"] = getattr(torch.version, "hip", None)
    metadata["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    # PyTorch ROCm intentionally exposes devices through torch.cuda.
    metadata["rocm_available"] = bool(torch.cuda.is_available())
    metadata["rocm_device_count"] = (
        int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    )
    devices: List[Dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": f"{properties.major}.{properties.minor}",
                    "total_memory": int(properties.total_memory),
                }
            )
    metadata["devices"] = devices
    return metadata


def _write_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    metadata: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rocm_smi": _rocm_smi(),
    }
    metadata.update(_torch_rocm_metadata())

    if args.metadata:
        _write_metadata(args.metadata, metadata)

    print(json.dumps(metadata, indent=2, sort_keys=True))

    if not args.require_rocm:
        return 0

    if not metadata.get("rocm_available"):
        print(
            "ROCm/DCU is required but torch.cuda.is_available() is false.",
            file=sys.stderr,
        )
        return 1

    device_count = int(metadata.get("rocm_device_count", 0))
    if device_count < args.min_device_count:
        print(
            f"ROCm/DCU device count {device_count} is below required "
            f"{args.min_device_count}.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
