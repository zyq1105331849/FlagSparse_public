"""Checks for the FlagSparse operator registry."""

import re
from pathlib import Path

import flagsparse
from run_flagsparse_pytest import DEFAULT_EXCLUDED_OPS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = PROJECT_ROOT / "conf" / "operators.yaml"


def _registry_text():
    return REGISTRY.read_text(encoding="utf-8")


def _operator_ids(text):
    return re.findall(r"^  - id: ([a-z0-9_]+)$", text, flags=re.MULTILINE)


def _registered_interfaces(text):
    return re.findall(r"^      - ([A-Za-z_][A-Za-z0-9_]*)$", text, flags=re.MULTILINE)


def _active_registered_interfaces(text):
    current_id = None
    in_for = False
    interfaces = []
    for line in text.splitlines():
        match = re.match(r"^  - id: ([a-z0-9_]+)$", line)
        if match:
            current_id = match.group(1)
            in_for = False
            continue
        if re.match(r"^    for:\s*$", line):
            in_for = True
            continue
        if line.startswith("    ") and not line.startswith("      - "):
            in_for = False
        if in_for and current_id not in DEFAULT_EXCLUDED_OPS:
            item = re.match(r"^      - ([A-Za-z_][A-Za-z0-9_]*)$", line)
            if item:
                interfaces.append(item.group(1))
    return interfaces


def test_operator_registry_uses_flagext_style_shape():
    text = _registry_text()
    assert text.startswith("ops:\n")
    for snippet in [
        "description: |",
        "for:",
        "labels:",
        "kind:",
        "stages:",
    ]:
        assert snippet in text


def test_operator_registry_ids_are_unique():
    text = _registry_text()
    ids = _operator_ids(text)
    assert ids
    assert len(ids) == len(set(ids))


def test_operator_registry_interfaces_are_public_exports():
    text = _registry_text()
    exported = set(flagsparse.__all__)
    ignored_yaml_values = {
        "flagsparse",
        "sparse",
        "triton",
        "public-api",
        "csr",
        "coo",
        "optimized",
        "alpha-sparse",
        "descriptor",
        "format",
        "Sparse",
        "SparseLinearAlg",
        "SparseSolver",
        "SparseFormat",
        "Utility",
    }
    interfaces = {
        name
        for name in _active_registered_interfaces(text)
        if name not in ignored_yaml_values
    }
    assert interfaces
    assert interfaces <= exported
