"""Config loading helpers for the pick-and-place controller."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def _parse_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if not value.strip():
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
            continue
        current[key] = _parse_scalar(value.strip())
    return root


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else Path(__file__).resolve().parent / "config" / "default.yaml"
    return _parse_simple_yaml(config_path.read_text(encoding="utf-8"))
