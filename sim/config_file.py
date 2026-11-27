"""Reading and editing the YAML files under configs/ (comments are kept when a tool writes a value)."""

from __future__ import annotations

import json
import pathlib
import re

import yaml


def load(path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    old = [k for k in cfg if isinstance(k, str) and k.isupper()]
    if old:  # written before the key names became lower case
        raise ValueError(
            f"{path}：参数名已经改成小写（{old[0]} 改为 {old[0].lower()}），这个文件里还有 {len(old)} 个大写的参数名"
        )
    return cfg


def set_keys(path, values: dict, after: str | None = None) -> None:
    """Rewrite top-level `KEY: value` lines in place, keeping the layout and trailing comments.

    A key that is not in the file yet is added after the line of key `after` (or at the end).
    Values are written in YAML flow style (numbers, strings, lists).
    """
    path = pathlib.Path(path)
    text = path.read_text()
    for key, val in values.items():
        flow = json.dumps(val)  # JSON numbers, strings and lists are valid YAML flow values
        pat = re.compile(rf"^({re.escape(key)}:[ \t]*)([^#\n]*?)([ \t]*#[^\n]*)?$", re.M)
        new, n = pat.subn(lambda m, flow=flow: m.group(1) + flow + (m.group(3) or ""), text, count=1)
        if n == 0:
            line = f"{key}: {flow}\n"
            m = re.search(rf"^{re.escape(after)}:[^\n]*\n", text, re.M) if after else None
            new = text[: m.end()] + line + text[m.end() :] if m else text.rstrip("\n") + "\n" + line
        text = new
    path.write_text(text)
    check = load(path)
    for key, val in values.items():
        if check.get(key) != val:
            raise ValueError(f"{path}: {key} reads back as {check.get(key)!r}, expected {val!r}")
