"""Shared data-platform loader for llm_alpha_discovery/exp2 (Validation Bottleneck, Exp 2).

data_platform reads its API keys from ~/git/data_platform/.env via pydantic's upward
.env walk, which misses sibling repos. Inject the keys before constructing DataPlatform.

Usage:
    from _dp import get_dp
    dp = get_dp()
    doc = dp.signals.open_asset_pricing_doc()
"""
from __future__ import annotations

import os
from pathlib import Path

DP_ROOT = Path.home() / "git" / "data_platform"


def _load_env() -> None:
    env_path = DP_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def get_dp():
    _load_env()
    from data_platform import DataPlatform

    return DataPlatform()
