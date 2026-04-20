"""Small shared helpers."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers and not logging.getLogger().handlers:
        # Only install a handler if nothing higher up has one already.
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def short_hash(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


class _JsonEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, BaseModel):
            return o.model_dump(mode="json")
        return super().default(o)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, cls=_JsonEncoder)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
