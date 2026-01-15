import json
import os
from typing import Any, Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


class DataNotFoundError(FileNotFoundError):
    pass


def _load_json(path: str) -> Any:
    if not os.path.exists(path):
        raise DataNotFoundError(f"Missing data file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_blocks() -> List[Dict[str, Any]]:
    path = os.path.join(DATA_DIR, "competency_blocks.json")
    return _load_json(path)


def load_jobs() -> List[Dict[str, Any]]:
    path = os.path.join(DATA_DIR, "job_profiles.json")
    return _load_json(path)


def block_skill_texts(blocks: List[Dict[str, Any]]) -> Tuple[List[str], List[Tuple[str, str]]]:
    texts: List[str] = []
    mapping: List[Tuple[str, str]] = []
    for block in blocks:
        block_id = block.get("block_id")
        for skill in block.get("skills", []):
            texts.append(skill)
            mapping.append((block_id, skill))
    return texts, mapping


__all__ = ["load_blocks", "load_jobs", "block_skill_texts", "DataNotFoundError", "DATA_DIR"]
