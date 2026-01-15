import hashlib
import json
import os
from typing import Any, Optional

from diskcache import Cache

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")
GENAI_NAMESPACE = "genai_outputs"
EMBED_NAMESPACE = "embeddings"


def _ensure_cache_dir() -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def get_cache(namespace: str) -> Cache:
    base = _ensure_cache_dir()
    return Cache(os.path.join(base, namespace))


def cache_get(namespace: str, key: str) -> Optional[Any]:
    cache = get_cache(namespace)
    return cache.get(_hash_key(key))


def cache_set(namespace: str, key: str, value: Any, expire: Optional[int] = None) -> None:
    cache = get_cache(namespace)
    cache.set(_hash_key(key), value, expire=expire)


def cache_json_key(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(obj)


__all__ = [
    "CACHE_DIR",
    "GENAI_NAMESPACE",
    "EMBED_NAMESPACE",
    "cache_get",
    "cache_set",
    "cache_json_key",
    "get_cache",
]
