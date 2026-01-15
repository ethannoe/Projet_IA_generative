import logging
import os
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from . import cache
from .data_loader import block_skill_texts, load_blocks

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticEngine:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.blocks = load_blocks()
        self.skill_texts, self.skill_mapping = block_skill_texts(self.blocks)
        self.skill_embeddings = self._precompute_skill_embeddings()

    def _load_model(self, model_name: str) -> SentenceTransformer:
        LOGGER.info("Loading embedding model %s", model_name)
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".models")
        os.makedirs(cache_dir, exist_ok=True)
        return SentenceTransformer(model_name, cache_folder=cache_dir)

    def _embed_with_cache(self, texts: List[str]) -> np.ndarray:
        embeddings: List[np.ndarray] = []
        uncached_texts: List[str] = []
        cached_indices: List[int] = []

        for idx, text in enumerate(texts):
            key = cache.cache_json_key({"model": self.model_name, "text": text})
            cached = cache.cache_get(cache.EMBED_NAMESPACE, key)
            if cached is not None:
                embeddings.append(np.array(cached))
                cached_indices.append(idx)
            else:
                uncached_texts.append(text)
                embeddings.append(None)  # type: ignore

        if uncached_texts:
            new_embs = self.model.encode(uncached_texts, show_progress_bar=False, convert_to_numpy=True)
            insert_pos = 0
            for i, emb in enumerate(embeddings):
                if emb is None:
                    embeddings[i] = new_embs[insert_pos]
                    key = cache.cache_json_key({"model": self.model_name, "text": texts[i]})
                    cache.cache_set(cache.EMBED_NAMESPACE, key, embeddings[i].tolist())
                    insert_pos += 1

        return np.vstack(embeddings)

    def _precompute_skill_embeddings(self) -> np.ndarray:
        LOGGER.info("Precomputing skill embeddings with caching")
        return self._embed_with_cache(self.skill_texts)

    def embed_user_inputs(self, texts: List[str]) -> np.ndarray:
        return self._embed_with_cache(texts)

    def similarity_to_skills(self, user_texts: List[str]) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
        user_embs = self.embed_user_inputs(user_texts)
        sims = cosine_similarity(user_embs, self.skill_embeddings)
        return sims, self.skill_mapping

    def get_blocks(self):
        return self.blocks


__all__ = ["SemanticEngine", "DEFAULT_MODEL"]
