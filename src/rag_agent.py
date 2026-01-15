from typing import Dict, List, Tuple

import numpy as np

from .semantic_engine import SemanticEngine
from .scoring import aggregate_block_scores


def retrieve_top_skills(
    engine: SemanticEngine, user_texts: List[str], top_k: int = 8
) -> List[Tuple[str, str, float]]:
    sims, mapping = engine.similarity_to_skills(user_texts)
    max_per_skill = sims.max(axis=0)
    entries: List[Tuple[str, str, float]] = []
    for score, (block_id, skill) in zip(max_per_skill, mapping):
        entries.append((block_id, skill, float(score)))
    entries.sort(key=lambda x: x[2], reverse=True)
    return entries[:top_k]


def build_context_pack(
    top_skills: List[Tuple[str, str, float]],
    blocks: List[Dict],
    block_scores: Dict[str, float],
) -> str:
    block_lookup = {b["block_id"]: b for b in blocks}
    lines: List[str] = []
    lines.append("Référentiel pertinent (top compétences):")
    for block_id, skill, score in top_skills:
        block_name = block_lookup.get(block_id, {}).get("block_name", block_id)
        lines.append(f"- [{block_name}] {skill} (similarité: {score:.2f})")
    lines.append("\nScores par bloc:")
    for block_id, sc in block_scores.items():
        block_name = block_lookup.get(block_id, {}).get("block_name", block_id)
        lines.append(f"- {block_name}: {sc:.2f}")
    return "\n".join(lines)


def recompute_block_scores_from_engine(
    engine: SemanticEngine, user_texts: List[str], top_k: int = 3
) -> Dict[str, float]:
    sims, mapping = engine.similarity_to_skills(user_texts)
    return aggregate_block_scores(sims, mapping, engine.get_blocks())


__all__ = ["retrieve_top_skills", "build_context_pack", "recompute_block_scores_from_engine"]
