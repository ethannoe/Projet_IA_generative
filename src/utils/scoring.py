import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.scoring import (
    aggregate_block_scores,
    coverage_score,
    find_missing_skills,
    interpret_coverage,
    rank_jobs,
)


@dataclass
class ScoringInputs:
    user_texts: List[str]
    blocks: List[Dict[str, Any]]
    jobs: List[Dict[str, Any]]
    similarity_matrix: np.ndarray
    skill_mapping: List[Tuple[str, str]]


def _normalize_similarity(similarity_matrix: np.ndarray) -> np.ndarray:
    # Imposer des floats et borner à [0,1] pour éviter que des cosinus négatifs ne biaisent les scores
    return np.clip(similarity_matrix.astype(float), 0.0, 1.0)


def _block_lookup(blocks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {b.get("block_id"): b for b in blocks}


def compute_block_scores_with_details(
    similarity_matrix: np.ndarray, skill_mapping: List[Tuple[str, str]], blocks: List[Dict[str, Any]]
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Calcule les scores par bloc + détails pour le debug (max par texte).
    """
    block_scores = aggregate_block_scores(similarity_matrix, skill_mapping, blocks)
    block_scores = {k: float(np.clip(v, 0.0, 1.0)) for k, v in block_scores.items()}

    details: Dict[str, Any] = {}
    for block in blocks:
        block_id = block.get("block_id")
        skill_indices = [i for i, (bid, _) in enumerate(skill_mapping) if bid == block_id]
        if not skill_indices:
            continue
    # Similarité maximale par texte utilisateur pour ce bloc
        per_text_max = similarity_matrix[:, skill_indices].max(axis=1)
        details[block_id] = {
            "skill_indices": skill_indices,
            "per_text_max": [float(x) for x in per_text_max],
        }
    return block_scores, details


def _job_lookup_by_name(jobs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {j.get("job_name"): j for j in jobs}


def compute_job_breakdown(ranked_jobs: List[Tuple[str, float, str, List[Tuple[str, float, float, float]]]]) -> List[Dict[str, Any]]:
    breakdown: List[Dict[str, Any]] = []
    for job_name, score, justification, contributions in ranked_jobs:
        total_weight = sum(w for _, _, w, _ in contributions) or 1.0
        breakdown.append(
            {
                "job_name": job_name,
                "score": float(score),
                "justification": justification,
                "total_weight": float(total_weight),
                "contributions": [
                    {
                        "block_id": bid,
                        "block_score": float(block_score),
                        "weight": float(weight),
                        "weight_norm": float(weight / total_weight),
                        "contribution": float(contribution),
                    }
                    for bid, block_score, weight, contribution in contributions
                ],
            }
        )
    return breakdown


def compute_skill_scores(
    similarity_matrix: np.ndarray,
    skill_mapping: List[Tuple[str, str]],
    user_texts: List[str],
    blocks: List[Dict[str, Any]],
    block_scores: Optional[Dict[str, float]] = None,
    focus_blocks: Optional[List[str]] = None,
    min_threshold: float = 0.35,
    top_k: int = 10,
    focus_boost: float = 0.12,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Retourne (toutes_compétences_triées, top_compétences_filtrées).
    Chaque entrée contient un aperçu de texte source pour le debug.
    """
    block_map = _block_lookup(blocks)
    focus_set = set(focus_blocks or [])
    block_scores = block_scores or {}
    max_per_skill = similarity_matrix.max(axis=0)
    argmax_per_skill = similarity_matrix.argmax(axis=0)

    entries: List[Dict[str, Any]] = []
    for idx, (block_id, skill) in enumerate(skill_mapping):
        base_score = float(max_per_skill[idx])
        block_weight = float(block_map.get(block_id, {}).get("weight", 1.0))
        block_score_factor = float(block_scores.get(block_id, 1.0))
        boosted = base_score * block_weight * (0.6 + 0.4 * block_score_factor)
        if block_id in focus_set:
            boosted *= 1 + focus_boost
        score = float(np.clip(boosted, 0.0, 1.0))
        best_text_idx = int(argmax_per_skill[idx]) if similarity_matrix.shape[0] > 0 else -1
        source_text = user_texts[best_text_idx] if 0 <= best_text_idx < len(user_texts) else ""
        preview = source_text[:120] + ("…" if len(source_text) > 120 else "")
        entries.append(
            {
                "block_id": block_id,
                "block_name": block_map.get(block_id, {}).get("block_name", block_id),
                "skill": skill,
                "score": score,
                "base_score": float(base_score),
                "block_weight": block_weight,
                "boost_applied": focus_boost if block_id in focus_set else 0.0,
                "source_text_index": best_text_idx,
                "source_len": len(source_text),
                "source_preview": preview,
                "signal": "embedding",
            }
        )

    entries.sort(key=lambda e: e["score"], reverse=True)
    filtered = [e for e in entries if e["score"] >= min_threshold]

    if focus_set:
        focus_entries = [e for e in entries if e["block_id"] in focus_set]
        filtered = filtered or []
        for f in focus_entries:
            if f not in filtered:
                filtered.append(f)

    filtered.sort(key=lambda e: e["score"], reverse=True)
    top_entries = filtered[:top_k]
    return entries, top_entries


def run_scoring_pipeline(
    inputs: ScoringInputs,
    *,
    top_k_skills: int = 8,
    min_skill_score: float = 0.35,
    focus_boost: float = 0.12,
) -> Dict[str, Any]:
    """
    Pipeline centralisé pour calculer scores métiers/compétences + traces debug.
    """
    sims = _normalize_similarity(inputs.similarity_matrix)
    block_scores, block_details = compute_block_scores_with_details(sims, inputs.skill_mapping, inputs.blocks)

    coverage = coverage_score(block_scores, inputs.blocks)
    coverage_label = interpret_coverage(coverage)

    ranked_jobs = rank_jobs(block_scores, inputs.jobs)
    job_breakdown = compute_job_breakdown(ranked_jobs)

    job_lookup = _job_lookup_by_name(inputs.jobs)
    top_job_name = ranked_jobs[0][0] if ranked_jobs else None
    focus_blocks = [rb.get("block_id") for rb in job_lookup.get(top_job_name, {}).get("required_blocks", [])] if top_job_name else []

    all_skills, top_skills = compute_skill_scores(
        sims,
        inputs.skill_mapping,
        inputs.user_texts,
        inputs.blocks,
        block_scores=block_scores,
        focus_blocks=focus_blocks,
        min_threshold=min_skill_score,
        top_k=top_k_skills,
        focus_boost=focus_boost,
    )

    missing_skills = find_missing_skills(sims, inputs.skill_mapping, inputs.blocks, threshold=min_skill_score)

    return {
        "block_scores": block_scores,
        "coverage": coverage,
        "coverage_label": coverage_label,
        "ranked_jobs": ranked_jobs,
        "top_jobs": [(j, s) for j, s, _, _ in ranked_jobs[:3]],
        "missing_skills": missing_skills,
        "top_skills": [(e["block_id"], e["skill"], e["score"]) for e in top_skills],
        "debug": {
            "jobs": job_breakdown,
            "skills": top_skills,
            "all_skills": all_skills,
            "blocks": block_details,
            "similarity_shape": sims.shape,
            "focus_blocks": focus_blocks,
            "min_skill_score": min_skill_score,
        },
    }


__all__ = [
    "ScoringInputs",
    "run_scoring_pipeline",
    "compute_block_scores_with_details",
    "compute_job_breakdown",
    "compute_skill_scores",
]
