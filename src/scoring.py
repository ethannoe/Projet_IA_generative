import logging
from typing import Dict, List, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


def aggregate_block_scores(
    similarity_matrix: np.ndarray,
    skill_mapping: List[Tuple[str, str]],
    blocks: List[Dict],
) -> Dict[str, float]:
    """
    Score par bloc :
    - Pour chaque bloc, pour chaque texte utilisateur, on prend la meilleure similarité avec une compétence du bloc.
    - Le score du bloc est la moyenne de ces meilleures similarités (sur l'ensemble des textes utilisateur).
    """
    block_scores: Dict[str, List[float]] = {b["block_id"]: [] for b in blocks}

    # Matrice de similarité : (nb_textes_utilisateur, nb_compétences)
    for block in blocks:
        block_id = block["block_id"]
        skill_indices = [i for i, (bid, _) in enumerate(skill_mapping) if bid == block_id]
        if not skill_indices:
            block_scores[block_id] = []
            continue
        # Pour chaque texte utilisateur, on prend le max des compétences du bloc
        best_per_text = []
        for row in similarity_matrix:  # forme de la ligne : (nb_compétences,)
            sims_block = row[skill_indices]
            best_per_text.append(float(sims_block.max()))
        block_scores[block_id] = best_per_text

    aggregated: Dict[str, float] = {}
    for block in blocks:
        scores = block_scores.get(block["block_id"], [])
        aggregated[block["block_id"]] = float(np.mean(scores)) if scores else 0.0
    return aggregated


def coverage_score(block_scores: Dict[str, float], blocks: List[Dict]) -> float:
    weights = np.array([b.get("weight", 1.0) for b in blocks], dtype=float)
    scores = np.array([block_scores.get(b["block_id"], 0.0) for b in blocks], dtype=float)
    if weights.sum() == 0:
        return 0.0
    return float(np.clip(np.dot(weights, scores) / weights.sum(), 0.0, 1.0))


def interpret_coverage(score: float) -> str:
    if score >= 0.7:
        return "forte couverture"
    if score >= 0.5:
        return "couverture moyenne"
    return "couverture faible"


def _job_signature(job: Dict) -> Tuple[Tuple[str, float], ...]:
    reqs = job.get("required_blocks", []) or []
    norm = tuple(sorted([(r.get("block_id"), float(r.get("weight", 1.0))) for r in reqs], key=lambda x: x[0]))
    return norm


def warn_indistinguishable_jobs(jobs: List[Dict]) -> None:
    seen = {}
    for job in jobs:
        sig = _job_signature(job)
        seen.setdefault(sig, []).append(job.get("job_name", "unknown"))
    for sig, names in seen.items():
        if len(names) > 1:
            LOGGER.warning("Jobs share identical block requirements: %s -> %s", sig, names)


def rank_jobs(block_scores: Dict[str, float], jobs: List[Dict]) -> List[Tuple[str, float, str, List[Tuple[str, float, float, float]]]]:
    """
    Retourne une liste triée (job_name, score, justification, contributions)
    où contributions = [(block_id, block_score, weight, contribution)]
    et score = sum(weight * block_score) / sum(weight) sur les blocs requis.
    """
    warn_indistinguishable_jobs(jobs)
    ranked: List[Tuple[str, float, str, List[Tuple[str, float, float, float]]]] = []
    for job in jobs:
        reqs = job.get("required_blocks", []) or []
        weights = []
        weighted = []
        contributions: List[Tuple[str, float, float, float]] = []
        for rb in reqs:
            block_id = rb.get("block_id")
            weight = float(rb.get("weight", 1.0))
            block_score = float(block_scores.get(block_id, 0.0))
            weights.append(weight)
            weighted.append(weight * block_score)
            contributions.append((block_id, block_score, weight, weight * block_score))
        job_score = float(np.sum(weighted) / np.sum(weights)) if weights else 0.0
        # justification simple basée sur blocs dominants
        strong = [bid for bid, sc, _, _ in contributions if sc >= 0.6]
        justification = (
            "Ce métier est recommandé car tes compétences en "
            + (", ".join(strong) if strong else "plusieurs blocs pertinents")
            + " sont bien représentées."
        )
        ranked.append((job.get("job_name", ""), job_score, justification, contributions))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def find_missing_skills(
    similarity_matrix: np.ndarray,
    skill_mapping: List[Tuple[str, str]],
    blocks: List[Dict],
    threshold: float = 0.5,
) -> Dict[str, List[str]]:
    missing: Dict[str, List[str]] = {b["block_id"]: [] for b in blocks}
    for col_idx, (block_id, skill) in enumerate(skill_mapping):
        col_sims = similarity_matrix[:, col_idx]
        if float(col_sims.max()) < threshold:
            missing[block_id].append(skill)
    return missing


__all__ = [
    "aggregate_block_scores",
    "coverage_score",
    "interpret_coverage",
    "rank_jobs",
    "find_missing_skills",
    "warn_indistinguishable_jobs",
]
