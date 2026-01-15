import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

from src.genai_client import GenAIUnavailable, generate_with_cache, is_ollama_available

DATA_JOBS_PATH = "data/jobs_v2.json"
DATA_SKILLS_PATH = "data/skills_v2.json"


@dataclass
class ScoringInputsV2:
    responses: Dict[str, Any]
    jobs: List[Dict[str, Any]]
    skills: List[Dict[str, Any]]
    use_ollama: bool = True


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jobs_v2() -> List[Dict[str, Any]]:
    return _load_json(DATA_JOBS_PATH)


def load_skills_v2() -> List[Dict[str, Any]]:
    return _load_json(DATA_SKILLS_PATH)


def sha_signature(payload: Any) -> str:
    try:
        data = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    except TypeError:
        data = str(payload)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _normalize_text(s: str) -> str:
    return s.lower().strip()


def build_inputs_from_session(responses: Dict[str, Any]) -> Dict[str, Any]:
    likert = responses.get("likert", {}) or {}
    selected_skills = responses.get("checkbox_skills", []) or []
    qcm_multi = responses.get("qcm_multi", []) or []
    guided = responses.get("guided", {}) or {}

    text_fields = [
        responses.get("free_text", ""),
        responses.get("ambitions", ""),
        responses.get("proj_tech", ""),
        responses.get("prob_complex", ""),
        responses.get("model_build", ""),
        responses.get("projection", ""),
    ]
    texts = [t.strip() for t in text_fields if t and t.strip()]

    experiences = texts  # reuse texts as experience evidence

    return {
        "likert": likert,
        "selected_skills": selected_skills,
        "tools": qcm_multi,
        "guided": guided,
        "texts": texts,
        "experiences": experiences,
    }


def _match_tags(skill: Dict[str, Any], corpus: List[str]) -> bool:
    tags = [t.lower() for t in skill.get("tags", [])]
    for c in corpus:
        lc = _normalize_text(c)
        if any(tag in lc for tag in tags):
            return True
    return False


def _likert_to_score(val: int) -> float:
    if not isinstance(val, (int, float)):
        return 0.0
    return float(np.clip((val - 1) / 4, 0, 1) * 60.0)  # 0..60 contribution


def compute_skill_scores(inputs: Dict[str, Any], skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    likert = inputs.get("likert", {})
    selected = [_normalize_text(s) for s in inputs.get("selected_skills", [])]
    tools = [_normalize_text(s) for s in inputs.get("tools", [])]
    texts = inputs.get("texts", []) or []

    skill_scores: List[Dict[str, Any]] = []

    # map likert keys to tags
    likert_tag_map = {
        "python_level": ["python", "pandas"],
        "ml_level": ["ml"],
        "nlp_level": ["nlp", "text"],
        "etl_level": ["etl", "pipeline"],
        "da_analysis_level": ["sql", "eda"],
        "da_viz_level": ["viz", "dashboard"],
        "da_biz_level": ["communication", "product"],
        "pm_level": ["product", "pm"],
        "comms_level": ["communication"],
    }

    for skill in skills:
        skill_id = skill.get("skill_id")
        name = skill.get("skill_name", skill_id)
        tags = [t.lower() for t in skill.get("tags", [])]

        selected_contrib = 40.0 if _normalize_text(name) in selected or any(tag in selected for tag in tags) else 0.0
        if not selected_contrib and any(tag in tools for tag in tags):
            selected_contrib = 25.0

        likert_contrib = 0.0
        for key, taglist in likert_tag_map.items():
            if any(tag in tags for tag in taglist):
                likert_val = likert.get(key)
                likert_contrib = max(likert_contrib, _likert_to_score(likert_val))

        text_contrib = 0.0
        if _match_tags(skill, texts):
            text_contrib = 20.0

        score = float(np.clip(selected_contrib + likert_contrib + text_contrib, 0.0, 100.0))
        skill_scores.append(
            {
                "skill_id": skill_id,
                "skill_name": name,
                "score": score,
                "sources": {
                    "selected": selected_contrib,
                    "likert": likert_contrib,
                    "text": text_contrib,
                },
            }
        )

    skill_scores.sort(key=lambda x: x["score"], reverse=True)
    return skill_scores


def compute_job_scores(skill_scores: List[Dict[str, Any]], jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup = {s["skill_id"]: s for s in skill_scores}
    job_scores: List[Dict[str, Any]] = []
    for job in jobs:
        req = job.get("required_skills", [])
        opt = job.get("optional_skills", [])
        req_scores = [lookup.get(sid, {}).get("score", 0.0) for sid in req]
        opt_scores = [lookup.get(sid, {}).get("score", 0.0) for sid in opt]
        req_avg = float(np.mean(req_scores)) if req_scores else 0.0
        opt_avg = float(np.mean(opt_scores)) if opt_scores else 0.0
        score = 0.7 * req_avg + 0.3 * opt_avg
        coverage = 0.0
        if req:
            coverage = sum(1 for s in req if lookup.get(s, {}).get("score", 0) >= 50) / len(req) * 100.0
        job_scores.append(
            {
                "job_id": job.get("job_id"),
                "job_name": job.get("job_name"),
                "score": float(np.clip(score, 0.0, 100.0)),
                "coverage": float(np.clip(coverage, 0.0, 100.0)),
                "required_skills": req,
                "optional_skills": opt,
                "required_scores": req_scores,
                "optional_scores": opt_scores,
            }
        )
    job_scores.sort(key=lambda x: x["score"], reverse=True)
    return job_scores


def derive_level(score: float) -> str:
    if score < 45:
        return "Junior"
    if score < 70:
        return "Mid"
    return "Senior"


def build_summary(job_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not job_scores:
        return {"title": "Données insuffisantes", "confidence": 0, "short_explanation": "Fournis plus d'informations."}
    top = job_scores[0]
    level = derive_level(top.get("score", 0.0))
    return {
        "title": f"{top.get('job_name', 'Profil')} ({level})",
        "confidence": top.get("score", 0.0),
        "short_explanation": f"Principalement basé sur les compétences requises pour {top.get('job_name')} et ton niveau {level}.",
    }


def generate_texts(job_scores: List[Dict[str, Any]], skill_scores: List[Dict[str, Any]]) -> Tuple[str, str]:
    top_job = job_scores[0] if job_scores else None
    top_skills = skill_scores[:5]
    bio = ""
    explanation = ""
    if is_ollama_available():
        try:
            prompt_bio = (
                "Rédige une courte bio (4-6 lignes) basée sur le métier recommandé et les compétences.")
            prompt_bio += f"\nMétier top: {top_job}\nCompétences top: {top_skills}"
            bio = generate_with_cache("bio_v2", prompt_bio)
            prompt_exp = "Explique brièvement pourquoi ces métiers sont recommandés en citant les compétences fortes."
            prompt_exp += f"\nTop métiers: {job_scores[:3]}\nTop compétences: {top_skills}"
            explanation = generate_with_cache("exp_v2", prompt_exp)
        except GenAIUnavailable:
            bio = "Bio non générée (Ollama indisponible)."
            explanation = "Ollama indisponible, explication simplifiée."
    else:
        bio = "Bio non générée (Ollama indisponible)."
        explanation = "Explication locale : les métiers sont classés selon la correspondance des compétences requises."
    return bio, explanation


def run_scoring_v2(scoring_inputs: ScoringInputsV2) -> Dict[str, Any]:
    start = time.time()
    inputs_struct = build_inputs_from_session(scoring_inputs.responses)
    sig = sha_signature(inputs_struct)

    skill_scores = compute_skill_scores(inputs_struct, scoring_inputs.skills)
    job_scores = compute_job_scores(skill_scores, scoring_inputs.jobs)
    summary = build_summary(job_scores)

    bio, explanation = generate_texts(job_scores, skill_scores)

    duration = time.time() - start
    return {
        "meta": {
            "version": "v2",
            "run_at": datetime.now().isoformat(),
            "input_signature": sig,
            "duration_sec": duration,
            "model": "ollama" if is_ollama_available() else "none",
        },
        "inputs_resume": {
            "nb_skills_selected": len(inputs_struct.get("selected_skills", [])),
            "nb_texts": len(inputs_struct.get("texts", [])),
            "avg_text_len": (
                sum(len(t) for t in inputs_struct.get("texts", [])) / len(inputs_struct.get("texts", []))
                if inputs_struct.get("texts")
                else 0
            ),
        },
        "job_scores": job_scores,
        "skill_scores": skill_scores,
        "coverage_score": job_scores[0].get("coverage", 0.0) if job_scores else 0.0,
        "summary_profile": {**summary, "long_explanation": explanation},
        "professional_bio": bio,
        "debug": {
            "inputs_struct": inputs_struct,
            "skill_scores": skill_scores,
            "job_scores": job_scores,
        },
    }


__all__ = [
    "ScoringInputsV2",
    "run_scoring_v2",
    "load_jobs_v2",
    "load_skills_v2",
    "build_inputs_from_session",
]
