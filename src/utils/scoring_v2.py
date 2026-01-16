from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from src.genai_client import GenAIUnavailable, generate_with_cache, is_ollama_available

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DATA_JOBS_PATH = DATA_DIR / "jobs_v2.json"
DATA_SKILLS_PATH = DATA_DIR / "skills_v2.json"

# Correspondance des clés Likert vers les compétences pour la contribution d'auto-évaluation
SKILL_LIKERT_MAP: Dict[str, str] = {
    "sql": "da_analysis_level",
    "bi_tools": "da_viz_level",
    "data_viz": "da_viz_level",
    "stats_foundations": "da_analysis_level",
    "kpi_definition": "da_biz_level",
    "data_storytelling": "comms_level",
    "excel": "da_analysis_level",
    "ab_testing": "da_analysis_level",
    "python": "python_level",
    "git": "python_level",
    "testing": "python_level",
    "clean_code": "python_level",
    "etl_elt": "etl_level",
    "data_modeling": "etl_level",
    "orchestration": "etl_level",
    "data_warehouse": "etl_level",
    "spark": "etl_level",
    "streaming": "etl_level",
    "data_quality": "etl_level",
    "ml_basics": "ml_level",
    "feature_engineering": "ml_level",
    "training_eval": "ml_level",
    "model_deployment": "ml_level",
    "mlflow": "ml_level",
    "model_monitoring": "ml_level",
    "nlp_basics": "nlp_level",
    "api_design": "python_level",
    "backend_framework": "python_level",
    "db_design": "da_analysis_level",
    "auth_security": "pm_level",
    "performance": "python_level",
    "caching": "python_level",
    "system_design": "pm_level",
    "linux": "etl_level",
    "docker": "etl_level",
    "kubernetes": "etl_level",
    "ci_cd": "etl_level",
    "iac": "etl_level",
    "cloud": "etl_level",
    "observability": "etl_level",
    "networking": "etl_level",
    "sre_reliability": "etl_level",
    "communication": "comms_level",
    "requirements": "pm_level",
}

# Mots-clés additionnels pour améliorer la détection dans les textes et sélections
KEYWORD_OVERRIDES: Dict[str, List[str]] = {
    "bi_tools": ["powerbi", "tableau", "looker", "bi"],
    "data_viz": ["visualisation", "dashboard", "chart"],
    "stats_foundations": ["stat", "probabilité", "probability"],
    "kpi_definition": ["kpi", "metrics", "métriques"],
    "data_storytelling": ["storytelling", "présentation", "restitution"],
    "etl_elt": ["etl", "elt", "pipeline", "ingestion"],
    "data_modeling": ["schema", "modélisation", "star schema", "snowflake"],
    "orchestration": ["airflow", "prefect"],
    "data_warehouse": ["warehouse", "bigquery", "snowflake", "redshift"],
    "spark": ["spark"],
    "streaming": ["kafka", "pubsub", "stream"],
    "data_quality": ["qualité", "tests de données"],
    "ml_basics": ["ml", "machine learning"],
    "feature_engineering": ["feature", "variables"],
    "training_eval": ["entrainement", "évaluation", "evaluation"],
    "model_deployment": ["déploiement", "serving", "inférence", "api"],
    "mlflow": ["mlflow"],
    "model_monitoring": ["monitoring", "drift"],
    "nlp_basics": ["nlp", "texte", "language"],
    "api_design": ["api", "rest"],
    "backend_framework": ["fastapi", "django", "node", "spring", "java", "framework"],
    "db_design": ["base de données", "schema", "index"],
    "auth_security": ["auth", "jwt", "oauth", "sécurité"],
    "performance": ["performance", "optimisation"],
    "caching": ["redis", "cache"],
    "system_design": ["architecture", "system design"],
    "docker": ["docker", "container"],
    "kubernetes": ["kubernetes", "k8s"],
    "ci_cd": ["ci", "cd", "pipeline"],
    "iac": ["terraform", "iac"],
    "cloud": ["aws", "gcp", "azure", "cloud"],
    "observability": ["logs", "metrics", "traces", "observabilité"],
    "networking": ["réseau", "network"],
    "sre_reliability": ["sre", "fiabilité", "reliability"],
    "communication": ["communication", "collaboration"],
    "requirements": ["cadrage", "besoin"],
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jobs() -> List[Dict[str, Any]]:
    return _load_json(DATA_JOBS_PATH)


def load_skills() -> List[Dict[str, Any]]:
    return _load_json(DATA_SKILLS_PATH)


def compute_input_signature(payload: Any) -> str:
    try:
        data = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    except TypeError:
        data = str(payload)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _normalize(text: str | None) -> str:
    return (text or "").lower().strip()


def _auto_eval_score(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    v_clamped = min(5.0, max(1.0, v))
    return ((v_clamped - 1.0) / 4.0) * 70.0  # échelle 0..70


def _skill_keywords(skill: Dict[str, Any]) -> List[str]:
    keywords: List[str] = []
    skill_id = _normalize(skill.get("skill_id"))
    skill_name = _normalize(skill.get("skill_name"))
    keywords.extend(skill_id.replace("_", " ").split())
    keywords.extend(skill_name.replace("/", " ").split())
    keywords.extend(KEYWORD_OVERRIDES.get(skill.get("skill_id", ""), []))
    return [k for k in {_normalize(k) for k in keywords} if k]


def _match_any(keywords: Iterable[str], text: str) -> bool:
    return any(k in text for k in keywords)


def build_inputs_from_session_state(session_state: Mapping[str, Any]) -> Dict[str, Any]:
    responses = session_state.get("responses", {}) if hasattr(session_state, "get") else {}
    likert = responses.get("likert", {}) or {}
    selected_skills = responses.get("checkbox_skills", []) or []
    tools = responses.get("qcm_multi", []) or []

    text_fields = ["free_text", "ambitions", "proj_tech", "prob_complex", "model_build", "projection"]
    texts = [responses.get(k, "") for k in text_fields if responses.get(k, "").strip()]
    experiences = texts.copy()

    return {
        "likert": likert,
        "selected_skills": selected_skills,
        "tools": tools,
        "texts": texts,
        "experiences": experiences,
        "raw_responses": responses,
    }


def compute_skill_scores(inputs: Mapping[str, Any], skills: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    likert = inputs.get("likert", {}) or {}
    selections = [_normalize(s) for s in (inputs.get("selected_skills", []) or [])]
    tools = [_normalize(s) for s in (inputs.get("tools", []) or [])]
    selected_pool = selections + tools
    text_corpus = " ".join((inputs.get("texts", []) or []) + (inputs.get("experiences", []) or []))
    text_corpus_norm = _normalize(text_corpus)

    skill_scores: List[Dict[str, Any]] = []

    for skill in skills:
        skill_id = skill.get("skill_id")
        name = skill.get("skill_name", skill_id)
        block_id = skill.get("block_id")

        auto_key = SKILL_LIKERT_MAP.get(skill_id, "")
        auto_eval_contrib = _auto_eval_score(likert.get(auto_key)) if auto_key else 0.0

        keywords = _skill_keywords(skill)
        selection_hit = any(k in sel for sel in selected_pool for k in keywords) or any(
            _normalize(name) in sel for sel in selected_pool
        )
        selected_contrib = 25.0 if selection_hit else 0.0

        text_hit = _match_any(keywords, text_corpus_norm)
        text_contrib = 15.0 if text_hit else 0.0

        score = min(100.0, auto_eval_contrib + selected_contrib + text_contrib)

        skill_scores.append(
            {
                "skill_id": skill_id,
                "skill_name": name,
                "block_id": block_id,
                "score": float(score),
                "sources": {
                    "auto_eval": round(auto_eval_contrib, 2),
                    "selected": selected_contrib,
                    "text": text_contrib,
                },
            }
        )

    skill_scores.sort(key=lambda s: s.get("score", 0), reverse=True)
    return skill_scores


def compute_job_scores(skill_scores: Sequence[Dict[str, Any]], jobs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup = {s["skill_id"]: s for s in skill_scores}
    job_scores: List[Dict[str, Any]] = []
    for job in jobs:
        req_ids = job.get("required_skills", []) or []
        opt_ids = job.get("optional_skills", []) or []
        req_vals = [lookup.get(sid, {}).get("score", 0.0) for sid in req_ids]
        opt_vals = [lookup.get(sid, {}).get("score", 0.0) for sid in opt_ids]
        req_avg = float(mean(req_vals)) if req_vals else 0.0
        opt_avg = float(mean(opt_vals)) if opt_vals else 0.0
        job_score = 0.75 * req_avg + 0.25 * opt_avg
        job_scores.append(
            {
                "job_id": job.get("job_id"),
                "job_name": job.get("job_name"),
                "score": round(float(job_score), 2),
                "coverage": compute_coverage_value(req_ids, lookup),
                "required_skills": req_ids,
                "optional_skills": opt_ids,
                "required_scores": req_vals,
                "optional_scores": opt_vals,
            }
        )
    job_scores.sort(key=lambda j: j.get("score", 0), reverse=True)
    return job_scores


def compute_coverage_value(required_skill_ids: Sequence[str], skill_lookup: Mapping[str, Dict[str, Any]], threshold: float = 60.0) -> float:
    if not required_skill_ids:
        return 0.0
    hits = sum(1 for sid in required_skill_ids if skill_lookup.get(sid, {}).get("score", 0.0) >= threshold)
    return round(hits / len(required_skill_ids) * 100.0, 2)


def compute_coverage(top_job: Dict[str, Any] | None, skill_scores: Sequence[Dict[str, Any]], threshold: float = 60.0) -> float:
    """Coverage = required skills >= threshold / total required * 100.

    - Only required_skills are considered.
    - Missing skills default to score 0 (not covered).
    - If no required_skills, returns 0.0 (avoid inflating to 100%).
    - Threshold expects skill scores on 0..100 scale.
    """

    if not top_job:
        return 0.0
    required = top_job.get("required_skills", []) or []
    if not required:
        return 0.0
    lookup = {s["skill_id"]: s for s in skill_scores}
    return compute_coverage_value(required, lookup, threshold)


def classify_level(score: float) -> str:
    if score < 45:
        return "Junior"
    if score <= 70:
        return "Mid"
    return "Senior"


def _inputs_resume(inputs: Mapping[str, Any]) -> Dict[str, Any]:
    likert_vals = [v for v in (inputs.get("likert", {}) or {}).values() if isinstance(v, (int, float))]
    texts = inputs.get("texts", []) or []
    return {
        "nb_skills_selected": len(inputs.get("selected_skills", []) or []),
        "nb_tools": len(inputs.get("tools", []) or []),
        "avg_auto_eval": round(mean(likert_vals), 2) if likert_vals else 0.0,
        "nb_texts": len(texts),
        "total_text_len": sum(len(t) for t in texts),
    }


def _generate_texts(top_job: Dict[str, Any] | None, top_skills: Sequence[Dict[str, Any]], use_ollama: bool = True) -> Tuple[str, str]:
    job_label = top_job.get("job_name") if top_job else "profil"
    skill_snippet = ", ".join([s.get("skill_name", "") for s in top_skills])

    if use_ollama and is_ollama_available():
        try:
            prompt_bio = (
                "Rédige une bio professionnelle concise (4-6 lignes) en français."
                " Base-toi uniquement sur le métier et les compétences listées."
                " Style: clair, à la première personne."
                f"\nMétier recommandé: {job_label}\nCompétences fortes: {skill_snippet}"
            )
            bio = generate_with_cache("bio_v2", prompt_bio)

            prompt_exp = (
                "Explique en 3 phrases pourquoi ce métier est recommandé."
                " Cite 3 compétences fortes et précise le niveau (Junior/Mid/Senior)."
                f"\nMétier: {job_label}\nCompétences fortes: {skill_snippet}"
            )
            explanation = generate_with_cache("exp_v2", prompt_exp)
            return bio, explanation
        except GenAIUnavailable:
            pass

    fallback_bio = (
        f"Je me positionne sur le rôle {job_label}. "
        f"Mes forces principales: {skill_snippet}. "
        "Je cherche à capitaliser sur ces compétences et à les renforcer par des projets concrets."
    )
    fallback_exp = (
        f"Le métier {job_label} ressort car tes compétences requises sont majoritairement couvertes, "
        "avec un mix auto-évaluation, sélection explicite et signaux texte."
    )
    return fallback_bio, fallback_exp


def run_analysis_v2(inputs: Mapping[str, Any], jobs: Sequence[Dict[str, Any]] | None = None, skills: Sequence[Dict[str, Any]] | None = None, use_ollama: bool = True) -> Dict[str, Any]:
    start = time.time()

    structured_inputs = inputs if "likert" in inputs else build_inputs_from_session_state(inputs)
    skills_data = list(skills) if skills is not None else load_skills()
    jobs_data = list(jobs) if jobs is not None else load_jobs()

    skill_scores = compute_skill_scores(structured_inputs, skills_data)
    job_scores = compute_job_scores(skill_scores, jobs_data)
    top_job = job_scores[0] if job_scores else None
    coverage = compute_coverage(top_job, skill_scores)
    level = classify_level(top_job.get("score", 0.0)) if top_job else "N/A"

    bio, explanation = _generate_texts(top_job, skill_scores[:5], use_ollama=use_ollama)

    duration = time.time() - start
    input_sig = compute_input_signature(structured_inputs)

    summary_profile = {
        "title": top_job.get("job_name", "Profil") if top_job else "Profil non défini",
        "level": level,
        "short_explanation": explanation,
    }

    analysis = {
        "meta": {
            "version": "scoring_v2",
            "run_at": datetime.now().isoformat(),
            "input_signature": input_sig,
            "duration_sec": round(duration, 3),
            "use_ollama": bool(use_ollama and is_ollama_available()),
        },
        "inputs_resume": _inputs_resume(structured_inputs),
        "job_scores": job_scores,
        "skill_scores": skill_scores,
        "coverage_score": coverage,
        "summary_profile": summary_profile,
        "professional_bio": bio,
        "debug": {
            "input_signature": input_sig,
            "top_job": top_job,
            "top_jobs": job_scores[:5],
            "top_skills": skill_scores[:10],
            "inputs": structured_inputs,
        },
    }
    return analysis


def self_check():
    jobs = load_jobs()
    skills = load_skills()

    profiles = {
        "data_analyst": {
            "likert": {
                "da_analysis_level": 5,
                "da_viz_level": 5,
                "da_biz_level": 4,
                "python_level": 3,
                "comms_level": 4,
            },
            "selected_skills": ["sql", "tableau", "data viz"],
            "tools": ["sql", "power bi"],
            "texts": ["Tableaux de bord SQL/Tableau, définition de KPI produit"],
            "experiences": [],
        },
        "mlops_engineer": {
            "likert": {
                "python_level": 5,
                "ml_level": 5,
                "etl_level": 4,
                "comms_level": 3,
            },
            "selected_skills": ["docker", "ci cd", "mlflow", "kubernetes"],
            "tools": ["docker", "k8s", "mlflow"],
            "texts": ["Mise en prod de modèles avec Docker, CI/CD, suivi MLflow, monitoring"],
            "experiences": [],
        },
        "backend_engineer": {
            "likert": {
                "python_level": 5,
                "da_analysis_level": 3,
                "comms_level": 3,
            },
            "selected_skills": ["api", "auth", "tests", "database"],
            "tools": ["fastapi", "postgres"],
            "texts": ["APIs REST sécurisées (JWT), tests unitaires, design système léger"],
            "experiences": [],
        },
    }

    for name, profile in profiles.items():
        result = run_analysis_v2(profile, jobs=jobs, skills=skills, use_ollama=False)
        top_job_id = (result.get("job_scores") or [{}])[0].get("job_id")
        print(name, "=>", top_job_id)


__all__ = [
    "load_jobs",
    "load_skills",
    "compute_input_signature",
    "build_inputs_from_session_state",
    "compute_skill_scores",
    "compute_job_scores",
    "compute_coverage",
    "classify_level",
    "run_analysis_v2",
    "self_check",
]
