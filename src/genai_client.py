import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, Optional

import requests

from . import cache

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "phi3:mini"
OLLAMA_ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")


class GenAIUnavailable(Exception):
    pass


def is_ollama_available() -> bool:
    try:
        resp = requests.get(f"{OLLAMA_ENDPOINT}/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _ollama_generate(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.4) -> str:
    payload = {"model": model, "prompt": prompt, "temperature": temperature}
    start = time.time()
    try:
        resp = requests.post(f"{OLLAMA_ENDPOINT}/api/generate", json=payload, timeout=120, stream=False)
        resp.raise_for_status()
    # Ollama renvoie des chunks en streaming ; api/generate retourne du texte avec des lignes `data: {...}`
        text_parts = []
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
                if "response" in obj:
                    text_parts.append(obj["response"])
            except json.JSONDecodeError:
                continue
        output = "".join(text_parts).strip()
        duration = time.time() - start
        LOGGER.info(
            "Ollama generate ok | model=%s | prompt_len=%s | duration=%.2fs",
            model,
            len(prompt),
            duration,
        )
        return output
    except Exception as e:
        duration = time.time() - start
        LOGGER.error("Ollama generation failed: %s", e)
        LOGGER.info(
            "Ollama generate failed | model=%s | prompt_len=%s | duration=%.2fs",
            model,
            len(prompt),
            duration,
        )
        raise GenAIUnavailable(str(e))


def _llama_cpp_fallback(prompt: str) -> str:
    # Repli minimal via llama.cpp en ligne de commande si disponible dans le PATH
    cli = os.environ.get("LLAMACPP_CLI", "llama-cli")
    try:
        out = subprocess.check_output([cli, "-p", prompt], timeout=60)
        return out.decode("utf-8")
    except Exception as e:
        LOGGER.error("llama.cpp fallback failed: %s", e)
        raise GenAIUnavailable("Ni Ollama ni llama.cpp n'ont été trouvés. Installez Ollama ou configurez llama.cpp.")


def generate_with_cache(task: str, prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.4) -> str:
    key = cache.cache_json_key({"task": task, "prompt": prompt, "model": model, "temperature": temperature})
    cached = cache.cache_get(cache.GENAI_NAMESPACE, key)
    if cached:
        LOGGER.info("GenAI cache hit | task=%s | model=%s | prompt_len=%s", task, model, len(prompt))
        return cached

    start = time.time()
    if is_ollama_available():
        output = _ollama_generate(prompt, model=model, temperature=temperature)
    else:
        LOGGER.error("Ollama not available, attempting llama.cpp fallback")
        output = _llama_cpp_fallback(prompt)
    duration = time.time() - start
    LOGGER.info("GenAI generate done | task=%s | model=%s | duration=%.2fs", task, model, duration)
    cache.cache_set(cache.GENAI_NAMESPACE, key, output)
    return output


# Modèles de prompts avec consignes anti-hallucination

def build_enrichment_prompt(user_text: str, context: str) -> str:
    return (
        "Tu es un assistant qui réécrit une phrase trop courte pour clarifier l'intention professionnelle.\n"
        "Ne rajoute aucune compétence qui n'est pas mentionnée.\n"
        "Si l'information manque, indique-le explicitement.\n"
        f"Phrase utilisateur: {user_text}\n"
        f"Contexte référentiel: {context}\n"
        "Réécris en 1 à 2 phrases factuelles, sans extrapoler."
    )


def build_plan_prompt(plan_context: Dict[str, Any]) -> str:
    """Construit un prompt data-driven pour le plan de progression.

    Attend un dict plan_context contenant :
    - job: {job_name, job_id, score, coverage}
    - top_skills: list[{skill_name, score}]
    - weak_skills: list[{skill_name, score}]
    - threshold: float (pour marquer les faiblesses)
    """

    job = plan_context.get("job", {}) or {}
    top_skills = plan_context.get("top_skills", []) or []
    weak_skills = plan_context.get("weak_skills", []) or []
    coverage = job.get("coverage")
    threshold = plan_context.get("threshold")

    def fmt_skills(skills):
        return "\n".join([f"- {s.get('skill_name', 'compétence')} ({s.get('score', 0):.1f}%)" for s in skills])

    top_block = fmt_skills(top_skills)
    weak_block = fmt_skills(weak_skills)

    coverage_txt = f"{coverage:.1f}%" if coverage is not None else "N/A"
    threshold_txt = f"{threshold:.0f}%" if threshold is not None else "60%"

    return (
        "Tu es coach carrière tech. Génère un plan de progression strictement aligné sur les résultats fournis.\n"
        "Règles: utilise uniquement les données données (métier, forces, faiblesses), aucune généralité.\n"
        "Langue: français. Ton: concret et actionnable.\n"
        "Structure obligatoire:\n"
        "- Objectif (métier cible)\n"
        "- Forces actuelles (3 points max)\n"
        "- Axes prioritaires (faiblesses principales)\n"
        "- Plan 30 jours / 60 jours / 90 jours (actions liées aux faiblesses)\n"
        "- Mini-projets recommandés (2 projets, chacun lié à une faiblesse)\n"
        "- Ressources/Exercices (types uniquement, pas de liens)\n"
        "Contraintes: chaque action doit citer la compétence faible ciblée; interdits: conseils vagues.\n"
        f"Métier cible: {job.get('job_name', 'N/A')} (score {job.get('score', 0):.1f}%, coverage {coverage_txt})\n"
        f"Forces (top compétences):\n{top_block}\n"
        f"Faiblesses prioritaires (< seuil {threshold_txt} ou plus faibles):\n{weak_block}\n"
        "Génère le plan maintenant en suivant la structure."
    )


def build_bio_prompt(context_pack: str, top_jobs: Any, strengths: Dict[str, float]) -> str:
    return (
        "Tu rédiges une bio LinkedIn (section À propos) en français, à la première personne, ton professionnel et humain.\n"
        "Exigences : 120 à 200 mots, ouverture type headline percutante, phrases fluides.\n"
        "Inclure: rôle/positionnement, secteurs ou types de projets, compétences et forces clés, réalisations ou impacts, ambitions/motivations.\n"
        "Utilise uniquement les informations fournies (ne rien inventer). Si une info manque, mentionne-le brièvement sans broder.\n"
        f"Contexte RAG:\n{context_pack}\n"
        f"Métiers recommandés: {top_jobs}\n"
        f"Forces par bloc ou compétences fortes: {strengths}\n"
        "Style: première personne (je), clair, engageant, sans liste à puces."
    )


__all__ = [
    "generate_with_cache",
    "build_enrichment_prompt",
    "build_plan_prompt",
    "build_bio_prompt",
    "GenAIUnavailable",
    "is_ollama_available",
]
