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
        # Ollama returns streaming chunks; api/generate returns text with newlines containing `data: {...}`
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
    # Minimal fallback using llama.cpp cli if available in PATH
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


# Prompt templates with anti-hallucination instructions

def build_enrichment_prompt(user_text: str, context: str) -> str:
    return (
        "Tu es un assistant qui réécrit une phrase trop courte pour clarifier l'intention professionnelle.\n"
        "Ne rajoute aucune compétence qui n'est pas mentionnée.\n"
        "Si l'information manque, indique-le explicitement.\n"
        f"Phrase utilisateur: {user_text}\n"
        f"Contexte référentiel: {context}\n"
        "Réécris en 1 à 2 phrases factuelles, sans extrapoler."
    )


def build_plan_prompt(context_pack: str, missing_skills: Dict[str, Any]) -> str:
    return (
        "Tu es un coach IA. Génère un plan de progression structuré (4 à 8 semaines) basé sur les écarts détectés.\n"
        "Utilise uniquement les compétences du référentiel et les écarts fournis.\n"
        "Si une info est manquante, dis-le.\n"
        "Format: étapes hebdomadaires avec objectifs, activités, jalons.\n"
        f"Contexte RAG:\n{context_pack}\n"
        f"Compétences à renforcer: {json.dumps(missing_skills, ensure_ascii=False)}\n"
        "Réponds en français, concis, listes à puces."
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
