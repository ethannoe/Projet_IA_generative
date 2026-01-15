import re
from typing import List, Optional

from src.genai_client import DEFAULT_MODEL, GenAIUnavailable, generate_with_cache, is_ollama_available


def build_linkedin_bio_prompt(job_name: str, top_skills: List[str], optional_context: str) -> str:
    job_part = job_name or "profil data"
    skills_part = ", ".join(top_skills[:6]) if top_skills else "compétences data"
    context_part = optional_context.strip() if optional_context else ""

    return (
        "Tu rédiges une bio LinkedIn (section À propos) en français, à la première personne.\n"
        "Longueur STRICTE : entre 90 et 110 mots, ne jamais dépasser 110 mots.\n"
        "1 seul paragraphe, pas de titres, pas de listes, pas d'emojis.\n"
        "Structure : (1) accroche en 1 phrase, (2) expertise/compétences/stack et types de projets en 3-4 phrases,\n"
        "(3) valeur apportée en 1 phrase, (4) phrase de conclusion orientée contact en 1 phrase.\n"
        "Si le texte dépasse 110 mots, raccourcis-le automatiquement pour rester dans la limite.\n"
        "Utilise uniquement les infos fournies, indique brièvement si une info manque sans inventer.\n"
        f"Métier ou rôle ciblé : {job_part}\n"
        f"Compétences / forces clés : {skills_part}\n"
        f"Contexte additionnel : {context_part}\n"
        "Style : première personne (je), ton professionnel et humain, clair et direct."
    )


def _clean_output(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"^(Résumé|Profil|Bio)\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _truncate_words(text: str, max_words: int = 110) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    truncated = " ".join(words[:max_words]).rstrip(",;: ")
    if not truncated.endswith(('.', '!', '?')):
        truncated = truncated.rstrip('.!?') + "."
    return truncated


def generate_linkedin_bio_ollama(
    job_name: str,
    top_skills: List[str],
    optional_context: str,
    model: Optional[str] = None,
) -> str:
    if not is_ollama_available():
        raise GenAIUnavailable("Ollama indisponible")

    prompt = build_linkedin_bio_prompt(job_name, top_skills, optional_context)
    model_to_use = model or DEFAULT_MODEL
    output = generate_with_cache("linkedin_bio_v3", prompt, model=model_to_use)
    cleaned = _clean_output(output)
    return _truncate_words(cleaned, max_words=110)


__all__ = [
    "build_linkedin_bio_prompt",
    "generate_linkedin_bio_ollama",
]
