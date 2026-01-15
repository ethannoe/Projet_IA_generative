import hashlib
import json
import os
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd
import streamlit as st

from src.utils.scoring_v2 import (
    ScoringInputsV2,
    load_jobs_v2,
    load_skills_v2,
    run_scoring_v2,
)
from ui.components import inject_css, stepper

steps = [
    "Introduction",
    "Auto-évaluation",
    "Expériences",
    "Compétences",
    "Analyse",
    "Résultats",
]

LIKERT_LABELS = {
    "python_level": "Maîtrise Python / Pandas",
    "ml_level": "Connaissance ML",
    "nlp_level": "Connaissance NLP",
    "pm_level": "Pratiques projet / agile",
    "etl_level": "Pipelines de données (ETL, preprocessing, feature engineering)",
    "comms_level": "Communication vers non-tech",
    "da_analysis_level": "Exploration et analyse de données (Python/SQL)",
    "da_viz_level": "Visualisation / reporting",
    "da_biz_level": "Interprétation métier",
}

TEXT_FIELDS = {
    "free_text": "Décrivez vos projets data/NLP récents",
    "ambitions": "Objectifs de carrière",
    "proj_tech": "Projet technique structurant",
    "prob_complex": "Résolution de problème complexe (données / ML / IA)",
    "model_build": "Conception ou amélioration de modèles ML/NLP",
    "projection": "Projection professionnelle (rôles / missions visées)",
}

GUIDED_LABELS = {
    "has_prod_model": "A déjà mis un modèle en production",
    "works_team": "Travaille en équipe data",
    "big_data": "A déjà travaillé avec des datasets volumineux",
    "prod_app": "A déjà déployé une application data ou IA",
    "advanced_nlp": "A utilisé des bibliothèques NLP avancées",
    "collab_business": "Collabore avec des équipes produit/métier",
}


@st.cache_data
def load_jobs_cached_v2():
    return load_jobs_v2()


@st.cache_data
def load_skills_cached_v2():
    return load_skills_v2()


def init_state():
    if "responses" not in st.session_state:
        st.session_state["responses"] = {}
    if "analysis_result_v2" not in st.session_state:
        st.session_state["analysis_result_v2"] = None
    if "analysis_error" not in st.session_state:
        st.session_state["analysis_error"] = None
    if "analysis_input_signature_v2" not in st.session_state:
        st.session_state["analysis_input_signature_v2"] = None


def build_user_texts(responses: Dict) -> List[str]:
    texts: List[str] = []
    # Likert
    for key, label in LIKERT_LABELS.items():
        score = responses.get("likert", {}).get(key)
        if score:
            texts.append(f"{label}: niveau {score}/5")
    # QCM / cases à cocher
    if responses.get("qcm_single"):
        texts.append(f"Fréquence d'usage des données: {responses['qcm_single']}")
    if responses.get("qcm_multi"):
        texts.append("Outils principaux: " + ", ".join(responses["qcm_multi"]))
    if responses.get("checkbox_skills"):
        texts.append("Compétences revendiquées: " + ", ".join(responses["checkbox_skills"]))
    # Questions guidées oui/non
    for key, label in GUIDED_LABELS.items():
        val = responses.get("guided", {}).get(key)
        if val:
            texts.append(f"{label}: {val}")
    # Textes libres
    for key, label in TEXT_FIELDS.items():
        val = responses.get(key, "").strip()
        if val:
            texts.append(f"{label}: {val}")
    return [t for t in texts if t.strip()]


def persist_responses(responses: Dict):
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join("outputs", f"user_responses_{timestamp}.json")
    csv_path = os.path.join("outputs", f"user_responses_{timestamp}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    # flatten for csv
    flat = []
    for k, v in responses.items():
        flat.append({"question": k, "response": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v})
    pd.DataFrame(flat).to_csv(csv_path, index=False)
    return json_path, csv_path


def compute_signature(responses: Dict) -> str:
    try:
        payload = json.dumps(responses, sort_keys=True, ensure_ascii=False)
    except TypeError:
        payload = str(responses)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_analysis():
    responses = st.session_state.get("responses", {})
    signature = compute_signature(responses)

    jobs = load_jobs_cached_v2()
    skills = load_skills_cached_v2()

    if not responses:
        raise ValueError("Aucun contenu saisi. Merci de renseigner les sections précédentes.")

    start_time = time.time()
    scoring_inputs = ScoringInputsV2(responses=responses, jobs=jobs, skills=skills, use_ollama=True)
    scores = run_scoring_v2(scoring_inputs)
    duration = time.time() - start_time

    result = {
        **scores,
        "meta": {**scores.get("meta", {}), "input_signature": signature, "duration_sec": duration},
    }

    persist_responses(responses)

    st.session_state["analysis_result_v2"] = result
    st.session_state["analysis_input_signature_v2"] = signature
    # Nettoyage des anciennes clés pour éviter les collisions
    for legacy in ["analysis", "analysis_result", "analysis_signature", "analysis_input_signature"]:
        if legacy in st.session_state:
            del st.session_state[legacy]


def main():
    inject_css()
    init_state()
    stepper(steps, 4)

    st.markdown("### Analyse personnalisée")
    st.write(
        "L'analyse v2 calcule un scoring déterministe (compétences + métiers) et génère un résumé."
    )

    if st.button("Lancer l'analyse", type="primary"):
        with st.spinner("Calcul des embeddings et des scores..."):
            try:
                run_analysis()
                st.session_state["analysis_error"] = None
                st.success("Analyse terminée. Rendez-vous sur la page Résultats.")
            except Exception as e:  # noqa: BLE001
                st.session_state["analysis_error"] = str(e)
                st.error(str(e))

    if st.session_state.get("analysis_result_v2"):
        st.info("Analyse disponible. Tu peux consulter la page Résultats ou relancer avec de nouvelles réponses.")

    if st.session_state.get("analysis_error"):
        st.error(st.session_state["analysis_error"])

    st.page_link("pages/4_Competences.py", label="Retour : Compétences")
    st.page_link("pages/6_Resultats.py", label="Étape suivante : Résultats")


if __name__ == "__main__":
    main()
