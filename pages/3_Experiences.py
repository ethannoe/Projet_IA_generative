import streamlit as st

from ui.components import inject_css, stepper

steps = [
    "Introduction",
    "Auto-évaluation",
    "Expériences",
    "Compétences",
    "Analyse",
    "Résultats",
]

TEXT_KEYS = [
    ("free_text", "Décrivez vos projets data/NLP récents"),
    ("ambitions", "Objectifs de carrière"),
    (
        "proj_tech",
        "Projet technique structurant (contexte, objectif, techno, résultats)",
    ),
    (
        "prob_complex",
        "Résolution de problème complexe (données / ML / IA)",
    ),
    (
        "model_build",
        "Conception ou amélioration de modèles ML/NLP (démarche)",
    ),
    ("projection", "Projection professionnelle (rôles / missions visées)"),
]


def init_state():
    if "responses" not in st.session_state:
        st.session_state["responses"] = {}


def main():
    inject_css()
    init_state()
    stepper(steps, 2)
    st.markdown("### Expériences et réalisations")

    for key, label in TEXT_KEYS:
        default_val = st.session_state["responses"].get(key, "")
        val = st.text_area(label, value=default_val, height=120)
        st.session_state["responses"][key] = val.strip()

    st.page_link("pages/2_Auto_evaluation.py", label="Retour : Auto-évaluation")
    st.page_link("pages/4_Competences.py", label="Étape suivante : Compétences")


if __name__ == "__main__":
    main()
