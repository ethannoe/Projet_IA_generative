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


def init_state():
    if "responses" not in st.session_state:
        st.session_state["responses"] = {}
    if "likert" not in st.session_state["responses"]:
        st.session_state["responses"]["likert"] = {k: 3 for k in LIKERT_LABELS}


def likert_row(key: str, label: str):
    current = st.session_state["responses"]["likert"].get(key, 3)
    st.markdown(f"**{label}**")
    choice = st.radio(
        "Niveau",
        [1, 2, 3, 4, 5],
        index=(current - 1) if current in [1, 2, 3, 4, 5] else 2,
        key=f"{key}_radio",
        horizontal=True,
        help="1 = Débutant, 5 = Expert",
        label_visibility="collapsed",
    )
    st.session_state["responses"]["likert"][key] = choice


def main():
    inject_css()
    init_state()
    stepper(steps, 1)

    st.markdown("### Auto-évaluation (1 à 5) – Débutant ➜ Expert")
    for k, label in LIKERT_LABELS.items():
        likert_row(k, label)
        st.markdown("<hr style='border: none; height: 1px; background: #e6ebf1;'>", unsafe_allow_html=True)

    st.page_link("pages/3_Experiences.py", label="Étape suivante : Expériences")


if __name__ == "__main__":
    main()
