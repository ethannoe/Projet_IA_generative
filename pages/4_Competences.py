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

TOOLS = ["pandas", "sql", "spark", "mlflow", "airflow", "powerbi", "tableau", "streamlit"]
CHECKBOX_SKILLS = [
    "feature engineering (création de variables)",
    "analyse exploratoire (EDA)",
    "visualisation de données",
    "pipeline Airflow",
    "suivi d'expériences MLflow",
    "embeddings NLP",
    "dashboard Streamlit",
]


def init_state():
    if "responses" not in st.session_state:
        st.session_state["responses"] = {}
    st.session_state["responses"].setdefault("guided", {})


def main():
    inject_css()
    init_state()
    stepper(steps, 3)

    st.markdown("### Compétences et contexte")
    st.info("Ces éléments complètent l'auto-évaluation pour affiner le matching métier.")
    qcm_single = st.selectbox(
        "Fréquence d'usage des données",
        ["quotidienne", "hebdomadaire", "mensuelle", "rarement"],
        index=0,
        key="qcm_single",
    )
    qcm_multi = st.multiselect("Outils principaux", TOOLS, default=["pandas", "sql"], key="qcm_multi")
    checkbox_skills = st.multiselect("Compétences revendiquées", CHECKBOX_SKILLS, default=["analyse exploratoire (EDA)", "dashboard Streamlit"], key="checkbox_skills")

    st.markdown("#### Questions guidées (oui/non)")
    guided = {
        "has_prod_model": st.radio("Avez-vous déjà mis un modèle en production?", ["oui", "non"], index=1, key="has_prod_model"),
        "works_team": st.radio("Travaillez-vous en équipe data?", ["oui", "non"], index=0, key="works_team"),
        "big_data": st.radio("As-tu déjà travaillé avec des datasets volumineux (plus d’un million de lignes) ?", ["oui", "non"], index=1, key="big_data"),
        "prod_app": st.radio("As-tu déjà déployé une application data ou un modèle IA en production ?", ["oui", "non"], index=1, key="prod_app"),
        "advanced_nlp": st.radio("As-tu utilisé des bibliothèques NLP avancées (spaCy, HuggingFace, transformers) ?", ["oui", "non"], index=1, key="advanced_nlp"),
        "collab_business": st.radio("As-tu déjà collaboré avec des équipes produit, métier ou business ?", ["oui", "non"], index=0, key="collab_business"),
    }

    st.session_state["responses"].update(
        {
            "qcm_single": qcm_single,
            "qcm_multi": qcm_multi,
            "checkbox_skills": checkbox_skills,
            "guided": guided,
        }
    )

    st.page_link("pages/3_Experiences.py", label="Retour : Expériences")
    st.page_link("pages/5_Analyse.py", label="Étape suivante : Analyse")


if __name__ == "__main__":
    main()
