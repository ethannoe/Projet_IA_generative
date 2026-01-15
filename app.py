import os
import streamlit as st

from ui.components import inject_css

st.set_page_config(page_title="AISCA - Evaluation Data & IA", layout="wide")


def init_state():
    if "responses" not in st.session_state:
        st.session_state["responses"] = {}


def main():
    inject_css()
    init_state()
    st.markdown(
        """
        <div class="hero">
          <h1>Découvrez votre profil Data & IA</h1>
          <p>Évaluez vos compétences, obtenez des recommandations métiers et un plan de progression personnalisé.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Commencer l'évaluation"):
        try:
            st.switch_page("pages/1_Introduction.py")
        except Exception:
            st.error("Page d'introduction introuvable. Vérifie que 'pages/1_Introduction.py' existe.")

    st.markdown("### Parcours")
    cols = st.columns(3)
    cards = [
        ("Analyse Sémantique", "Embeddings SBERT locaux et RAG pour guider le LLM."),
        ("Recommandation Métiers", "Matching par blocs de compétences pondérés."),
        ("Plan de progression", "Génération locale et contextualisée, 1 seul appel."),
    ]
    for col, (title, desc) in zip(cols, cards):
        with col:
            st.markdown(f"<div class='card'><h3>{title}</h3><p>{desc}</p></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
