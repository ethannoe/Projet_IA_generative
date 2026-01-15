import streamlit as st

from ui.components import inject_css, stepper

st.set_page_config(page_title="AISCA - Introduction", layout="wide")

steps = [
    "Introduction",
    "Auto-évaluation",
    "Expériences",
    "Compétences",
    "Analyse",
    "Résultats",
]


def main():
    inject_css()
    stepper(steps, 0)
    st.markdown(
        """
        <div class="hero">
          <h1>Découvrez votre profil Data & IA</h1>
          <p>Répondez en quelques minutes, laissez l'analyse sémantique et le RAG guider les recommandations métiers et votre plan de progression.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    cards = [
        ("Analyse Sémantique", "Embeddings SBERT et similarité cosinus sur votre référentiel."),
        ("Recommandation Métiers", "Matching par blocs pondérés, transparent et explicable."),
        ("Plan de progression", "Génération locale via Ollama, 1 appel, contenu mis en cache."),
    ]
    for col, (title, desc) in zip(cols, cards):
        with col:
            st.markdown(f"<div class='card'><h3>{title}</h3><p>{desc}</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='center'>", unsafe_allow_html=True)
    st.page_link("pages/2_Auto_evaluation.py", label="Commencer l'auto-évaluation")
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
