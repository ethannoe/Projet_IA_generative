import streamlit as st
from typing import List


def inject_css():
    try:
        with open("ui/styles.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Fichier CSS introuvable (ui/styles.css)")


def stepper(steps: List[str], active_index: int):
    st.markdown('<div class="stepper">' + "".join(_step_html(steps, active_index)) + "</div>", unsafe_allow_html=True)


def _step_html(steps: List[str], active_index: int):
    items = []
    for idx, label in enumerate(steps):
        state = ""
        if idx == active_index:
            state = "active"
        elif idx < active_index:
            state = "done"
        badge = f"<span class='badge'>{idx+1}</span>"
        items.append(f"<div class='step {state}'>{badge}<span>{label}</span></div>")
    return items
