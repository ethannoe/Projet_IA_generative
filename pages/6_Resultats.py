import hashlib
import html
import json
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go

from src.genai_client import (
    GenAIUnavailable,
    build_plan_prompt,
    generate_with_cache,
    is_ollama_available,
)
from src.utils.bio_ollama import generate_linkedin_bio_ollama
from src.viz import bar_top_jobs
from ui.components import inject_css, stepper

steps = [
    "Introduction",
    "Auto-évaluation",
    "Expériences",
    "Compétences",
    "Analyse",
    "Résultats",
]


BLOCK_LABELS = {
    "analytics": "Analytics / Viz",
    "ml": "Machine Learning",
    "mlops": "MLOps / Ops",
    "data_eng": "Data Engineering",
    "nlp": "NLP",
    "engineering": "Python / Dev",
    "soft": "Communication / Produit",
}

SKILL_TO_BLOCK = {
    "sql": "analytics",
    "data_viz": "analytics",
    "statistics": "analytics",
    "bi_tools": "analytics",
    "eda": "analytics",
    "python": "engineering",
    "ml": "ml",
    "nlp": "nlp",
    "mlops": "mlops",
    "docker": "mlops",
    "cicd": "mlops",
    "monitoring": "mlops",
    "mlflow": "mlops",
    "etl": "data_eng",
    "airflow": "data_eng",
    "spark": "data_eng",
    "communication": "soft",
    "product": "soft",
}


def init_state():
    if "analysis_result_v2" not in st.session_state:
        st.session_state["analysis_result_v2"] = None
    if "responses" not in st.session_state:
        st.session_state["responses"] = {}
    if "plan_output" not in st.session_state:
        st.session_state["plan_output"] = None
    if "professional_bio_text" not in st.session_state:
        st.session_state["professional_bio_text"] = None
    if "professional_bio_status" not in st.session_state:
        st.session_state["professional_bio_status"] = "idle"
    if "professional_bio_error" not in st.session_state:
        st.session_state["professional_bio_error"] = None
def compute_signature(responses):
    try:
        payload = json.dumps(responses, sort_keys=True, ensure_ascii=False)
    except TypeError:
        payload = str(responses)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def debug_enabled():
    params = st.query_params
    gate = params.get("debug", ["0"])[0] == "1" or st.session_state.get("debug", False)
    if gate:
        st.session_state["debug"] = True
        default = st.session_state.get("debug_mode", True)
        debug_mode = st.toggle("Scoring debug mode", value=default, help="Affiche les breakdowns métiers/compétences")
        st.session_state["debug_mode"] = debug_mode
        return debug_mode
    return False

def render_jobs_table(job_scores):
    for job in job_scores:
        with st.expander(f"{job.get('job_name')} – score {job.get('score', 0):.1f}"):
            st.write(f"Couverture compétences requises : {job.get('coverage', 0):.1f}%")
            if job.get("required_skills"):
                st.markdown("- Compétences requises : " + ", ".join(job.get("required_skills", [])))
            if job.get("optional_skills"):
                st.markdown("- Compétences optionnelles : " + ", ".join(job.get("optional_skills", [])))


def render_debug_panel(analysis):
    meta = analysis.get("meta", {})
    debug = analysis.get("debug", {}) or {}
    skill_scores = debug.get("skill_scores", [])
    job_scores = debug.get("job_scores", [])

    st.markdown("### Debug résultats")
    st.json(
        {
            "meta": meta,
            "inputs_resume": analysis.get("inputs_resume", {}),
            "nb_skill_scores": len(skill_scores),
            "nb_job_scores": len(job_scores),
        },
        expanded=False,
    )

    if job_scores:
        st.markdown("#### Top métiers (brut)")
        st.dataframe(job_scores[:5], hide_index=True, use_container_width=True)

    if skill_scores:
        st.markdown("#### Top compétences (brut)")
        st.dataframe(skill_scores[:10], hide_index=True, use_container_width=True)


def aggregate_block_scores(skill_scores):
    buckets = {}
    for s in skill_scores or []:
        block = SKILL_TO_BLOCK.get(s.get("skill_id"))
        if not block:
            continue
        buckets.setdefault(block, []).append(s.get("score", 0) or 0)
    averages = {b: (sum(vals) / len(vals)) for b, vals in buckets.items() if vals}
    # Limiter à 8 axes max
    return sorted(averages.items(), key=lambda kv: kv[1], reverse=True)[:8]


def render_radar(skill_scores):
    block_scores = aggregate_block_scores(skill_scores)
    if not block_scores:
        st.info("Radar indisponible (aucune compétence agrégée).")
        return

    labels = [BLOCK_LABELS.get(b, b) for b, _ in block_scores]
    values = [v for _, v in block_scores]
    if values and max(values) <= 1.0:
        values = [v * 100 for v in values]

    if len(values) == 1:
        # Dupliquer pour éviter un seul point
        labels = labels * 2
        values = values * 2

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=labels + [labels[0]], fill="toself"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def fmt_num(x) -> str:
    try:
        if x is None:
            return ""
        if isinstance(x, (int, float)):
            return f"{float(x):.2f}"
    except Exception:
        pass
    return str(x)


def format_plan_to_html(plan_text: str) -> str:
    if not plan_text:
        return "Non généré"

    cleaned_text = plan_text.replace("**", "").replace("*", "")
    lines = [ln.strip("\u2022 ") for ln in cleaned_text.splitlines() if ln.strip()]
    blocks = []
    current_list = None
    list_type = None

    def flush_list():
        nonlocal current_list, list_type, blocks
        if current_list:
            if list_type == "ul":
                blocks.append("<ul>" + "".join([f"<li>{html.escape(item)}</li>" for item in current_list]) + "</ul>")
            elif list_type == "ol":
                blocks.append("<ol>" + "".join([f"<li>{html.escape(item)}</li>" for item in current_list]) + "</ol>")
        current_list = None
        list_type = None

    for ln in lines:
        esc = html.escape(ln)
        # Headings
        if ln.endswith(":") or (len(ln) <= 40 and ln.lower() == ln.replace(":", "").lower() and ln.endswith(":")):
            flush_list()
            blocks.append(f"<h4>{esc}</h4>")
            continue

        # Ordered list detection
        if ln[:2].isdigit() and (ln[1:2] in [".", ")"]):
            if list_type != "ol":
                flush_list()
                current_list = []
                list_type = "ol"
            content = ln[2:].strip()
            current_list.append(content)
            continue

        # Unordered list detection
        if ln.startswith(("- ", "• ")) or (len(ln) > 1 and ln[0] in ["-", "•"] and ln[1] == " "):
            if list_type != "ul":
                flush_list()
                current_list = []
                list_type = "ul"
            content = ln[1:].strip() if ln.startswith(('-', '•')) else ln[2:].strip()
            current_list.append(content)
            continue

        # Paragraph
        flush_list()
        blocks.append(f"<p>{esc}</p>")

    flush_list()
    return "\n".join(blocks)


def build_results_html(analysis_data: dict, bio_text: str | None, plan_text: str | None) -> str:
        now = datetime.now().isoformat()
        meta = analysis_data.get("meta", {}) or {}
        signature = meta.get("input_signature") or meta.get("signature") or "N/A"
        version = meta.get("version", "N/A")
        duration = meta.get("duration_sec") or meta.get("duration")

        job_scores = analysis_data.get("job_scores", []) or []
        skill_scores = analysis_data.get("skill_scores", []) or []
        top_job = job_scores[0] if job_scores else {}
        coverage = analysis_data.get("coverage_score")

        # Tables
        jobs_rows = "".join(
            [
                f"<tr><td>{i+1}</td><td>{html.escape(str(j.get('job_name','')))}</td><td>{fmt_num(j.get('score'))}</td></tr>"
                for i, j in enumerate(job_scores)
            ]
        )
        skills_rows = "".join(
            [
                f"<tr><td>{i+1}</td><td>{html.escape(str(s.get('skill_name','')))}</td><td>{html.escape(str(s.get('block_id','')))}</td><td>{fmt_num(s.get('score'))}</td></tr>"
                for i, s in enumerate(skill_scores)
            ]
        )

        # Bloc scores si dispo via block_id dans skill_scores
        block_map = {}
        for s in skill_scores:
                block_id = s.get("block_id")
                if block_id:
                        block_map.setdefault(block_id, []).append(float(s.get("score", 0) or 0))
        block_rows = "".join(
                [
                        f"<tr><td>{html.escape(str(b))}</td><td>{fmt_num(sum(vals)/len(vals))}</td></tr>"
                        for b, vals in block_map.items()
                ]
        )

        bio_section = html.escape(bio_text) if bio_text else "Non générée"
        plan_section = format_plan_to_html(plan_text) if plan_text else "Non généré"

        raw_json = html.escape(json.dumps(analysis_data, ensure_ascii=False, indent=2, default=str))

        return f"""
<!DOCTYPE html>
<html lang='fr'>
<head>
    <meta charset='utf-8'/>
    <title>Rapport de résultats</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.5; }}
        h1, h2, h3 {{ margin-bottom: 8px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
        th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
        .kpis {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }}
        .card {{ border: 1px solid #ddd; padding: 10px 12px; border-radius: 6px; background: #fafafa; min-width: 160px; }}
        .section {{ margin-top: 20px; }}
        details {{ margin-top: 12px; }}
        pre {{ background: #f5f5f5; padding: 12px; overflow-x: auto; }}
            .plan {{ line-height: 1.6; }}
            .plan p {{ margin: 6px 0; }}
            .plan ul, .plan ol {{ margin: 6px 0 10px 18px; }}
            .plan h4 {{ margin: 10px 0 6px; }}
    </style>
</head>
<body>
    <h1>Rapport de résultats</h1>
    <p>Date d'extraction : {now}</p>
        <p>Méta : version={html.escape(str(version))}, signature={html.escape(str(signature))}, durée={fmt_num(duration)}</p>

    <div class='kpis'>
            <div class='card'><strong>Top métier</strong><br>{html.escape(str(top_job.get('job_name','N/A')))} ({fmt_num(top_job.get('score'))})</div>
            <div class='card'><strong>Coverage</strong><br>{fmt_num(coverage) if coverage is not None else 'N/A'}</div>
    </div>

    <div class='section'>
        <h2>Top métiers</h2>
        <table>
            <thead><tr><th>Rang</th><th>Métier</th><th>Score</th></tr></thead>
            <tbody>{jobs_rows}</tbody>
        </table>
    </div>

    <div class='section'>
        <h2>Top compétences</h2>
        <table>
            <thead><tr><th>Rang</th><th>Compétence</th><th>Bloc</th><th>Score</th></tr></thead>
            <tbody>{skills_rows}</tbody>
        </table>
    </div>

    {f"<div class='section'><h2>Scores par bloc</h2><table><thead><tr><th>Bloc</th><th>Score moyen</th></tr></thead><tbody>{block_rows}</tbody></table></div>" if block_rows else ''}

    <div class='section'>
        <h2>Bio professionnelle (Ollama)</h2>
        <p>{bio_section}</p>
    </div>

    <div class='section'>
        <h2>Plan de progression</h2>
            <div class='plan'>{plan_section}</div>
    </div>

    <div class='section'>
        <h2>Données brutes</h2>
        <details><summary>Voir données brutes</summary><pre>{raw_json}</pre></details>
    </div>

</body>
</html>
"""


def compute_missing_skills_for_plan(analysis):
    job_scores = analysis.get("job_scores", []) or []
    if not job_scores:
        return {}
    top_job = job_scores[0]
    skill_lookup = {s.get("skill_id"): s for s in (analysis.get("skill_scores", []) or [])}
    missing = []
    for sid in top_job.get("required_skills", []):
        score = skill_lookup.get(sid, {}).get("score", 0)
        if score < 50:
            missing.append(skill_lookup.get(sid, {}).get("skill_name", sid))
    return {"Compétences requises à renforcer": missing} if missing else {}


def build_plan_context(analysis):
    job_scores = analysis.get("job_scores", []) or []
    top_job = job_scores[0] if job_scores else {}
    top_skills = (analysis.get("skill_scores", []) or [])[:5]
    lines = [f"Métier recommandé: {top_job.get('job_name', 'N/A')} (score {top_job.get('score', 0):.1f})"]
    if top_skills:
        lines.append("Compétences fortes: " + ", ".join([f"{s.get('skill_name')} ({s.get('score',0):.0f})" for s in top_skills]))
    return "\n".join(lines)


def render_plan_section(analysis):
    st.markdown("### Plan de progression (Ollama)")
    if not is_ollama_available():
        st.warning("Plan de progression indisponible (Ollama non détecté).", icon="⚠️")
        return

    missing = compute_missing_skills_for_plan(analysis)
    context = build_plan_context(analysis)

    if st.button("Générer un plan de progression"):
        with st.spinner("Génération avec Ollama..."):
            try:
                prompt = build_plan_prompt(context, missing)
                st.session_state["plan_output"] = generate_with_cache("plan_progression_v2", prompt)
            except GenAIUnavailable as e:  # noqa: BLE001
                st.error(str(e))

    if st.session_state.get("plan_output"):
        with st.expander("Plan de progression généré", expanded=True):
            st.markdown(st.session_state["plan_output"])


def render_professional_bio_section(analysis):
    st.markdown("### Bio professionnelle (Ollama)")

    status = st.session_state.get("professional_bio_status", "idle")
    bio_text = st.session_state.get("professional_bio_text")
    bio_error = st.session_state.get("professional_bio_error")

    if not is_ollama_available():
        st.warning("Bio indisponible (Ollama non détecté).", icon="⚠️")
        return

    job_scores = analysis.get("job_scores", []) or []
    top_job = job_scores[0] if job_scores else {}
    job_name = top_job.get("job_name") or "Profil data"
    skill_names = [s.get("skill_name") for s in (analysis.get("skill_scores", []) or []) if s.get("skill_name")][:6]
    context_lines = [f"Profil recommandé: {job_name} ({top_job.get('score', 0):.1f})"] if job_name else []
    if skill_names:
        context_lines.append("Compétences fortes: " + ", ".join(skill_names))
    optional_context = "\n".join(context_lines)

    if st.button("Générer la bio", key="btn_gen_bio") and status in ("idle", "error") and not bio_text:
        st.session_state["professional_bio_status"] = "requested"
        st.session_state["professional_bio_error"] = None

    if st.session_state.get("professional_bio_status") == "requested":
        st.session_state["professional_bio_status"] = "generating"
        with st.spinner("Génération de la bio..."):
            try:
                bio = generate_linkedin_bio_ollama(job_name, skill_names, optional_context)
                st.session_state["professional_bio_text"] = bio
                st.session_state["professional_bio_status"] = "done"
                st.session_state["professional_bio_error"] = None
            except GenAIUnavailable as e:  # noqa: BLE001
                st.session_state["professional_bio_status"] = "error"
                st.session_state["professional_bio_error"] = str(e)
            st.rerun()

    if st.session_state.get("professional_bio_status") == "error" and bio_error:
        st.error(f"Impossible de générer la bio via Ollama : {bio_error}")

    bio_text = st.session_state.get("professional_bio_text")
    if st.session_state.get("professional_bio_status") == "done" and bio_text:
        with st.expander("Voir la bio", expanded=True):
            st.markdown(bio_text)


def main():
    inject_css()
    init_state()
    stepper(steps, 5)

    st.markdown("### Résultats")
    analysis = st.session_state.get("analysis_result_v2") or {}
    current_sig = compute_signature(st.session_state.get("responses", {}))
    meta = analysis.get("meta", {}) if analysis else {}
    analysis_sig = meta.get("input_signature")

    debug_mode = debug_enabled()

    if not analysis:
        st.info("Aucune analyse disponible. Lance l'analyse depuis la page précédente.")
        st.page_link("pages/5_Analyse.py", label="Aller à Analyse")
        return

    if analysis_sig and analysis_sig != current_sig:
        st.warning("Les réponses ont changé depuis la dernière analyse. Relance l'analyse pour mettre à jour les résultats.")
        st.page_link("pages/5_Analyse.py", label="Recalculer")

    coverage = analysis.get("coverage_score", 0.0)
    st.success("Analyse prête : scoring v2 déterministe.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Coverage score", f"{coverage:.1f}%")
    with c2:
        st.metric("Profil", analysis.get("summary_profile", {}).get("title", ""))
    with c3:
        top_job = (analysis.get("job_scores", []) or [None])[0]
        if top_job:
            st.metric("Métier le plus compatible", top_job.get("job_name"), delta=f"score {top_job.get('score', 0):.1f}")

    st.subheader("Répartition des compétences (radar)")
    render_radar(analysis.get("skill_scores", []))

    st.subheader("Top métiers compatibles")
    top_jobs_simple = [
        (job.get("job_name"), job.get("score", 0) / 100.0)
        for job in (analysis.get("job_scores", []) or [])[:3]
    ]
    if top_jobs_simple:
        st.plotly_chart(bar_top_jobs(top_jobs_simple), use_container_width=True)

    st.markdown("#### Top métiers (détails)")
    render_jobs_table((analysis.get("job_scores", []) or [])[:3])

    st.markdown("#### Top compétences détectées")
    for skill in (analysis.get("skill_scores", []) or [])[:8]:
        st.markdown(f"- **{skill.get('skill_name')}** ({skill.get('score', 0):.1f})")

    render_plan_section(analysis)
    render_professional_bio_section(analysis)

    if debug_mode:
        render_debug_panel(analysis)

    st.subheader("Export")
    analysis_data = st.session_state.get("analysis_result_v2")
    if not analysis_data:
        st.warning("Aucun résultat à exporter")
    else:
        bio_text = st.session_state.get("professional_bio_text")
        plan_text = st.session_state.get("plan_output")
        html_report = build_results_html(analysis_data, bio_text, plan_text)
        st.download_button(
            label="Extraire les données",
            data=html_report.encode("utf-8"),
            file_name="rapport_resultats.html",
            mime="text/html",
        )

    st.page_link("pages/5_Analyse.py", label="Retour : Analyse")


if __name__ == "__main__":
    main()
