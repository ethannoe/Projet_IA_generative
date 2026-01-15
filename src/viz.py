from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def radar_blocks(block_scores: Dict[str, float], block_names: Dict[str, str]):
    items = sorted(block_scores.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    display_labels = [block_names.get(k, k) for k in labels]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(r=values + [values[0]], theta=display_labels + [display_labels[0]], fill="toself")
    )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
    return fig


def bar_top_jobs(top_jobs: List[Tuple[str, float]]):
    jobs = [j for j, _ in top_jobs]
    scores = [s for _, s in top_jobs]
    df = pd.DataFrame({"Métier": jobs, "Score": scores})
    fig = px.bar(df, x="Métier", y="Score", text="Score", range_y=[0, 1])
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(yaxis_title="Score de matching", xaxis_title="Métier")
    return fig


__all__ = ["radar_blocks", "bar_top_jobs"]
