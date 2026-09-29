"""Midsemester read-only results explorer. Run from the x-cpdp project root."""
from pathlib import Path
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="X-CPDP | Research preview",
                   page_icon="🔎", layout="wide")
ROOT = Path(__file__).resolve().parents[1]
METHODS = {"logistic": "No adaptation",
           "coral_mean_logistic": "CORAL + mean", "linear_tca_logistic": "Linear TCA"}
PROJECTS = {"ant": "Ant 1.7", "camel": "Camel 1.6",
            "ivy": "Ivy 2.0", "jedit": "jEdit 4.2"}
st.markdown('''<style>
.block-container {
    padding-top: 2rem;
    max-width: 1400px;
}
[data-testid="stMetric"] {
    border: 1px solid #80808055;
    border-radius: 14px;
    padding: 18px;
}
h1,h2,h3 {
    letter-spacing: -.025em;
}
.hero {
    background: #152b4c;
    border-radius: 18px;
    padding: 28px 32px;
    color: white;
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
    margin: 4px 0 10px;
    font-size: 36px;
}
.hero p {
    color: #c9d8eb;
    margin: 0;
}
.eyebrow {
    color: #85d8d0;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 700;
}
</style>''', unsafe_allow_html=True)
st.markdown('''<div class="hero"><span class="eyebrow">X-CPDP · MIDSEMESTER PROTOTYPE</span>
<h1>Where should we inspect first?</h1><p>Explore cross-project defect predictions and compare model performance.</p></div>''', unsafe_allow_html=True)

runs = {p.name: p for p in (ROOT / "reports" / "reliability").glob("*")
        if p.is_dir() and (p / "outer_results.csv").exists() and (p / "instance_results.csv").exists()}
if not runs:
    st.error("Results not found. Put the dashboard folder inside your x-cpdp folder, beside reports and src. Run the full reliability experiment first.")
    st.stop()
options = sorted(runs, key=lambda n: (n != "laptop-full", n != "full", n))
with st.sidebar:
    st.title("X-CPDP")
    st.caption("Software quality research")
    run = st.selectbox("Saved experiment", options)
    st.divider()
    st.markdown(
        "**Working in this preview**\n\nProject overview · model comparison · class ranking · CSV export")
    st.markdown(
        "**Planned next**\n\nInteractive explanations · new metrics upload · inference workflow")
    st.caption(
        "Read-only results explorer. Selecting a project does not retrain a model.")


@st.cache_data
def load_results(path, outer_stamp, instance_stamp):
    folder = Path(path)
    return pd.read_csv(folder / "outer_results.csv"), pd.read_csv(folder / "instance_results.csv")


folder = runs[run]
try:
    results, instances = load_results(str(folder), (folder / "outer_results.csv").stat(
    ).st_mtime_ns, (folder / "instance_results.csv").stat().st_mtime_ns)
    assert {"project", "method", "average_precision",
            "roc_auc", "training_jaccard_5"} <= set(results)
    assert {"project", "method", "class_name",
            "risk_score", "defective", "loc"} <= set(instances)
    for project_name, group in results.groupby("project"):
        reference = None
        for method_name in group.method:
            subset = instances.loc[instances.project.eq(
                project_name) & instances.method.eq(method_name)]
            if subset.empty or subset.class_name.duplicated().any():
                raise ValueError(
                    f"Missing or duplicate predictions for {project_name}/{method_name}.")
            identities = subset.set_index("class_name")[
                "defective"].sort_index()
            if reference is not None and not identities.equals(reference):
                raise ValueError(
                    f"Incomplete or inconsistent class predictions for {project_name}. Use a completed full run.")
            reference = identities
except (AssertionError, ValueError, OSError) as error:
    st.error(f"Could not read compatible experiment reports: {error}")
    st.stop()

left, right = st.columns(2)
available = [p for p in PROJECTS if p in results.project.unique()]
project = left.selectbox("Held-out project", available,
                         format_func=PROJECTS.get)
methods = [m for m in METHODS if m in results.loc[results.project.eq(
    project), "method"].unique()]
method = right.selectbox("Model for class ranking",
                         methods, format_func=METHODS.get)
rows = instances.loc[instances.project.eq(
    project) & instances.method.eq(method)].copy()
comparison = results.loc[results.project.eq(
    project) & results.method.isin(METHODS)].copy()
if rows.empty:
    st.error("No class predictions are available for this selection.")
    st.stop()
selected = comparison.loc[comparison.method.eq(method)].iloc[0]
cols = st.columns(4)
cols[0].metric("Java classes", f"{len(rows):,}")
cols[1].metric("Recorded defective", f"{int(rows.defective.sum()):,}",
               help="Historical benchmark labels; these are not model predictions.")
cols[2].metric("Defective share", f"{rows.defective.mean():.1%}")
cols[3].metric("Model ROC-AUC", f"{selected.roc_auc:.3f}",
               help="Ranking quality: 0.5 is chance and 1.0 is perfect.")
st.caption("Each model was trained on other projects. Adaptation methods also used unlabelled target metrics. Target defect labels are used for evaluation.")
st.subheader("01 · Compare the models")
metric = st.radio("Comparison measure", [
                  "Average precision", "Explanation stability"], horizontal=True)
column = "average_precision" if metric == "Average precision" else "training_jaccard_5"
comparison["Model"] = comparison.method.map(METHODS)
chart = alt.Chart(comparison).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
    x=alt.X("Model:N", sort=list(METHODS.values()),
            axis=alt.Axis(labelAngle=0), title=None),
    y=alt.Y(f"{column}:Q", scale=alt.Scale(domain=[0, 1]), title=metric),
    color=alt.Color("Model:N", scale=alt.Scale(domain=list(
        METHODS.values()), range=["#3868ed", "#12a89d", "#e4a13b"]), legend=None),
    tooltip=["Model:N", alt.Tooltip(f"{column}:Q", format=".4f")]).properties(height=250)
st.altair_chart(chart, width="stretch")
if column == "average_precision":
    st.caption("Higher is better: average precision measures how well defective classes appear near the top of the ranking. It is not accuracy.")
else:
    st.caption("Higher is more consistent: top-5 Jaccard compares sets of important explanation metrics across retraining runs. Stability does not establish correctness.")
st.subheader("02 · Classes to inspect first")
st.caption("Risk scores rank inspection priority. They are not calibrated probabilities or proof that a class contains a defect.")
a, b = st.columns([3, 1])
query = a.text_input(
    "Find a class", placeholder="Type part of a class name, e.g. Task")
limit = b.selectbox("Rows to display", [10, 25, 50, 100])
ranked = rows.sort_values(["risk_score", "class_name"], ascending=[
                          False, True]).reset_index(drop=True)
ranked.insert(0, "Rank", ranked.index + 1)
filtered = ranked.loc[ranked.class_name.str.contains(
    query, case=False, regex=False, na=False)]
display = filtered[["Rank", "class_name", "risk_score", "loc", "defective"]].rename(
    columns={"class_name": "Class", "risk_score": "Risk score", "loc": "Lines of code", "defective": "Recorded defect"})
st.dataframe(display.head(limit), hide_index=True, width="stretch",
             column_config={"Risk score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.4f"), "Class": st.column_config.TextColumn(width="large")})
st.caption(f"Showing {min(limit, len(display))} of {len(display)} matching classes. Recorded defect: 1 = defective, 0 = no defect recorded. Rank stays relative to the whole project.")
st.download_button("Download matching rankings", display.to_csv(index=False).encode(
    "utf-8"), file_name=f"{project}_{method}_rankings.csv", mime="text/csv")
st.divider()
st.caption(
    f"Midsemester preview · Saved experiment: {run} · 20 structural metrics · Class-level Java benchmarks")
