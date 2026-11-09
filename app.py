"""Démo 2 minutes pour un forum : tu bouges la capacité, le plan se recalcule."""

import pandas as pd
import streamlit as st

from optimize import DEFAULT_CAPACITY, load_demand, run

st.set_page_config(page_title="Allocation sous contraintes", layout="wide")
st.title("Allocation de production sous capacité")
st.caption("Prévision naive + MILP (PuLP). Projet personnel — Hafsa Mehdaoui, ENSEM.")

capacity = st.slider("Capacité hebdomadaire (heures machine)", 80, 280, int(DEFAULT_CAPACITY), 10)

if st.button("Calculer le plan", type="primary") or True:
    summary = run(capacity=float(capacity))
    c1, c2, c3 = st.columns(3)
    c1.metric("Coût décision glouton", f"{summary['cout_glouton']:.0f}")
    c2.metric("Coût décision MILP", f"{summary['cout_milp']:.0f}")
    c3.metric("Gain", f"{summary['gain_pct']:.1f} %")
    st.image("output/couts.png", caption="Coût par semaine")
    plan = pd.read_csv("output/plan.csv")
    st.subheader("Plan MILP")
    st.dataframe(plan[plan["policy"] == "milp"], hide_index=True, use_container_width=True)

st.markdown(
    "Le glouton produit la demande prévue, borné par la capacité. "
    "Le MILP lisse la production dans le temps pour éviter les ruptures coûteuses."
)
