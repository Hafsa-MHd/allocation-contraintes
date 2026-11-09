"""Prévision naive + allocation de production sous capacité (MILP)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pulp

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "demand.csv"
OUT = ROOT / "output"

HOLDING = {"A": 2.0, "B": 1.5, "C": 3.0}
STOCKOUT = {"A": 20.0, "B": 15.0, "C": 25.0}
PROD_COST = {"A": 8.0, "B": 6.0, "C": 10.0}
HOURS = {"A": 1.0, "B": 0.8, "C": 1.4}
DEFAULT_CAPACITY = 150.0
INITIAL_STOCK = {"A": 20.0, "B": 10.0, "C": 5.0}


def load_demand() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    return df.pivot(index="week", columns="product", values="demand").sort_index().astype(float)


def moving_average_forecast(demand: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """À chaque semaine t, prévision = moyenne des `window` semaines réelles précédentes."""
    forecast = demand.copy().astype(float)
    for i, week in enumerate(demand.index):
        if i < window:
            continue
        forecast.loc[week] = demand.iloc[i - window : i].mean()
    return forecast


def greedy_plan(forecast: pd.DataFrame, capacity: float) -> pd.DataFrame:
    """Produit la prévision, bornée par la capacité en heures."""
    rows = []
    stock = INITIAL_STOCK.copy()
    for week, row in forecast.iterrows():
        remaining = capacity
        production = {}
        # priorité : coût de rupture décroissant
        order = list(row.index)  # Excel naïf : A, puis B, puis C — pas de priorité économique
        for p in order:
            need = max(row[p] - stock[p], 0.0)
            max_units = remaining / HOURS[p]
            x = min(need, max_units)
            production[p] = x
            remaining -= x * HOURS[p]
        for p in row.index:
            available = stock[p] + production[p]
            served = min(available, row[p])
            rupture = row[p] - served
            stock[p] = available - served
            rows.append(
                {
                    "week": week,
                    "product": p,
                    "forecast": row[p],
                    "production": production[p],
                    "stock": stock[p],
                    "rupture": rupture,
                    "policy": "greedy",
                }
            )
    return pd.DataFrame(rows)


def milp_plan(forecast: pd.DataFrame, capacity: float) -> pd.DataFrame:
    products = list(forecast.columns)
    weeks = list(forecast.index)
    m = pulp.LpProblem("allocation_capacite", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("prod", (products, weeks), lowBound=0)
    s = pulp.LpVariable.dicts("stock", (products, weeks), lowBound=0)
    r = pulp.LpVariable.dicts("rupture", (products, weeks), lowBound=0)

    m += pulp.lpSum(
        HOLDING[p] * s[p][t] + STOCKOUT[p] * r[p][t] + PROD_COST[p] * x[p][t]
        for p in products
        for t in weeks
    )

    for t in weeks:
        m += pulp.lpSum(HOURS[p] * x[p][t] for p in products) <= capacity, f"cap_{t}"
        for p in products:
            prev = INITIAL_STOCK[p] if t == weeks[0] else s[p][t - 1]
            demand = float(forecast.loc[t, p])
            m += s[p][t] == prev + x[p][t] - demand + r[p][t], f"bilan_{p}_{t}"

    m.solve(pulp.PULP_CBC_CMD(msg=False))
    if m.status != pulp.LpStatusOptimal:
        raise RuntimeError(f"Solveur: {pulp.LpStatus[m.status]}")

    rows = []
    for t in weeks:
        for p in products:
            rows.append(
                {
                    "week": t,
                    "product": p,
                    "forecast": float(forecast.loc[t, p]),
                    "production": x[p][t].value(),
                    "stock": s[p][t].value(),
                    "rupture": r[p][t].value(),
                    "policy": "milp",
                }
            )
    return pd.DataFrame(rows)


def total_cost(plan: pd.DataFrame) -> float:
    c = 0.0
    for _, row in plan.iterrows():
        p = row["product"]
        c += HOLDING[p] * row["stock"]
        c += STOCKOUT[p] * row["rupture"]
        c += PROD_COST[p] * row["production"]
    return c


def decision_cost(plan: pd.DataFrame) -> float:
    """Stock + rupture seulement : c'est ce que le plan décide, pas le coût de fabrication incompressible."""
    c = 0.0
    for _, row in plan.iterrows():
        p = row["product"]
        c += HOLDING[p] * row["stock"]
        c += STOCKOUT[p] * row["rupture"]
    return c


def plot_costs(greedy: pd.DataFrame, milp: pd.DataFrame, path: Path) -> None:
    g = greedy.groupby("week").apply(lambda d: decision_cost(d), include_groups=False)
    o = milp.groupby("week").apply(lambda d: decision_cost(d), include_groups=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(g.index, g.values, marker="o", label="Glouton")
    ax.plot(o.index, o.values, marker="o", label="MILP")
    ax.set_xlabel("Semaine")
    ax.set_ylabel("Coût de décision (stock + rupture)")
    ax.set_title("Coût hebdomadaire : glouton vs allocation optimale")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(capacity: float = DEFAULT_CAPACITY) -> dict:
    OUT.mkdir(exist_ok=True)
    demand = load_demand()
    forecast = moving_average_forecast(demand)
    greedy = greedy_plan(forecast, capacity)
    milp = milp_plan(forecast, capacity)
    cg, co = decision_cost(greedy), decision_cost(milp)
    gain = (cg - co) / cg * 100 if cg else 0.0
    plan = pd.concat([greedy, milp], ignore_index=True)
    plan.to_csv(OUT / "plan.csv", index=False)
    plot_costs(greedy, milp, OUT / "couts.png")
    summary = {
        "capacity": capacity,
        "cout_glouton": round(cg, 1),
        "cout_milp": round(co, 1),
        "gain_pct": round(gain, 1),
        "rupture_glouton": round(float(greedy["rupture"].sum()), 1),
        "rupture_milp": round(float(milp["rupture"].sum()), 1),
        "solver_status": "Optimal",
    }
    print(summary)
    return summary


if __name__ == "__main__":
    run()
