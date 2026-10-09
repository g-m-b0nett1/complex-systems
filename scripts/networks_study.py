"""Real networks and the Watts-Strogatz model.

Reproduces every figure and number in docs/networks.md.

Data (from the Stanford Network Analysis Project, https://snap.stanford.edu/data/):
    data/facebook_combined.txt[.gz]   ego-Facebook
    data/roadNet-PA.txt[.gz]          Pennsylvania road network
Run `python scripts/fetch_data.py` to download them.

Usage:  python scripts/networks_study.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import special, stats

from complexsys import networks as nw
from complexsys import plotting as P

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, RES = ROOT / "data", ROOT / "figures", ROOT / "results"
DATASETS = {"Facebook (ego networks)": "facebook_combined.txt", "Pennsylvania roads": "roadNet-PA.txt"}


def find(name: str) -> Path:
    for candidate in (DATA / name, DATA / f"{name}.gz"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name} not found in data/; run scripts/fetch_data.py")


def real_networks(rng) -> dict:
    out, degs = {}, {}
    for label, fname in DATASETS.items():
        A = nw.load_edge_list(find(fname))
        k = nw.degrees(A)
        degs[label] = k
        fp = nw.friendship_paradox(A)
        gof = nw.poisson_gof(k)
        pl = nw.fit_power_law(k)
        p_pl = nw.power_law_gof(k, pl, n_boot=500, rng=rng)
        out[label] = {
            "nodes": int(A.shape[0]),
            "edges": int(A.nnz // 2),
            "mean_degree": float(k.mean()),
            "max_degree": int(k.max()),
            "average_clustering": nw.average_clustering(A),
            "friendship_paradox": {key: float(v) for key, v in vars(fp).items()},
            "poisson_chi2": vars(gof),
            "power_law": {**vars(pl), "bootstrap_p": p_pl},
        }
        print(label, json.dumps(out[label], indent=1))

    # Degree distributions with the two candidate models.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    label = "Facebook (ego networks)"
    k = degs[label]
    m, pm = nw.degree_pmf(k)
    nz = pm > 0
    a1.loglog(m[nz], pm[nz], "o", ms=3.5, color=P.BLUE, alpha=0.75, label="data")
    lam = k.mean()
    mm = np.arange(1, m.max() + 1)
    pois = stats.poisson.pmf(mm, lam)
    a1.loglog(mm[pois > 1e-6], pois[pois > 1e-6], "--", color=P.THEORY, lw=1.2, label=f"Poisson, λ = {lam:.1f}")
    fit = out[label]["power_law"]
    kk = np.arange(fit["k_min"], m.max() + 1)
    p_tail = fit["n_tail"] / k.size
    a1.loglog(kk, p_tail * kk.astype(float) ** -fit["alpha"] / special.zeta(fit["alpha"], fit["k_min"]),
              color=P.ORANGE, lw=1.8, label=f"power law, α = {fit['alpha']:.2f} (k ≥ {fit['k_min']})")
    a1.set(xlabel="degree m", ylabel="p_m = P(K = m)", ylim=(1e-5, 0.1), title=label)
    a1.legend(loc="lower left", fontsize=8.5)

    label = "Pennsylvania roads"
    k = degs[label]
    m, pm = nw.degree_pmf(k)
    a2.bar(m, pm, width=0.7, color=P.BLUE, label="data")
    a2.plot(m, stats.poisson.pmf(m, k.mean()), "o--", color=P.THEORY, lw=1.0, ms=4,
            label=f"Poisson, λ = {k.mean():.2f}")
    a2.set(xlabel="degree m", ylabel="p_m", title=label, xticks=m)
    a2.grid(axis="x", visible=False)
    a2.legend()
    fig.tight_layout()
    P.save(fig, "networks_degree_distributions", FIG)

    # Hill plot for the Facebook tail.
    k = degs["Facebook (ego networks)"]
    n, a = nw.hill_estimator(k, np.arange(10, 2000))
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.plot(n, a, color=P.BLUE, lw=1.4)
    ax.axhline(out["Facebook (ego networks)"]["power_law"]["alpha"], color=P.ORANGE, lw=1.2, ls="--",
               label="discrete MLE (Clauset et al.)")
    ax.set(xscale="log", xlabel="number of upper order statistics used", ylabel="Hill estimate of α",
           ylim=(1.5, 9), title="Facebook: the tail exponent never stabilises")
    ax.legend(loc="upper left")
    P.save(fig, "networks_hill_plot", FIG)
    return out


def watts_strogatz(rng) -> dict:
    N, r = 500, 5
    ps = [0.2, 0.4, 1.0]
    n_graphs = 50  # pool several realisations so the empirical pmf is not dominated by noise
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3), sharey=True)
    errs = {}
    for ax, p in zip(axes, ps):
        k = np.concatenate([
            np.array([d for _, d in nx.watts_strogatz_graph(N, 2 * r, p, seed=int(rng.integers(1 << 31))).degree()])
            for _ in range(n_graphs)
        ])
        m = np.arange(0, 26)
        emp = np.bincount(k, minlength=m.size)[: m.size] / k.size
        theory = np.array([nw.ws_degree_pmf(int(x), r, p) for x in m])
        errs[p] = float(np.abs(emp - theory).max())
        ax.bar(m, emp, width=0.7, color=P.BLUE, label=f"simulation ({n_graphs} graphs)")
        ax.plot(m, theory, "o-", color=P.THEORY, lw=1.0, ms=3.5, label="theory")
        if p == 1.0:
            ax.plot(m, stats.poisson.pmf(m, 2 * r), "s:", color=P.ORANGE, ms=3.5, lw=1.0,
                    label="Erdős–Rényi, same ⟨k⟩")
        ax.set(title=f"p = {p}", xlabel="degree m")
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("p_m")
    fig.suptitle(f"Watts–Strogatz degree distribution, N = {N}, 2r = {2 * r}", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    P.save(fig, "ws_degree_distributions", FIG)

    # Small-world regime: clustering survives while path length collapses.
    N = 1000
    grid = np.logspace(-4, 0, 13)
    C, L = [], []
    C0 = nx.average_clustering(nx.watts_strogatz_graph(N, 2 * r, 0.0))
    L0 = nx.average_shortest_path_length(nx.watts_strogatz_graph(N, 2 * r, 0.0))
    for p in grid:
        cs, ls = [], []
        for _ in range(5):
            G = nx.connected_watts_strogatz_graph(N, 2 * r, p, seed=int(rng.integers(1 << 31)))
            cs.append(nx.average_clustering(G))
            ls.append(nx.average_shortest_path_length(G))
        C.append(np.mean(cs) / C0)
        L.append(np.mean(ls) / L0)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.semilogx(grid, C, "o", color=P.BLUE, label="clustering C(p)/C(0)")
    ax.semilogx(grid, L, "s", color=P.ORANGE, label="mean path length ℓ(p)/ℓ(0)")
    pp = np.logspace(-4, 0, 200)
    ax.semilogx(pp, (1 - pp) ** 3, "--", color=P.THEORY, lw=1.1, label="(1 − p)³")
    ax.set(xlabel="rewiring probability p", ylabel="relative to p = 0", ylim=(-0.03, 1.08),
           title=f"The small-world window (N = {N}, 2r = {2 * r})")
    ax.legend(loc="lower left")
    P.save(fig, "ws_small_world", FIG)
    return {"degree_pmf_max_abs_error": errs, "C0": C0, "L0": L0,
            "p": grid.tolist(), "C_rel": C, "L_rel": L}


def main() -> None:
    P.setup()
    rng = np.random.default_rng(2023)
    res = {"real": real_networks(rng), "watts_strogatz": watts_strogatz(rng)}
    RES.mkdir(exist_ok=True)
    (RES / "networks.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
