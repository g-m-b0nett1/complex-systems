"""Metastability and nucleation in the probabilistic cellular automaton.

Reproduces every figure and number in docs/pca.md:

1. kappa = 1 (self-interaction): relaxation from d (all minus) to u (all plus).
2. kappa = 1: hitting-time statistics of u and an effective energy barrier.
3. kappa = 0: the intermediate checkerboard phase on the way from d to u.

Usage:  python scripts/pca_nucleation.py [--quick]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from complexsys import pca
from complexsys import plotting as P

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / "figures", ROOT / "results"


def summarise(tau: np.ndarray) -> dict:
    hit = tau[~np.isnan(tau)]
    return {
        "n": int(tau.size),
        "censored": int(np.isnan(tau).sum()),
        "mean": float(hit.mean()) if hit.size else None,
        "std_error": float(hit.std(ddof=1) / np.sqrt(hit.size)) if hit.size > 1 else None,
        # Censored runs count as +inf: the median stays exact while fewer than half are censored.
        "median": float(np.median(np.where(np.isnan(tau), np.inf, tau))) if hit.size > tau.size / 2 else None,
        "values": [None if np.isnan(x) else int(x) for x in tau],
    }


def with_self_interaction(rng, quick: bool) -> dict:
    L, h, kappa, T = (100, 0.4, 1.0, 3000) if quick else (250, 0.4, 1.0, 8000)
    betas = [0.50, 0.55, 0.57]

    # 1. Magnetisation trajectories ------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for beta in betas:
        out = pca.PCA(L, beta, h, kappa).run(T, rng)
        ax.plot(out["M"], label=f"β = {beta}")
    ax.set(xlabel="time step t", ylabel="magnetisation M(t)", ylim=(-1.05, 1.05),
           title=f"Escape from the metastable state d  (κ = 1, h = {h}, L = {L})")
    ax.legend(loc="center right")
    P.save(fig, "pca_kappa1_magnetisation", FIG)

    # 2. Hitting times of u -----------------------------------------------------------
    n_rep = 8 if quick else 30
    max_steps = 6000 if quick else 60000
    grid = [0.48, 0.50, 0.52, 0.54] if quick else [0.48, 0.50, 0.52, 0.54, 0.56, 0.57]
    results = {}
    taus = {}
    for beta in grid:
        tau = pca.hitting_times(pca.PCA(L, beta, h, kappa), n_rep, max_steps, rng, "up", 0.85)
        taus[beta] = tau
        results[f"{beta:.2f}"] = summarise(tau)
        print(f"  kappa=1 beta={beta:.2f}: {results[f'{beta:.2f}']}")

    # Effective barrier: slope of log(median tau) against beta.
    b = np.array([x for x in grid if results[f"{x:.2f}"]["median"]])
    med = np.array([results[f"{x:.2f}"]["median"] for x in b])
    slope, intercept = np.polyfit(b, np.log(med), 1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    t057 = taus[grid[-1]]
    a1.hist(t057[~np.isnan(t057)], bins=12, color=P.BLUE, edgecolor=P.SURFACE, linewidth=1.5)
    a1.set(xlabel="hitting time τ_u (steps)", ylabel="runs",
           title=f"τ_u at β = {grid[-1]}  ({n_rep} runs)")
    a1.grid(axis="x", visible=False)
    a1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Individual hitting times (log scale) and the Arrhenius fit.
    for x in grid:
        y = taus[x][~np.isnan(taus[x])]
        a2.scatter(np.full(y.size, x), y, s=10, color=P.BLUE, alpha=0.35, linewidths=0)
    a2.plot(b, med, "o", color=P.BLUE, markersize=7, label="median")
    bb = np.linspace(b.min(), b.max(), 50)
    a2.plot(bb, np.exp(intercept + slope * bb), "--", color=P.THEORY, lw=1.2,
            label=f"exp(Γ_eff β),  Γ_eff ≈ {slope:.0f}")
    a2.set_yscale("log")
    a2.set(xlabel="inverse temperature β", ylabel="τ_u (steps)", title="Arrhenius scaling")
    a2.legend(loc="upper left")
    fig.tight_layout()
    P.save(fig, "pca_kappa1_hitting_times", FIG)

    return {"L": L, "h": h, "kappa": kappa, "threshold": 0.85, "max_steps": max_steps,
            "hitting_times": results, "gamma_eff": float(slope)}


def without_self_interaction(rng, quick: bool) -> dict:
    L, h, kappa, T = (120, 0.2, 0.0, 6000) if quick else (350, 0.2, 0.0, 8000)
    betas = [0.50, 0.55, 0.70]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3), sharey=True)
    snaps = None
    for ax, beta in zip(axes, betas):
        out = pca.PCA(L, beta, h, kappa).run(T, rng, snapshot_every=50 if beta == 0.70 else None)
        ax.plot(out["M"], color=P.BLUE, label="magnetisation M")
        # |M_stag|: the checkerboard is a stable *pair*, so M_stag flips sign every step.
        ax.plot(np.abs(out["M_stag"]), color=P.ORANGE, label="|staggered magnetisation|")
        ax.plot(out["C"], color=P.AQUA, ls="--", label="checkerboard fraction (both parities)")
        ax.set(title=f"β = {beta}", xlabel="time step t", ylim=(-1.05, 1.05))
        if beta == 0.70:
            snaps, C_traj, M_traj = out["snapshots"], out["C"], out["M"]
    axes[0].set_ylabel("order parameter")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Without self-interaction (κ = 0, h = {h}, L = {L})", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    P.save(fig, "pca_kappa0_order_parameters", FIG)

    # Snapshots spanning the transition: from the first checkerboard patches
    # (C >= 0.05) to the near-complete takeover by u (M >= 0.95).
    t0 = int(np.argmax(C_traj >= 0.05))
    t1 = int(np.argmax(M_traj >= 0.95)) if (M_traj >= 0.95).any() else T
    times = sorted({int(round(t / 50) * 50) for t in np.linspace(t0, t1, 6)})
    # Show local structure on a window, otherwise the checkerboard renders as moiré.
    w = min(L, 120)
    fig, axes = plt.subplots(1, len(times), figsize=(2.0 * len(times), 2.3))
    for ax, t in zip(axes, times):
        ax.imshow(snaps[t][:w, :w], cmap=P.SPIN_CMAP, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"t = {t}", fontsize=9.5, fontweight="normal")
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(f"β = 0.7: d → checkerboard domains → u   ({w}×{w} window; blue = +1)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    P.save(fig, "pca_kappa0_snapshots", FIG)

    # Thirty independent runs at beta = 0.7, evolved together as one batch.
    n_rep = 8 if quick else 30
    m = pca.PCA(L, 0.70, h, kappa)
    run = m.run(T, rng, spins=m.initial_state(n_rep))
    C, M = run["C"], run["M"]  # shape (T+1, n_rep)
    peak = C.max(axis=0)
    t_peak = C.argmax(axis=0)
    reached = M >= 0.85
    tau_u = np.where(reached.any(axis=0), reached.argmax(axis=0), np.nan)
    out = {"L": L, "h": h, "kappa": kappa, "beta": 0.70, "steps": T, "runs": n_rep,
           "peak_checkerboard_fraction": {"min": float(peak.min()), "median": float(np.median(peak)),
                                          "max": float(peak.max()), "values": np.round(peak, 3).tolist()},
           "time_of_peak": summarise(t_peak.astype(float)),
           "runs_with_peak_above_half": int((peak >= 0.5).sum()),
           "up_085": summarise(tau_u)}
    print("  kappa=0:", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small lattices, for a fast smoke test")
    ap.add_argument("--seed", type=int, default=2023)
    args = ap.parse_args()
    P.setup()
    rng = np.random.default_rng(args.seed)
    res = {"seed": args.seed, "kappa1": with_self_interaction(rng, args.quick),
           "kappa0": without_self_interaction(rng, args.quick)}
    RES.mkdir(exist_ok=True)
    (RES / "pca.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
