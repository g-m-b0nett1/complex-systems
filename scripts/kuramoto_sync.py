"""Spontaneous synchronisation in the mean-field Kuramoto model.

Reproduces every figure and number in docs/kuramoto.md.

Usage:  python scripts/kuramoto_sync.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from complexsys import kuramoto as ku
from complexsys import plotting as P

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / "figures", ROOT / "results"
DT = 0.05  # RK4 step; halving it changes r_inf by < 1e-4 (tests/test_kuramoto.py)


def stationary_r(r: np.ndarray, frac: float = 0.25) -> np.ndarray:
    """Time average over the final `frac` of the run (less noisy than r(T))."""
    n = r.shape[-1]
    return r[..., int(n * (1 - frac)):].mean(axis=-1)


def gaussian(rng) -> dict:
    Kc = 2 * np.sqrt(2 / np.pi)
    K = np.round(np.arange(0.2, 5.01, 0.2), 2)
    K_th = np.linspace(0, 5, 400)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    out = {"Kc": Kc}
    for N, colour, marker in ((1000, P.BLUE, "o"), (20000, P.ORANGE, "s")):
        theta0 = rng.uniform(-np.pi, np.pi, N)
        omega = rng.normal(0, 1, N)
        _, r, _ = ku.simulate(theta0, omega, K, t_max=100, dt=DT)
        r_inf = stationary_r(r)
        out[f"r_inf_N{N}"] = dict(zip(K.tolist(), r_inf.tolist()))
        a1.plot(K, r_inf, marker, color=colour, label=f"simulation, N = {N}")
    a1.plot(K_th, ku.r_stationary_gaussian(K_th), "--", color=P.THEORY, lw=1.2, label="self-consistency (N → ∞)")
    a1.axvline(Kc, color=P.MUTED, lw=0.9, ls=":")
    a1.annotate(f"K_c = 2√(2/π) ≈ {Kc:.3f}", (Kc, 0.45), xytext=(-6, 0), textcoords="offset points",
                ha="right", fontsize=8.5, color=P.INK_2)
    a1.set(xlabel="coupling K", ylabel="stationary order parameter r∞",
           title="ω ~ N(0, 1): continuous transition")
    a1.legend(loc="lower right", fontsize=8.5)

    N = 1000
    theta0, omega = rng.uniform(-np.pi, np.pi, N), rng.normal(0, 1, N)
    t, r, _ = ku.simulate(theta0, omega, np.array([1.0, 2.0]), t_max=100, dt=DT)
    for row, k, colour in zip(r, (1.0, 2.0), (P.BLUE, P.ORANGE)):
        a2.plot(t, row, color=colour, label=f"K = {k}" + ("  (< K_c)" if k < Kc else "  (> K_c)"))
        a2.axhline(ku.r_stationary_gaussian(np.array([k]))[0], color=colour, lw=0.9, ls="--")
    a2.set(xlabel="time t", ylabel="r(t)", ylim=(0, 1), title="Relaxation, N = 1000 (dashed: theory)")
    a2.legend(loc="center right")
    fig.tight_layout()
    P.save(fig, "kuramoto_gaussian", FIG)
    out["finite_size_noise_K1"] = float(r[0, -1000:].mean())
    return out


def uniform(rng) -> dict:
    gamma = 0.5
    Kc = 4 * gamma / np.pi
    N = 2000
    K = np.round(np.arange(0.03, 1.501, 0.03), 2)
    theta0 = rng.uniform(-np.pi, np.pi, N)
    omega = rng.uniform(-gamma, gamma, N)
    _, r, _ = ku.simulate(theta0, omega, K, t_max=200, dt=DT)
    r_inf = stationary_r(r)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    a1.plot(K, r_inf, "o", color=P.BLUE, label=f"simulation, N = {N}")
    K_lo = np.linspace(0, Kc * 0.9999, 50)
    K_hi = np.linspace(Kc, 1.5, 300)
    a1.plot(K_lo, np.zeros_like(K_lo), "--", color=P.THEORY, lw=1.2, label="theory (N → ∞)")
    a1.plot(K_hi, ku.r_stationary_uniform(K_hi, gamma), "--", color=P.THEORY, lw=1.2)
    a1.plot([Kc, Kc], [0, np.pi / 4], ":", color=P.THEORY, lw=1.0)
    a1.annotate("jump to π/4 at K_c = 2/π", (Kc, np.pi / 4), xytext=(8, -4), textcoords="offset points",
                fontsize=8.5, color=P.INK_2)
    a1.set(xlabel="coupling K", ylabel="r∞", title="ω ~ U[−½, ½]: discontinuous transition")
    a1.legend(loc="lower right", fontsize=8.5)

    # Relaxation at K = 1: which randomness matters, initial phases or frequencies?
    K1, n_runs, n_plot, thr = 1.0, 50, 10, 0.75
    r_target = float(ku.r_stationary_uniform(np.array([K1]), gamma)[0])
    omega_fixed = rng.uniform(-gamma, gamma, N)
    theta_fixed = rng.uniform(-np.pi, np.pi, N)
    thetas = rng.uniform(-np.pi, np.pi, (n_runs, N))
    omegas = rng.uniform(-gamma, gamma, (n_runs, N))
    tau_I, tau_II = [], []
    for i in range(n_runs):
        t, rI, _ = ku.simulate(thetas[i], omega_fixed, K1, t_max=60, dt=DT)
        _, rII, _ = ku.simulate(theta_fixed, omegas[i], K1, t_max=60, dt=DT)
        tau_I.append(ku.relaxation_time(t, rI, thr))
        tau_II.append(ku.relaxation_time(t, rII, thr))
        if i < n_plot:
            a2.plot(t, rI, color=P.BLUE, lw=1.0, alpha=0.8, label="fixed ω, random θ(0)" if i == 0 else None)
            a2.plot(t, rII, color=P.ORANGE, lw=1.0, alpha=0.8, label="fixed θ(0), random ω" if i == 0 else None)
    a2.axhline(thr, color=P.MUTED, lw=0.9, ls=":")
    a2.set(xlabel="time t", ylabel="r(t)", ylim=(0, 1), title=f"K = {K1}: {n_plot} of {n_runs} runs of each kind shown")
    a2.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    P.save(fig, "kuramoto_uniform", FIG)

    def stats_(x):
        x = np.array(x)
        return {"mean": float(np.nanmean(x)), "std": float(np.nanstd(x, ddof=1)),
                "not_reached": int(np.isnan(x).sum())}

    # Is the spread really larger when the frequencies vary? Levene test (robust to non-normality).
    tI, tII = np.array(tau_I), np.array(tau_II)
    lev = stats.levene(tI[~np.isnan(tI)], tII[~np.isnan(tII)])
    welch = stats.ttest_ind(tI[~np.isnan(tI)], tII[~np.isnan(tII)], equal_var=False)
    return {"levene_p": float(lev.pvalue), "welch_p": float(welch.pvalue), "Kc": Kc, "r_inf": dict(zip(K.tolist(), r_inf.tolist())), "r_theory_K1": r_target,
            "relaxation_threshold": thr, "tau_fixed_omega": stats_(tau_I), "tau_fixed_theta": stats_(tau_II)}


def main() -> None:
    P.setup()
    rng = np.random.default_rng(2024)
    res = {"gaussian": gaussian(rng), "uniform": uniform(rng)}
    RES.mkdir(exist_ok=True)
    (RES / "kuramoto.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res["uniform"][k] for k in ("tau_fixed_omega", "tau_fixed_theta", "r_theory_K1", "levene_p", "welch_p")}, indent=1))


if __name__ == "__main__":
    main()
