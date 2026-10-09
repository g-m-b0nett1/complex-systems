"""Mean-field Kuramoto model.

    d theta_i / dt = omega_i + (K/N) sum_j sin(theta_j - theta_i)
                   = omega_i + K r sin(psi - theta_i),

with r e^{i psi} = (1/N) sum_j e^{i theta_j}. The mean-field form makes each
right-hand side evaluation O(N) instead of O(N^2). Integration is vectorised
over oscillators *and* over a batch of couplings K, so a whole r(K) sweep is
a single array computation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import integrate, optimize

__all__ = [
    "order_parameter",
    "simulate",
    "relaxation_time",
    "critical_coupling",
    "r_stationary_gaussian",
    "r_stationary_uniform",
]


def order_parameter(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(r, psi) over the last axis."""
    z = np.exp(1j * theta).mean(axis=-1)
    return np.abs(z), np.angle(z)


def _rhs(theta: np.ndarray, omega: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = np.exp(1j * theta).mean(axis=-1, keepdims=True)
    # K r sin(psi - theta) = K Im(z e^{-i theta})
    return omega + K * np.imag(z * np.exp(-1j * theta))


def simulate(
    theta0: np.ndarray,
    omega: np.ndarray,
    K: float | np.ndarray,
    t_max: float,
    dt: float,
    method: str = "rk4",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the model and record r(t) at every step.

    K may be a scalar or a 1-D array; with an array, every coupling is
    integrated in parallel from the same theta0 and omega.

    Returns (t, r, theta_final) with r of shape (n_steps+1,) or (n_K, n_steps+1).
    """
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))[:, None]
    theta = np.broadcast_to(theta0, (K_arr.shape[0], theta0.size)).astype(float).copy()
    n_steps = int(round(t_max / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    r = np.empty((K_arr.shape[0], n_steps + 1))
    r[:, 0] = order_parameter(theta)[0]
    for n in range(1, n_steps + 1):
        if method == "euler":
            theta = theta + dt * _rhs(theta, omega, K_arr)
        elif method == "rk4":
            k1 = _rhs(theta, omega, K_arr)
            k2 = _rhs(theta + 0.5 * dt * k1, omega, K_arr)
            k3 = _rhs(theta + 0.5 * dt * k2, omega, K_arr)
            k4 = _rhs(theta + dt * k3, omega, K_arr)
            theta = theta + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError("method must be 'euler' or 'rk4'")
        r[:, n] = order_parameter(theta)[0]
    if np.ndim(K) == 0:
        return t, r[0], theta[0]
    return t, r, theta


def relaxation_time(t: np.ndarray, r: np.ndarray, threshold: float) -> float:
    """First time r(t) >= threshold; np.nan if it never gets there.

    Returning nan (not 0) matters: a run that never synchronises must not be
    averaged in as an instantaneous one.
    """
    idx = np.flatnonzero(r >= threshold)
    return float(t[idx[0]]) if idx.size else float("nan")


def critical_coupling(g0: float) -> float:
    """Kuramoto's K_c = 2 / (pi g(0)) for a symmetric unimodal g."""
    return 2.0 / (np.pi * g0)


def _self_consistency(K: float, g: Callable[[np.ndarray], np.ndarray]) -> Callable[[float], float]:
    # Stationary partially locked state: 1 = K * int_{-pi/2}^{pi/2} cos^2(x) g(K r sin x) dx
    def f(r: float) -> float:
        val, _ = integrate.quad(lambda x: np.cos(x) ** 2 * g(K * r * np.sin(x)), -np.pi / 2, np.pi / 2)
        return K * val - 1.0

    return f


def r_stationary_gaussian(K: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Stationary r(K) for omega ~ N(0, sigma^2) in the N -> infinity limit.

    Solves the self-consistency equation with a bracketing root-finder on
    (0, 1]; r = 0 below K_c = 2 sqrt(2/pi) sigma, where only the incoherent
    solution exists. (A Newton solver started at r = 1 can return spurious
    negative roots, which then have to be clipped.)
    """
    g = lambda w: np.exp(-0.5 * (w / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    Kc = critical_coupling(g(0.0))
    out = np.zeros(np.size(K))
    for i, k in enumerate(np.atleast_1d(K)):
        if k <= Kc:
            continue
        f = _self_consistency(float(k), g)
        out[i] = optimize.brentq(f, 1e-9, 1.0)
    return out


def r_stationary_uniform(K: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """Stationary r(K) for omega ~ U[-gamma, gamma] (N -> infinity).

    Closed form of the self-consistency integral: with sin(a) = gamma/(K r),
        1 = K/(2 gamma) * (a + sin(a) cos(a)).
    The transition at K_c = 4 gamma / pi is discontinuous: r jumps from 0
    to pi/4 (Pazo 2005), and full locking (r -> 1 asymptotically) follows.
    """
    Kc = 4 * gamma / np.pi
    out = np.zeros(np.size(K))
    for i, k in enumerate(np.atleast_1d(K)):
        if k < Kc:
            continue

        def f(a: float, k: float = float(k)) -> float:
            return k / (2 * gamma) * (a + np.sin(a) * np.cos(a)) - 1.0

        # a = pi/2 at K = K_c; a decreases as K grows.
        a = np.pi / 2 if np.isclose(k, Kc) else optimize.brentq(f, 1e-12, np.pi / 2)
        out[i] = gamma / (k * np.sin(a))
    return out
