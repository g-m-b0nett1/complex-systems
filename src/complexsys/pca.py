"""Probabilistic cellular automaton (PCA) with parallel Ising-like dynamics.

Each site x of an L x L torus carries a spin sigma(x) in {-1, +1}. At every
time step all spins are updated *simultaneously*, independently of each other,
with

    P[sigma_{t+1}(x) = s | sigma_t] = 1/2 * (1 + s * tanh(beta * (kappa*sigma_t(x) + S_t(x) + h)))

where S_t(x) is the sum of the four nearest neighbours, beta the inverse
temperature, h the external field and kappa >= 0 the self-interaction.

The dynamics is reversible with respect to the Gibbs-like measure
mu(sigma) ~ exp(-H(sigma)) with

    H(sigma) = -beta*h*sum_x sigma(x) - sum_x log cosh(beta*(kappa*sigma(x) + S(x) + h)).

The implementation is fully vectorised and can evolve many independent
replicas at once (leading array axis), which is what makes hitting-time
statistics cheap.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

__all__ = [
    "PCA",
    "neighbour_sum",
    "hamiltonian",
    "ground_energy",
    "transition_probability",
    "magnetisation",
    "staggered_magnetisation",
    "checkerboard_fraction",
    "hitting_times",
]


def neighbour_sum(spins: np.ndarray) -> np.ndarray:
    """Sum of the four nearest neighbours on a torus (last two axes)."""
    return (
        np.roll(spins, 1, axis=-1)
        + np.roll(spins, -1, axis=-1)
        + np.roll(spins, 1, axis=-2)
        + np.roll(spins, -1, axis=-2)
    )


def magnetisation(spins: np.ndarray) -> np.ndarray:
    """Mean spin over the last two axes."""
    return spins.mean(axis=(-2, -1))


def _checkerboard_sign(L: int) -> np.ndarray:
    i, j = np.indices((L, L))
    return np.where((i + j) % 2 == 0, 1, -1).astype(np.int8)


def staggered_magnetisation(spins: np.ndarray) -> np.ndarray:
    """(1/L^2) sum_x (-1)^{x1+x2} sigma(x): equals +1 or -1 on the two checkerboards."""
    L = spins.shape[-1]
    return (spins * _checkerboard_sign(L)).mean(axis=(-2, -1))


def checkerboard_fraction(spins: np.ndarray) -> np.ndarray:
    """Fraction of sites whose four neighbours all have the opposite spin.

    Unlike the staggered magnetisation, this counts checkerboard regions of
    *both* parities: a lattice split into an even and an odd checkerboard
    domain has M_stag ~ 0 but a checkerboard fraction close to 1.
    """
    return (neighbour_sum(spins) == -4 * spins).mean(axis=(-2, -1))


def hamiltonian(spins: np.ndarray, beta: float, h: float, kappa: float) -> np.ndarray:
    """Finite-volume Hamiltonian H_{beta,h} of the reversible measure."""
    field = kappa * spins + neighbour_sum(spins) + h
    return -beta * h * spins.sum(axis=(-2, -1)) - np.log(np.cosh(beta * field)).sum(axis=(-2, -1))


def ground_energy(spins: np.ndarray, h: float, kappa: float) -> np.ndarray:
    """Zero-temperature energy E(sigma) = lim_{beta->inf} H/beta."""
    field = kappa * spins + neighbour_sum(spins) + h
    return -h * spins.sum(axis=(-2, -1)) - np.abs(field).sum(axis=(-2, -1))


def transition_probability(
    sigma: np.ndarray, eta: np.ndarray, beta: float, h: float, kappa: float
) -> float:
    """p(sigma, eta): probability that one parallel step maps sigma to eta."""
    field = kappa * sigma + neighbour_sum(sigma) + h
    p_site = 0.5 * (1.0 + eta * np.tanh(beta * field))
    return float(np.prod(p_site))


@dataclass
class PCA:
    """Parallel stochastic dynamics on an L x L torus.

    Parameters
    ----------
    L : lattice side.
    beta : inverse temperature.
    h : external magnetic field.
    kappa : self-interaction (kappa = 0 removes the site's own spin from its field).
    """

    L: int
    beta: float
    h: float
    kappa: float = 0.0

    def __post_init__(self) -> None:
        # The local field kappa*sigma + S + h takes only 10 values (sigma = +-1,
        # S in {-4,-2,0,2,4}), so P[new spin = +1] is a lookup, not a tanh call.
        s_vals = np.arange(-4, 5, 2)
        table = np.empty((2, 5))
        for a, sigma in enumerate((-1, 1)):
            table[a] = 0.5 * (1.0 + np.tanh(self.beta * (self.kappa * sigma + s_vals + self.h)))
        self._p_up = table

    def initial_state(self, replicas: int | None = None, value: int = -1) -> np.ndarray:
        """Uniform configuration (all spins = value); default is d, all minus."""
        shape = (self.L, self.L) if replicas is None else (replicas, self.L, self.L)
        return np.full(shape, value, dtype=np.int8)

    def step(self, spins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One parallel update of every site (and every replica)."""
        s_idx = (neighbour_sum(spins) + 4) // 2
        a_idx = (spins + 1) // 2
        p_up = self._p_up[a_idx, s_idx]
        u = rng.random(spins.shape)
        return np.where(u < p_up, 1, -1).astype(np.int8)

    def run(
        self,
        steps: int,
        rng: np.random.Generator,
        spins: np.ndarray | None = None,
        snapshot_every: int | None = None,
    ) -> dict:
        """Evolve for `steps` steps and record observables at every step.

        Returns a dict with time series 'M', 'M_stag' and 'C' (checkerboard
        fraction), each of shape (steps+1,) or (steps+1, replicas), the
        final configuration and optional snapshots.
        """
        if spins is None:
            spins = self.initial_state()
        M = [magnetisation(spins)]
        Ms = [staggered_magnetisation(spins)]
        C = [checkerboard_fraction(spins)]
        snapshots = {0: spins.copy()} if snapshot_every else {}
        for t in range(1, steps + 1):
            spins = self.step(spins, rng)
            M.append(magnetisation(spins))
            Ms.append(staggered_magnetisation(spins))
            C.append(checkerboard_fraction(spins))
            if snapshot_every and t % snapshot_every == 0:
                snapshots[t] = spins.copy()
        return {
            "M": np.asarray(M),
            "M_stag": np.asarray(Ms),
            "C": np.asarray(C),
            "final": spins,
            "snapshots": snapshots,
        }


def hitting_times(
    model: PCA,
    replicas: int,
    max_steps: int,
    rng: np.random.Generator,
    target: str = "up",
    threshold: float = 0.85,
) -> np.ndarray:
    """First time each replica, started from d (all minus), reaches the target.

    target = 'up'           : first t with M(t) >= threshold.
    target = 'checkerboard' : first t with checkerboard fraction >= threshold.

    Replicas that never hit within max_steps get np.nan (right-censored); they
    are *not* silently counted as zero.
    """
    spins = model.initial_state(replicas)
    tau = np.full(replicas, np.nan)
    active = np.ones(replicas, dtype=bool)
    for t in range(1, max_steps + 1):
        spins = model.step(spins, rng)
        if target == "up":
            obs = magnetisation(spins)
        elif target == "checkerboard":
            obs = checkerboard_fraction(spins)
        else:
            raise ValueError("target must be 'up' or 'checkerboard'")
        hit = active & (obs >= threshold)
        tau[hit] = t
        active &= ~hit
        if not active.any():
            break
    return tau


def all_configurations(L: int) -> np.ndarray:
    """Every configuration of an L x L lattice (only sensible for L <= 4)."""
    n = L * L
    configs = np.array(list(itertools.product((-1, 1), repeat=n)), dtype=np.int8)
    return configs.reshape(-1, L, L)
