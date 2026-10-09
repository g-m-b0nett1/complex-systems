"""Statistics of real and random networks.

Everything works on scipy sparse adjacency matrices, so networks with millions
of nodes (e.g. the Pennsylvania road network, ~1.1M nodes) fit in memory.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from math import comb, exp, factorial
from pathlib import Path

import numpy as np
from scipy import optimize, sparse, special, stats

__all__ = [
    "load_edge_list",
    "from_edges",
    "degrees",
    "local_clustering",
    "average_clustering",
    "degree_pmf",
    "friendship_paradox",
    "poisson_gof",
    "PowerLawFit",
    "fit_power_law",
    "power_law_gof",
    "hill_estimator",
    "ws_degree_pmf",
    "ws_clustering",
]


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def from_edges(edges: np.ndarray, n_nodes: int | None = None) -> sparse.csr_matrix:
    """Symmetric, binary adjacency matrix without self-loops.

    Node labels are taken as given; use `load_edge_list` to relabel arbitrary
    identifiers to 0..N-1.
    """
    edges = np.asarray(edges, dtype=np.int64)
    edges = edges[edges[:, 0] != edges[:, 1]]
    n = int(edges.max()) + 1 if n_nodes is None else n_nodes
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    A = sparse.csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n, n))
    A.data[:] = 1  # collapse duplicate edges (an edge listed in both directions)
    A.eliminate_zeros()
    return A


def load_edge_list(path: str | Path) -> sparse.csr_matrix:
    """Read a SNAP-style edge list (plain or .gz, '#' comments) as an undirected graph.

    Node identifiers are relabelled to 0..N-1, so only nodes that actually
    appear in the file are counted. (Using max(id)+1 instead would add
    phantom isolated nodes and bias every per-node average downwards.)
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        raw = np.loadtxt(f, comments="#", dtype=np.int64)
    ids, inverse = np.unique(raw, return_inverse=True)
    return from_edges(inverse.reshape(raw.shape), n_nodes=ids.size)


# --------------------------------------------------------------------------- #
# Basic statistics
# --------------------------------------------------------------------------- #
def degrees(A: sparse.spmatrix) -> np.ndarray:
    return np.asarray(A.sum(axis=1)).ravel().astype(np.int64)


def local_clustering(A: sparse.spmatrix) -> np.ndarray:
    """C_i = (A^3)_ii / (k_i (k_i - 1)); zero for k_i < 2.

    (A^3)_ii = sum_j (A^2)_ij A_ji, so only the entries of A^2 on existing
    edges are needed: no dense matrix and no full A^3.
    """
    A = sparse.csr_matrix(A, dtype=np.int64)
    closed_walks = np.asarray((A @ A).multiply(A).sum(axis=1)).ravel()  # = 2 * triangles
    k = degrees(A)
    denom = k * (k - 1)
    C = np.zeros_like(closed_walks, dtype=float)
    mask = denom > 0
    C[mask] = closed_walks[mask] / denom[mask]
    return C


def average_clustering(A: sparse.spmatrix) -> float:
    return float(local_clustering(A).mean())


def degree_pmf(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical p_m = P(K = m) for m = 0..max(k)."""
    counts = np.bincount(k)
    return np.arange(counts.size), counts / counts.sum()


@dataclass
class FriendshipParadox:
    mean_degree: float
    mean_neighbour_degree: float  # E[k_{v1}] for v1 = random neighbour of a random node
    size_biased_mean: float  # <k^2>/<k>: degree at the end of a uniformly random edge
    frac_neighbours_above_self: float  # fraction of nodes whose neighbours average more than them
    frac_neighbours_above_mean: float  # ... more than the network mean


def friendship_paradox(A: sparse.spmatrix) -> FriendshipParadox:
    """Exact, whole-network version of the friendship paradox (no sampling)."""
    k = degrees(A).astype(float)
    has_nb = k > 0
    nb_mean = np.zeros_like(k)
    nb_mean[has_nb] = (A @ k)[has_nb] / k[has_nb]
    return FriendshipParadox(
        mean_degree=k.mean(),
        mean_neighbour_degree=nb_mean[has_nb].mean(),
        size_biased_mean=(k**2).mean() / k.mean(),
        frac_neighbours_above_self=float((nb_mean[has_nb] > k[has_nb]).mean()),
        frac_neighbours_above_mean=float((nb_mean[has_nb] > k.mean()).mean()),
    )


# --------------------------------------------------------------------------- #
# Poisson hypothesis
# --------------------------------------------------------------------------- #
@dataclass
class GofResult:
    lam: float
    statistic: float
    dof: int
    p_value: float
    n_bins: int


def poisson_gof(k: np.ndarray, min_expected: float = 5.0) -> GofResult:
    """Pearson chi-square test of 'degrees are Poisson(lambda)' with lambda = MLE = mean.

    Neighbouring degree classes are pooled until every bin has expected count
    >= min_expected; without pooling the chi-square approximation is invalid
    and the p-value meaningless. Degrees of freedom: bins - 1 - 1 (one fitted
    parameter).
    """
    k = np.asarray(k)
    n = k.size
    lam = float(k.mean())
    m_max = int(k.max())
    observed = np.bincount(k, minlength=m_max + 1).astype(float)
    expected = n * stats.poisson.pmf(np.arange(m_max + 1), lam)
    expected[-1] += n * stats.poisson.sf(m_max, lam)  # upper tail into the last class

    obs_bins, exp_bins = [], []
    o_acc = e_acc = 0.0
    for o, e in zip(observed, expected):
        o_acc += o
        e_acc += e
        if e_acc >= min_expected:
            obs_bins.append(o_acc)
            exp_bins.append(e_acc)
            o_acc = e_acc = 0.0
    if e_acc > 0 or o_acc > 0:  # leftover tail joins the last bin
        if obs_bins:
            obs_bins[-1] += o_acc
            exp_bins[-1] += e_acc
        else:
            obs_bins.append(o_acc)
            exp_bins.append(e_acc)
    obs_bins, exp_bins = np.array(obs_bins), np.array(exp_bins)
    stat = float(((obs_bins - exp_bins) ** 2 / exp_bins).sum())
    dof = obs_bins.size - 2
    p = float(stats.chi2.sf(stat, dof)) if dof > 0 else float("nan")
    return GofResult(lam=lam, statistic=stat, dof=dof, p_value=p, n_bins=obs_bins.size)


# --------------------------------------------------------------------------- #
# Power-law hypothesis (Clauset, Shalizi & Newman 2009, discrete case)
# --------------------------------------------------------------------------- #
@dataclass
class PowerLawFit:
    alpha: float  # p(k) ~ k^{-alpha} for k >= k_min
    k_min: int
    n_tail: int
    ks: float  # Kolmogorov-Smirnov distance on the tail


def _discrete_alpha_mle(tail: np.ndarray, k_min: int) -> float:
    log_sum = np.log(tail).sum()
    n = tail.size

    def nll(a: float) -> float:
        return n * np.log(special.zeta(a, k_min)) + a * log_sum

    res = optimize.minimize_scalar(nll, bounds=(1.01, 8.0), method="bounded")
    return float(res.x)


def _ks_discrete(tail: np.ndarray, alpha: float, k_min: int) -> float:
    k_max = int(tail.max())
    support = np.arange(k_min, k_max + 1)
    pmf = support.astype(float) ** (-alpha) / special.zeta(alpha, k_min)
    model_cdf = np.cumsum(pmf)
    emp_cdf = np.searchsorted(np.sort(tail), support, side="right") / tail.size
    return float(np.abs(emp_cdf - model_cdf).max())


def fit_power_law(k: np.ndarray, min_tail: int = 50) -> PowerLawFit:
    """Discrete power-law MLE with k_min chosen by minimising the KS distance."""
    k = np.asarray(k)
    k = k[k > 0]
    best: PowerLawFit | None = None
    for k_min in np.unique(k):
        tail = k[k >= k_min]
        if tail.size < min_tail:
            break
        alpha = _discrete_alpha_mle(tail, int(k_min))
        d = _ks_discrete(tail, alpha, int(k_min))
        if best is None or d < best.ks:
            best = PowerLawFit(alpha=alpha, k_min=int(k_min), n_tail=int(tail.size), ks=d)
    if best is None:
        raise ValueError("Not enough data above any k_min to fit a power law.")
    return best


def _sample_discrete_power_law(alpha: float, k_min: int, size: int, rng: np.random.Generator) -> np.ndarray:
    # Inverse-CDF on a truncated support large enough to hold essentially all mass.
    k_max = max(k_min * 10_000, 10_000)
    support = np.arange(k_min, k_max + 1)
    pmf = support.astype(float) ** (-alpha)
    cdf = np.cumsum(pmf / pmf.sum())
    return support[np.searchsorted(cdf, rng.random(size))]


def power_law_gof(
    k: np.ndarray, fit: PowerLawFit, n_boot: int = 200, rng: np.random.Generator | None = None, min_tail: int = 50
) -> float:
    """Semi-parametric bootstrap p-value of Clauset et al. (2009).

    Synthetic data sets reproduce the body (k < k_min) by resampling and the
    tail from the fitted power law; each is refitted from scratch (including
    k_min). p = fraction of synthetic KS distances >= the observed one.
    p < 0.1 rules the power law out.
    """
    rng = rng or np.random.default_rng()
    k = np.asarray(k)
    k = k[k > 0]
    body = k[k < fit.k_min]
    p_tail = fit.n_tail / k.size
    exceed = 0
    for _ in range(n_boot):
        n_t = rng.binomial(k.size, p_tail)
        synth = np.concatenate(
            [
                rng.choice(body, k.size - n_t) if body.size else np.empty(0, dtype=int),
                _sample_discrete_power_law(fit.alpha, fit.k_min, n_t, rng),
            ]
        )
        try:
            if fit_power_law(synth, min_tail=min_tail).ks >= fit.ks:
                exceed += 1
        except ValueError:
            continue
    return exceed / n_boot


def hill_estimator(k: np.ndarray, n_order: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Hill estimate of the tail exponent as a function of the number of order statistics.

    For P(K > k) ~ k^{-(alpha-1)}, xi_hat(n) = (1/n) sum_{i<=n} log(X_(i)/X_(n+1))
    estimates 1/(alpha-1), so alpha_hat(n) = 1 + 1/xi_hat(n). Returns (n, alpha_hat).
    """
    x = np.sort(np.asarray(k, dtype=float)[np.asarray(k) > 0])[::-1]
    logs = np.log(x)
    if n_order is None:
        n_order = np.arange(10, x.size - 1)
    cums = np.cumsum(logs)
    xi = cums[n_order - 1] / n_order - logs[n_order]
    with np.errstate(divide="ignore"):
        alpha = 1.0 + 1.0 / xi
    return n_order, alpha


# --------------------------------------------------------------------------- #
# Watts-Strogatz theory
# --------------------------------------------------------------------------- #
def ws_degree_pmf(m: int, r: int, p: float) -> float:
    """Approximate P(K = m) in WS(N, 2r, p) for large N.

    K = r + A + B with A ~ Bin(r, 1-p) (own clockwise edges kept) and
    B ~ Poisson(r p) (edges rewired onto the node):

        p_m = sum_{n=0}^{min(m-r, r)} C(r,n) (1-p)^n p^{r-n} (rp)^{m-r-n} e^{-rp} / (m-r-n)!

    The upper limit of the sum is *inclusive*.
    """
    if m < r:
        return 0.0
    total = 0.0
    for n in range(0, min(m - r, r) + 1):
        total += (
            comb(r, n)
            * (1 - p) ** n
            * p ** (r - n)
            * (r * p) ** (m - r - n)
            * exp(-r * p)
            / factorial(m - r - n)
        )
    return total


def ws_clustering(r: int, p: float) -> float:
    """C(p) ~ C(0) (1-p)^3 with C(0) = (3r - 3) / (4r - 2) (Barrat & Weigt 2000)."""
    return (3 * r - 3) / (4 * r - 2) * (1 - p) ** 3
