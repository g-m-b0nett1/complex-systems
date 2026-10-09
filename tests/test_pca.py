import numpy as np
import pytest

from complexsys import pca


def test_neighbour_sum_periodic():
    s = np.zeros((4, 4), dtype=np.int8)
    s[0, 0] = 1
    S = pca.neighbour_sum(s)
    # The four neighbours of the corner, wrapping round the torus.
    assert {(0, 1), (1, 0), (0, 3), (3, 0)} == set(zip(*np.nonzero(S)))


def test_detailed_balance_small_lattice():
    """p(s, e) mu(s) = p(e, s) mu(e) for every pair of configurations (L = 3)."""
    beta, h, kappa = 0.7, 0.3, 0.6
    configs = pca.all_configurations(3)
    H = pca.hamiltonian(configs, beta, h, kappa)
    rng = np.random.default_rng(0)
    for _ in range(300):
        a, b = rng.integers(len(configs), size=2)
        s, e = configs[a], configs[b]
        lhs = pca.transition_probability(s, e, beta, h, kappa) * np.exp(-H[a])
        rhs = pca.transition_probability(e, s, beta, h, kappa) * np.exp(-H[b])
        assert lhs == pytest.approx(rhs, rel=1e-10)


@pytest.mark.parametrize("h,kappa", [(0.4, 1.0), (0.2, 0.0), (0.5, 0.5)])
def test_ground_state_is_all_plus_for_positive_field(h, kappa):
    configs = pca.all_configurations(4)
    E = pca.ground_energy(configs, h, kappa)
    best = configs[np.isclose(E, E.min())]
    assert len(best) == 1 and (best[0] == 1).all()


def test_zero_field_no_self_interaction_has_four_ground_states():
    """h = 0, kappa = 0: u, d and the two checkerboards all minimise E."""
    configs = pca.all_configurations(4)
    E = pca.ground_energy(configs, 0.0, 0.0)
    best = configs[np.isclose(E, E.min())]
    assert len(best) == 4
    stag = pca.staggered_magnetisation(best)
    mag = pca.magnetisation(best)
    assert sorted(np.round(mag).tolist()) == [-1, 0, 0, 1]
    assert sorted(np.round(stag).tolist()) == [-1, 0, 0, 1]


def test_step_matches_single_site_probability():
    """Empirical flip frequency from an all-minus state equals the formula."""
    m = pca.PCA(L=64, beta=0.5, h=0.4, kappa=1.0)
    rng = np.random.default_rng(1)
    s = m.initial_state(replicas=50)
    up = (m.step(s, rng) == 1).mean()
    expected = 0.5 * (1 + np.tanh(0.5 * (-1 - 4 + 0.4)))
    assert up == pytest.approx(expected, abs=3e-3)


def test_hitting_times_censored_as_nan():
    m = pca.PCA(L=16, beta=3.0, h=0.1, kappa=1.0)  # cold: no nucleation in 5 steps
    tau = pca.hitting_times(m, replicas=4, max_steps=5, rng=np.random.default_rng(2))
    assert np.isnan(tau).all()


def test_checkerboard_fraction_counts_both_parities():
    L = 8
    i, j = np.indices((L, L))
    even = np.where((i + j) % 2 == 0, 1, -1).astype(np.int8)
    assert pca.checkerboard_fraction(even) == 1.0
    assert pca.checkerboard_fraction(-even) == 1.0
    assert pca.checkerboard_fraction(np.ones((L, L), dtype=np.int8)) == 0.0
