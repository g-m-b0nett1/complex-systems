import numpy as np
import pytest

from complexsys import kuramoto as ku


def test_order_parameter_limits():
    assert ku.order_parameter(np.zeros(100))[0] == pytest.approx(1.0)
    assert ku.order_parameter(np.linspace(0, 2 * np.pi, 100, endpoint=False))[0] == pytest.approx(0.0, abs=1e-12)


def test_mean_phase_velocity_is_mean_frequency():
    """d/dt (1/N) sum theta_i = mean(omega): coupling terms cancel pairwise."""
    rng = np.random.default_rng(0)
    theta0 = rng.uniform(-np.pi, np.pi, 200)
    omega = rng.normal(0, 1, 200)
    t, _, theta = ku.simulate(theta0, omega, 1.7, t_max=5.0, dt=0.01)
    assert (theta - theta0).mean() == pytest.approx(omega.mean() * t[-1], rel=1e-8)


def test_identical_oscillators_synchronise():
    rng = np.random.default_rng(1)
    _, r, _ = ku.simulate(rng.uniform(-np.pi, np.pi, 300), np.zeros(300), 1.0, t_max=30, dt=0.05)
    assert r[-1] > 0.999
    assert np.all(np.diff(r) >= -1e-9)  # r is non-decreasing for identical oscillators


def test_critical_couplings():
    g0 = 1 / np.sqrt(2 * np.pi)
    assert ku.critical_coupling(g0) == pytest.approx(2 * np.sqrt(2 / np.pi))
    assert ku.critical_coupling(1.0) == pytest.approx(2 / np.pi)


def test_gaussian_theory_near_threshold():
    """Near K_c, r ~ sqrt(16 (K - K_c) / (pi K_c^4 (-g''(0)))) (Kuramoto)."""
    Kc = 2 * np.sqrt(2 / np.pi)
    K = Kc * 1.001
    gpp = -1 / np.sqrt(2 * np.pi)
    expected = np.sqrt(16 * (K - Kc) / (np.pi * Kc**4 * -gpp))
    assert ku.r_stationary_gaussian(np.array([K]))[0] == pytest.approx(expected, rel=0.02)
    assert ku.r_stationary_gaussian(np.array([1.0]))[0] == 0.0


def test_uniform_jump_to_pi_over_four():
    Kc = 2 / np.pi
    r = ku.r_stationary_uniform(np.array([Kc * 0.999, Kc, Kc * 1.0001, 5.0]))
    assert r[0] == 0.0
    assert r[1] == pytest.approx(np.pi / 4)
    assert r[2] == pytest.approx(np.pi / 4, abs=0.02)
    assert 0.98 < r[3] < 1.0


def test_simulation_matches_gaussian_theory():
    rng = np.random.default_rng(2)
    N = 4000
    # Quantile-spaced frequencies remove sampling noise in g.
    from scipy.stats import norm
    omega = norm.ppf((np.arange(N) + 0.5) / N)
    K = np.array([2.5, 3.5])
    _, r, _ = ku.simulate(rng.uniform(-np.pi, np.pi, N), omega, K, t_max=60, dt=0.05)
    assert r[:, -200:].mean(axis=1) == pytest.approx(ku.r_stationary_gaussian(K), abs=0.02)


def test_relaxation_time_nan_when_never_reached():
    t = np.linspace(0, 1, 11)
    assert np.isnan(ku.relaxation_time(t, np.full(11, 0.2), 0.75))
    assert ku.relaxation_time(t, np.linspace(0, 1, 11), 0.75) == pytest.approx(0.8)


def test_rk4_step_converged():
    """dt = 0.05 is small enough: halving it moves the stationary r by < 1e-4."""
    rng = np.random.default_rng(3)
    N = 500
    theta0, omega = rng.uniform(-np.pi, np.pi, N), rng.uniform(-0.5, 0.5, N)
    out = []
    for dt in (0.05, 0.025):
        _, r, _ = ku.simulate(theta0, omega, np.array([0.8, 1.2]), t_max=40, dt=dt)
        out.append(r[:, int(r.shape[1] * 0.75):].mean(axis=1))
    assert np.abs(out[0] - out[1]).max() < 1e-4
