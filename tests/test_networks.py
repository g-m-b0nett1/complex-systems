import networkx as nx
import numpy as np
import pytest
from scipy import sparse

from complexsys import networks as nw


def _adj(G):
    return sparse.csr_matrix(nx.to_scipy_sparse_array(G, dtype=np.int8))


def test_clustering_matches_networkx():
    G = nx.watts_strogatz_graph(300, 8, 0.1, seed=3)
    A = _adj(G)
    assert nw.average_clustering(A) == pytest.approx(nx.average_clustering(G), rel=1e-12)


def test_from_edges_symmetrises_and_deduplicates():
    A = nw.from_edges(np.array([[0, 1], [1, 0], [1, 2], [2, 2]]))
    assert (A != A.T).nnz == 0
    assert nw.degrees(A).tolist() == [1, 2, 1]


def test_load_edge_list_relabels(tmp_path):
    f = tmp_path / "g.txt"
    f.write_text("# comment\n10 20\n20 30\n")
    A = nw.load_edge_list(f)
    assert A.shape == (3, 3)


def test_friendship_paradox_star():
    # Star with 5 leaves: leaves' neighbour has degree 5.
    A = _adj(nx.star_graph(5))
    fp = nw.friendship_paradox(A)
    assert fp.mean_degree == pytest.approx(10 / 6)
    assert fp.size_biased_mean == pytest.approx((25 + 5) / 6 / (10 / 6))
    assert fp.frac_neighbours_above_self == pytest.approx(5 / 6)


def test_poisson_gof_accepts_poisson_and_rejects_power_law():
    rng = np.random.default_rng(4)
    assert nw.poisson_gof(rng.poisson(6.0, 20_000)).p_value > 0.01
    heavy = (rng.pareto(1.5, 20_000) * 3 + 1).astype(int)
    assert nw.poisson_gof(heavy).p_value < 1e-10


def test_power_law_fit_recovers_exponent():
    rng = np.random.default_rng(5)
    k = nw._sample_discrete_power_law(2.5, 5, 20_000, rng)
    fit = nw.fit_power_law(k)
    assert fit.alpha == pytest.approx(2.5, abs=0.08)


def test_hill_on_pareto():
    rng = np.random.default_rng(6)
    x = (rng.pareto(1.5, 50_000) + 1) * 10  # survival exponent 1.5 -> alpha = 2.5
    n, a = nw.hill_estimator(x, np.array([2000]))
    assert a[0] == pytest.approx(2.5, abs=0.1)


@pytest.mark.parametrize("p", [0.2, 0.4, 1.0])
def test_ws_degree_pmf_normalised_and_matches_simulation(p):
    r, N = 5, 2000
    m = np.arange(0, 40)
    theory = np.array([nw.ws_degree_pmf(int(x), r, p) for x in m])
    assert theory.sum() == pytest.approx(1.0, abs=1e-6)
    k = np.concatenate([np.array([d for _, d in nx.watts_strogatz_graph(N, 2 * r, p, seed=s).degree()])
                        for s in range(5)])
    emp = np.bincount(k, minlength=m.size)[: m.size] / k.size
    assert np.abs(emp - theory).max() < 0.01
    assert (k >= r).all()  # p_m = 0 for m < r by construction


def test_ws_degree_pmf_p1_is_shifted_poisson():
    from scipy.stats import poisson
    for m in range(5, 20):
        assert nw.ws_degree_pmf(m, 5, 1.0) == pytest.approx(poisson.pmf(m - 5, 5))


def test_ws_clustering_lattice():
    G = nx.watts_strogatz_graph(500, 10, 0.0)
    assert nx.average_clustering(G) == pytest.approx(nw.ws_clustering(5, 0.0))
