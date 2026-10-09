# Complex systems: metastability, networks, synchronisation

Three computational studies of collective behaviour, each pairing an analytical
prediction with a simulation or a real data set, and checking one against the
other.

| Study | Question | Main result |
|---|---|---|
| [**Metastability in a cellular automaton**](docs/pca.md) | How does a lattice of spins updated *in parallel* escape a metastable state, and through which intermediate phases? | Escape times grow exponentially with inverse temperature; without self-interaction the system passes through a checkerboard phase that is a *stable pair*, flipping parity at every step. |
| [**Real and small-world networks**](docs/networks.md) | Are a social network and a road network scale-free? When is a graph "small-world"? | Neither network passes a power-law test (bootstrap p < 0.002); the friendship paradox is exact and quantified; the Watts–Strogatz degree law matches simulation to < 0.007. |
| [**Synchronisation in the Kuramoto model**](docs/kuramoto.md) | At what coupling do oscillators lock together, and how does the transition depend on the frequency distribution? | Simulations reproduce the self-consistency theory: a continuous transition for Gaussian frequencies, a jump to r = π/4 for uniform ones. |

<p align="center">
  <img src="figures/pca_kappa0_snapshots.png" width="100%" alt="Snapshots of the cellular automaton: an all-minus lattice is invaded by checkerboard domains, which are then replaced by the all-plus state.">
</p>

## Repository layout

```
src/complexsys/   library code: pca.py, networks.py, kuramoto.py, plotting.py
scripts/          one script per study; each regenerates its figures and results/*.json
tests/            pytest suite: analytical identities, limits and cross-checks
docs/             write-up of each study: model, method, results, limitations
figures/          figures used in the write-ups
results/          numbers quoted in the write-ups, as produced by the scripts
data/             SNAP data sets (downloaded, not versioned)
```

## Reproducing everything

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                                   # 30 tests, about ten seconds
python scripts/fetch_data.py             # SNAP networks into data/
python scripts/networks_study.py         # ~3 min
python scripts/kuramoto_sync.py          # ~10 min
python scripts/pca_nucleation.py         # ~1 h (hitting-time statistics); --quick for a 1-min version
```

All random numbers come from seeded `numpy.random.Generator`s, so every figure
and number is reproducible.

## What the tests check

The tests check the code against results that are known independently:

* **Cellular automaton**: detailed balance of the parallel dynamics with
  respect to its Gibbs measure, verified on every pair of configurations of a
  3×3 torus; ground states found by exhaustive enumeration of the 2¹⁶
  configurations of a 4×4 torus; single-site update frequencies against the
  transition formula.
* **Networks**: clustering coefficients against `networkx`; the friendship
  paradox on a star graph, where it can be computed by hand; recovery of a known
  exponent by the power-law and Hill estimators; the Watts–Strogatz degree
  formula against simulated graphs.
* **Kuramoto**: conservation of the mean phase velocity; monotone
  synchronisation of identical oscillators; Kuramoto's square-root law near
  K_c; the jump to π/4 for uniform frequencies; finite-N simulation against
  the self-consistency equation.

## Background

The studies originate in the *Introduction to Complex Systems* course of the
MSc in Theoretical Physics at Utrecht University (2023–24). The code has since
been rewritten: it is vectorised (the cellular automaton runs about 180 times
faster), tested, and its statistical methods have been revised. Each write-up
records what changed and why.

## Author

Giulio Bonetti, MSc Theoretical Physics (Utrecht University).

## Licence

All rights reserved (see `LICENSE`). The code is public so that it can be read;
no permission to use, copy, modify or distribute it is granted.
