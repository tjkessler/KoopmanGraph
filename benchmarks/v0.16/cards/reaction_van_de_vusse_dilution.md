# Van de Vusse CSTR, dilution-rate input (compact payload)

* **Scope:** Van de Vusse CSTR (`A -> B -> C`, `2 A -> D`) with a
  seeded piecewise-constant dilution rate as the only control input. The
  reactor is isothermal at the nominal 387.35 K. Derived from the packaged
  `reaction_van_de_vusse_v1` card, which varies temperature and dilution
  rate together
* **Size:** 32 trajectories x 128 samples x 4 species, plus one control
  channel, `float64` (about 170 kB)
* **Format:** `ReactionDataset.save` payload at
  `benchmarks/v0.16/data/reaction_van_de_vusse_dilution_v1.pt`, with the
  derived card embedded (`controls.fixed` holds the temperature).
  Control row `t` drives `t -> t+1`
* **Units:** time in h (step 0.01 h), concentration in mol/L, dilution
  rate in 1/h, drawn uniformly from [5, 30] and held for 10 steps.
  Features are raw concentrations (no log transform)
* **Source:** Generated in-tree by `scripts/build_reaction_payloads.py`
  with `ReactionNetworkBenchmark.generate_from_card` (20 substeps per
  step, seed 0). Kinetics: K.-U. Klatt and S. Engell, Comput. Chem. Eng.
  22(4-5), 491-502, 1998. Operating point: H. Chen, A. Kremling and
  F. Allgower, Proc. 3rd ECC, 1995
* **License:** Apache-2.0 (same as the package)
* **Identity:** SHA-256 `5a69bd5400a651069bd5c907041b8cddf79ffb8ffef37edb9447d84e3f944a02`
* **Limitations:** There is no energy balance. The dilution rate
  multiplies the state in the true dynamics (`D (c_feed - c)`), so a
  model that treats it as an additive input is approximate. The species
  graph covers the reactions, not the feed. The card has no oracle
  section. Regeneration matches within `rtol=1e-5`, not byte for byte.
  Default CI checks identity only (`executed=False`)
* **Version:** `reaction_van_de_vusse_dilution_v1` / `1`
