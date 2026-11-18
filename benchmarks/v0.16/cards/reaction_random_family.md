# Random mass-action network family (compact payloads)

* **Scope:** Twelve random element-balanced mass-action networks from
  `sample_random_network`, four at each species count `S` in
  {8, 12, 16}, as the training side of a topology-transfer protocol.
  Members are the first network seeds whose species graph is connected
  and uses every species: seeds 0, 1, 2, 5 at `S = 8`; 0, 1, 2, 3 at
  `S = 12`; 0, 2, 4, 5 at `S = 16`
* **Size:** 16 trajectories x 128 samples x `S` species per member,
  `float64` (about 2.5 MB in total)
* **Format:** One `ReactionDataset.save` payload per member under
  `benchmarks/v0.16/data/reaction_random_family_v1/`, plus
  `index.json` (format `koopman_graph.reaction_family`, version 1),
  which lists every member's file, SHA-256, `S`, reaction count, and
  network seed, with the sampler settings and the shift protocol. The
  manifest's `dataset.sha256` is the SHA-256 of `index.json`
* **Units:** time in s (step 0.1 s), concentration in mol/L. Node
  features are `log(max(c, 1e-8))`; the payloads store raw
  concentrations
* **Sampler:** 3 elements, at most 5 atoms per species, at most second
  order, `3 S / 2` reactions, `log10 k` uniform in [-1, 1], initial
  concentrations uniform in [0.1, 1.0] mol/L; RK4 with 20 substeps per
  step, initial-condition seed 0
* **Shifts:** `unseen_n` tests on `S = 32` networks from the same
  sampler; `rewire` keeps `S` and the reaction count and changes the
  network seed; `parameter_shift` draws `log10 k` from [1, 2]. The
  shift test networks are not committed
* **Source:** Generated in-tree by `scripts/build_reaction_payloads.py`
  with `ReactionNetworkBenchmark.generate_from_card`
* **License:** Apache-2.0 (same as the package)
* **Identity:** `index.json` SHA-256 `33d6b2250a7aef305f8b787140c942ea193482a9025f801672c5f595d3acdfd3`
* **Limitations:** The networks are synthetic and have no reverse
  reactions. Some species decay far below the log floor. Member cards
  have no oracle section, because the equilibrium solve does not
  converge on every sampled network. Negative transfer remains allowed.
  Regeneration matches within `rtol=1e-5`, not byte for byte. Default
  CI checks identity only (`executed=False`)
* **Version:** `reaction_random_family_v1` / `1`
