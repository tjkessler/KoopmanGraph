# Michaelis-Menten reaction network (compact payload)

* **Scope:** Mass-action Michaelis-Menten `E + S <-> ES -> E + P`, the
  packaged `reaction_michaelis_menten_v1` card regenerated on a
  128-sample grid so that the longest manifest horizon (100) plus the
  history length (8) fits in each trajectory
* **Size:** 32 trajectories x 128 samples x 4 species, `float64`
  (about 137 kB)
* **Format:** `ReactionDataset.save` payload (`torch.save`,
  `weights_only` loading) at
  `benchmarks/v0.16/data/reaction_michaelis_menten_v1.pt`; the card is
  embedded in the payload. Load with `ReactionDataset.load`
* **Units:** time in s (step 0.5 s), concentration in mol/L. Node
  features are `log(max(c, 1e-8))`; the payload stores raw
  concentrations
* **Source:** Generated in-tree by `scripts/build_reaction_payloads.py`
  (RK4, 50 substeps per step, initial-condition seed 0)
* **License:** Apache-2.0 (same as the package)
* **Identity:** SHA-256 `e4c6c42b47f23f95b6b6bf94ecb4a28cf6cff1e23f3c78a5d75edbab58918ad0`
* **Limitations:** Irreversible catalysis, so trajectories end on the
  boundary `S = ES = 0`, where the log transform saturates at the
  floor. Element balance holds to rounding and is a metric, not a
  constraint. Regeneration on another platform matches within
  `rtol=1e-5`, not byte for byte; the manifest binds the committed bytes.
  Default CI checks identity only (`executed=False`)
* **Version:** `reaction_michaelis_menten_v1` / `1`
