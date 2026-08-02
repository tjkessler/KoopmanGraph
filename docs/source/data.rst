Datasets and FAIR cards
=======================

KoopmanGraph ships synthetic generators and real-telemetry teaching caches.
Oversized raw archives (over 50 MB, e.g. METR-LA HDF5) stay out of git and are
rebuilt with the fetch scripts under SHA256 verification where digests are
known. Smaller source files and ``*.pt`` teaching caches under
``data/pems_bay/``, ``data/pems07/``, and ``data/contact_epidemic/`` are
tracked. Loaders live under :mod:`koopman_graph.datasets` (not on the package
root ``__all__``).

Acquisition scripts
-------------------

Shared helpers live in :mod:`koopman_graph.datasets.download` (HTTP + SHA256)
and :mod:`koopman_graph.datasets.cache_cli` (common CLI flags). Each script
supports ``--cache-dir``, ``--force``, and ``--print-acquisition``. Remote
benchmarks use boolean ``--fetch`` to download into ``--cache-dir``.
``--fetch`` for METR-LA and PEMS-BAY always verifies SHA256 against a pinned
default-mirror digest (override with ``--expected-sha256`` when using a
non-default ``--h5-url``). PEMS0X NPZ inputs should also supply
``--expected-sha256``.

* ``scripts/download_ieee118.py`` — IEEE 118-bus MATPOWER topology cache
* ``scripts/download_metr_la.py`` — METR-LA teaching cache
* ``scripts/download_pems.py`` — PEMS-BAY and PEMS03/04/07/08 teaching caches
* ``scripts/download_contact_epidemic.py`` — SocioPatterns primary-school
  contact teaching cache (CC-BY-NC-SA; rebuild + SHA256)
* ``scripts/download_cylinder_wake.py`` — cylinder-wake Hopf surrogate cache
  (generated locally; no remote download)

Dataset card: SocioPatterns primary-school contacts
---------------------------------------------------

* **Scope:** Face-to-face proximity contacts among children and teachers in a
  primary school (20-second resolution), used in infectious-disease contact
  studies
* **Size:** 242 individuals; default teaching cache stores ``24`` hourly bins
  (``bin_seconds=3600``) of per-node contact intensity
* **Format:** SocioPatterns TSV/GZ contacts + metadata →
  ``data/contact_epidemic/contact.pt`` (``float32``, 1 channel, z-scored
  intensities; undirected edges weighted by contact count in the teaching
  window)
* **Source:** SocioPatterns Primary School temporal network
  (https://sociopatterns.org/datasets/primary-school-temporal-network-data/);
  cite Gemmetto et al., BMC Infectious Diseases 14:695 (2014) and Stehlé et
  al., PLoS ONE 6(8):e23176 (2011); acknowledge the SocioPatterns
  collaboration
* **License:** CC-BY-NC-SA — upstream non-commercial / ShareAlike terms apply
  to the tracked raw contacts and derived teaching cache; commercial
  redistribution of those files is not permitted under the upstream license
* **Limitations:** Contact intensity is not SIR compartment state; the
  teaching window is a short aggregate; class/gender metadata are attributes,
  not dynamics; NC-SA restricts commercial redistribution of the upstream data
* **Version:** ``contact_epidemic_v1`` teaching-cache schema

Loader: :class:`~koopman_graph.datasets.ContactEpidemicBenchmark`.

Dataset card: PEMS-BAY
----------------------

* **Scope:** Bay Area highway loop-detector **speeds** (5-minute aggregation)
* **Size:** 325 sensors; full DCRNN history ~52k steps; default teaching cache
  stores one day (``288`` steps)
* **Format:** DCRNN HDF5 speeds + distance CSV → ``data/pems_bay/traffic.pt``
  (``float32``, 1 channel, z-scored)
* **Source:** Caltrans PeMS; DCRNN release (`Li2018DCRNN`); sensor graph from
  the DCRNN GitHub ``data/sensor_graph`` tree
* **License:** Upstream Caltrans PeMS data-use terms; DCRNN code MIT — raw
  HDF5 is not tracked (fetch + checksum); the teaching cache ``traffic.pt``
  is shipped
* **Limitations:** Zeros encode missing readings (imputed); teaching cache is a
  short window; adjacency is a distance-kernel approximation
* **Version:** ``pems_bay_v1`` teaching-cache schema

Loader: :class:`~koopman_graph.datasets.PemsBayTrafficBenchmark`.

Dataset card: PEMS03
--------------------

* **Scope:** Caltrans PeMS District 03 highway **flow** (5-minute)
* **Size:** 358 sensors; default teaching cache ``288`` steps
* **Format:** community ``PEMS03.npz`` (``data`` channel 0) + dense adjacency
  CSV → ``data/pems03/traffic.pt``
* **Source:** Caltrans PeMS; community NPZ packaging as used by ASTGCN /
  STFGNN-style releases (cite the archive you download — no invented DOI)
* **License:** Upstream Caltrans PeMS terms; do not bundle raw NPZ
* **Limitations:** Flow (not speed); static packaged adjacency; short teaching
  window
* **Version:** ``pems0x_v1`` teaching-cache schema

Loader: :class:`~koopman_graph.datasets.PemsTrafficBenchmark` with
``variant="03"``.

Dataset card: PEMS04
--------------------

* **Scope:** Caltrans PeMS District 04 highway **flow** (5-minute)
* **Size:** 307 sensors; default teaching cache ``288`` steps
* **Format:** community ``PEMS04.npz`` + adjacency CSV →
  ``data/pems04/traffic.pt``
* **Source:** Caltrans PeMS; community NPZ packaging (ASTGCN / STFGNN-style)
* **License:** Upstream Caltrans PeMS terms; do not bundle raw NPZ
* **Limitations:** Flow (not speed); static adjacency; short teaching window
* **Version:** ``pems0x_v1`` teaching-cache schema

Loader: :class:`~koopman_graph.datasets.PemsTrafficBenchmark` with
``variant="04"``.

Dataset card: PEMS07
--------------------

* **Scope:** Caltrans PeMS District 07 highway **flow** (5-minute)
* **Size:** 883 sensors; default teaching cache ``288`` steps
* **Format:** community ``PEMS07.npz`` + adjacency CSV →
  ``data/pems07/traffic.pt`` (raw ``PEMS07.npz`` / adjacency under
  ``data/pems07/raw/`` are tracked; each file is under 50 MB)
* **Source:** Caltrans PeMS; community NPZ packaging (ASTGCN / STFGNN-style)
* **License:** Upstream Caltrans PeMS data-use terms
* **Limitations:** Larger graph (scale costs); flow only; short teaching window
* **Version:** ``pems0x_v1`` teaching-cache schema

Loader: :class:`~koopman_graph.datasets.PemsTrafficBenchmark` with
``variant="07"``.

Dataset card: PEMS08
--------------------

* **Scope:** Caltrans PeMS District 08 highway **flow** (5-minute)
* **Size:** 170 sensors; default teaching cache ``288`` steps
* **Format:** community ``PEMS08.npz`` + adjacency CSV →
  ``data/pems08/traffic.pt``
* **Source:** Caltrans PeMS; community NPZ packaging (ASTGCN / STFGNN-style)
* **License:** Upstream Caltrans PeMS terms; do not bundle raw NPZ
* **Limitations:** Flow (not speed); static adjacency; short teaching window
* **Version:** ``pems0x_v1`` teaching-cache schema

Loader: :class:`~koopman_graph.datasets.PemsTrafficBenchmark` with
``variant="08"``.

Dataset card: synthetic two-state molecular oracle
--------------------------------------------------

* **Scope:** Seeded two-state Markov teaching trajectory on a fixed
  four-atom contact graph for GraphVAMP / implied-timescale CI oracles
* **Size:** 4 nodes, 2 feature channels; default ``256`` snapshot steps
  (procedural — no large binary in git)
* **Format:** :class:`~koopman_graph.data.GraphSnapshotSequence` of PyG
  ``Data`` (``float32`` node features, ``long`` contact ``edge_index``);
  oracle constants in package resource
  ``koopman_graph/datasets/molecular/data/synthetic_two_state_v1.json``
* **Source / provenance:** In-repo generator
  :func:`~koopman_graph.datasets.molecular.generate_synthetic_two_state`
  (:mod:`koopman_graph.datasets.molecular.synthetic`); not experimental
  molecular dynamics. Contact topology from fixed positions via
  :func:`~koopman_graph.datasets.molecular.contact_edge_index` (cutoff
  ``0.5`` nm)
* **Citation:** No external publication — package teaching fixture
  (``synthetic_two_state_v1``)
* **License:** Apache-2.0 (same as the package)
* **Units:** Positions and contact cutoff in **nm**; teaching timestep
  ``1.0`` **ps** per snapshot; oracle slow timescale in **snapshot steps**
  at lag ``1`` (closed form :math:`-1 / \ln(1 - 2p)` with default
  ``p = 0.05`` → :math:`\lambda = 0.9`, timescale approximately ``9.49``
  steps)
* **Limitations:** Toy two-state switching with additive Gaussian feature
  noise; not a biomolecule; not Folding@home-scale MD; not a PyEMMA
  replacement. Optional ``[md]`` / ``load_md_trajectory`` stubs do **not**
  ship a public alanine-dipeptide or other experimental MD loader
* **Version:** ``synthetic_two_state_v1``

Loader: :func:`~koopman_graph.datasets.molecular.generate_synthetic_two_state`
(metadata:
:func:`~koopman_graph.datasets.molecular.load_synthetic_two_state_metadata`).
See ``examples/44_graphvamp_md.ipynb`` and :doc:`tutorials`.

Presence masks vs observation masks
-----------------------------------

Sequences may carry two independent boolean mask stacks. Do not conflate
them:

* **Observation masks** (``observation_masks``, shape ``(T, N)`` or typed
  hetero equivalent) mark which nodes are **measured** at each timestep
  (``True`` = observed). They support partial observability and imputation
  workflows (e.g. notebook 17 / 25). A node may be present in the universe
  but unobserved.
* **Presence masks** (``presence_masks``, shape :math:`(T, N_{\max})`) mark
  which entities are **active in the fixed union** at each timestep
  (``True`` = present). Drops require ``allow_node_churn=True`` on
  :class:`~koopman_graph.data.GraphSnapshotSequence` (or the typed hetero
  peer). Inactive rows stay at capacity :math:`N_{\max}` (padded zeros is
  conventional); losses ignore inactive nodes; operator matvecs still run
  at full capacity. This is **not** unbounded open-world graph growth
  (see :doc:`limitations`).

Default sequences have neither stack (fully present and fully observed).
Observation masks alone do **not** model entity drop-in/out; presence masks
alone do **not** mark measurement gaps. See
``examples/41_node_churn_presence_masks.ipynb``.

Related pages
-------------

* :doc:`capabilities` — benchmark inventory table
* :doc:`limitations` — churn / MD honesty boundaries
* :doc:`tutorials` — notebook gallery (including molecular / churn demos)
* :doc:`api` — dataset module reference
* :doc:`architecture` — simulated vs real-telemetry factory idioms
