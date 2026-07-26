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

Related pages
-------------

* :doc:`capabilities` — benchmark inventory table
* :doc:`api` — dataset module reference
* :doc:`architecture` — simulated vs real-telemetry factory idioms
