# Security Policy

## Supported versions

Security updates are provided for the **latest published release** of
`koopman-graph` on [PyPI](https://pypi.org/project/koopman-graph/).

| Version lineage | Security updates |
| --------------- | ---------------- |
| Latest PyPI release (currently the 0.14.x line) | Supported |
| Older published majors/minors | Not regularly backported |

Check the installed version with:

```bash
python -c "import koopman_graph; print(koopman_graph.__version__)"
```

Upgrade when a fixed release is available:

```bash
pip install -U koopman-graph
```

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report privately by email to the maintainer:

[travis.j.kessler@gmail.com](mailto:travis.j.kessler@gmail.com)

Include:

- A description of the issue and its impact
- Steps to reproduce or a proof of concept (if available)
- Affected versions / commit if known

You should receive an acknowledgment when the report is reviewed. Fixes are
coordinated with the reporter when practical before a public disclosure or
release notes mention.

This channel is the same maintainer contact used for
[Code of Conduct](CODE_OF_CONDUCT.md) enforcement.

## Loading checkpoints and dataset caches

Model checkpoints use one of two on-disk containers (logical architecture
schema remains format-1 either way):

### Preferred: `safetensors_v1` directory or `.kgckpt` zip

Default `GraphKoopmanModel.save` / `save_checkpoint` writes a **directory**
containing `meta.json`, `config.json` (architecture metadata), and
`model.safetensors` (weights). When the destination path ends in `.kgckpt`
or `.zip`, the same three files are packed into a zip bundle (still no
pickle). Weight tensors load through the safetensors library rather than
pickle. Prefer this container when sharing trained models.

The JSON sidecar files are not executable code, but an untrusted checkpoint can
still carry misleading architecture metadata or adversarial weight values that
affect numerical results. Treat third-party directories or `.kgckpt` files as
untrusted inputs; inspect `config.json` before load when provenance is unclear.

### Legacy: `legacy_pt` pickle file

Single ``*.pt`` / ``*.pth`` files (and some training `checkpoint_path`
writers that still pin this format) are loaded with PyTorch
``torch.load(..., weights_only=False)``. That API deserializes Python objects
and can execute code from a malicious file. Pass ``format="legacy_pt"`` only
when you need a single pickle file.

**Trust boundary for `legacy_pt`:** only load pickle checkpoints that you
created yourself or obtained from a source you trust. Do not load untrusted
``.pt`` / checkpoint files shared over the network or from unknown mirrors.

### Teaching caches and other ``*.pt`` artifacts

Conformal calibration files, hierarchical wrappers, and on-disk teaching
caches (``*.pt``) remain pickle-based. Prefer rebuilding teaching caches with
the repository download scripts (SHA256-verified where digests are pinned)
rather than accepting third-party ``traffic.pt`` / ``contact.pt`` blobs.
