# Security Policy

## Supported versions

Security updates are provided for the **latest published release** of
`koopman-graph` on [PyPI](https://pypi.org/project/koopman-graph/).

| Version lineage | Security updates |
| --------------- | ---------------- |
| Latest PyPI release (currently the 0.4.x line until 0.5.0 is published) | Supported |
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
