# AGENTS.md — sdypy-EMA contributor & agent guide

Guidance for anyone (human or AI) developing **sdypy-EMA**. Claude Code reads it
via a one-line `CLAUDE.md` (`@AGENTS.md`). The org-wide rules (development
workflow, SEP governance, naming, canonical specs) live in the sdypy hub and are
linked at the end, never restated here.

## What this package is

Experimental and operational modal analysis: the `EMA` portion of the `sdypy`
namespace (PEP 420, `from sdypy import EMA`), successor of pyEMA. `Model`
estimates poles from measured FRFs (`lscf`, `lsce`, `rfp`), selects the
physical ones (stability chart or `select_closest_poles`) and computes modal
constants (`lsfd`). The public API is the curated `__all__` in
`sdypy/EMA/__init__.py`.

| Path | What lives there |
|---|---|
| `sdypy/EMA/EMA.py` | `Model`: pole estimation, selection, modal constants |
| `sdypy/EMA/stabilization.py`, `pole_picking*.py` | Stability chart; Qt window if a Qt binding is installed, else Tk |
| `sdypy/EMA/tools.py`, `normal_modes.py` | MAC, MSF, MCF, frequency/damping conversion, normal modes |
| `tests/` | Test suite (`test_tools/` holds the synthetic-FRF helpers) |
| `openspec/` | This package's own capability specs and changes |

## Development environment

Supported Python versions: `requires-python` in `pyproject.toml`, tested by the
CI matrix. Use `uv`:

```console
uv venv && uv pip install -e ".[dev]"   # dev install: docs, test and build tools
uv pip install -e ".[qt]"               # optional: Qt stability chart (PySide6)
pytest                                  # Qt tests skip without a Qt binding
python -m build                         # sdist + wheel
sphinx-build -b html docs/source docs/_build/html
```

The org-wide conformance checkers run from a hub clone next to this one; the
hub's `AGENTS.md` § Common commands lists them.

## Workflow

Changes follow the hub's OpenSpec workflow and definition of done (hub
`AGENTS.md`, linked below). This repository's `openspec/specs/` holds only
sdypy-EMA's own capabilities, and stays empty until the first non-trivial
change. A change to an org-wide contract (naming, public API, packaging
template) is proposed in the hub; its code change lands here.

## Org-wide rules (sdypy hub)

Copied verbatim from the hub's `AGENTS.md`; edit it there. The hub's template
checker compares this copy.

<!-- >>> sdypy hub links -->
- [AGENTS.md](https://github.com/sdypy/sdypy/blob/main/AGENTS.md) — workflow and definition of done
- [docs/seps/](https://github.com/sdypy/sdypy/tree/main/docs/seps/) — SEP governance docs
- [docs/source/dev/nomenclature.rst](https://github.com/sdypy/sdypy/blob/main/docs/source/dev/nomenclature.rst) — naming a new public term (SEP 2)
- [openspec/specs/](https://github.com/sdypy/sdypy/tree/main/openspec/specs/) — canonical capability specs
- [REQUIREMENTS.md](https://github.com/sdypy/sdypy/blob/main/REQUIREMENTS.md) — requirements roster
<!-- <<< sdypy hub links -->
