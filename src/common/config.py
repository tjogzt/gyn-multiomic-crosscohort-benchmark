"""
Configuration loader for the cross-cohort multi-omic integration benchmark.

Purpose
    Provide the single access point through which every module obtains
    filesystem locations and analysis parameters. No other module may contain a
    literal directory, filename, threshold or cohort name: all of them are read
    from config/paths.yaml and config/params.yaml through the helpers below.

Author
    Tao Zhu

Created
    2026-09-13

Design
    - ``config/paths.yaml`` holds filesystem locations.
    - ``config/params.yaml`` holds analysis parameters and categorical
      vocabularies.
    - ``${var}``, ``${env:NAME}`` and ``${env:NAME:-default}`` are expanded
      inside string values, so later entries can be written in terms of earlier
      ones.
    - Any leaf may be overridden by an environment variable named after its
      dotted path in upper case, e.g. ``data.xena_dir`` becomes
      ``GYN_DATA_XENA_DIR`` and ``data_root`` becomes ``GYN_DATA_ROOT``. This is
      how a run is redirected to another data volume or work directory without
      editing the repository.

Typical use
    >>> from common.config import data, work, results, param, SEED
    >>> xena = data("xena_ucec_dir")          # Path to the Xena UCEC directory
    >>> out = work("alignment_concordance")   # Path inside the work root
    >>> k = param("clustering", "k_primary")  # 4, from params.yaml
    >>> rng = random.Random(SEED)

Public API
    CONFIG          resolved configuration tree (dict)
    PROJECT_ROOT    repository root
    SEED            global random seed
    data(*keys)     resolve a path under the "data" section
    work(*keys)     resolve a path under the work root
    results(*keys)  resolve a path under the results root
    param(*keys)    resolve an analysis parameter
    ensure_dir(p)   create a directory (and parents) if absent
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

try:  # PyYAML is available in the analysis environment.
    import yaml
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "PyYAML is required to read config/*.yaml. Install it with "
        "'pip install pyyaml'."
    ) from exc

# --- Locations of the configuration files themselves -------------------------

# src/common/config.py -> the repository root is two levels up from src/common.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "config"
PATHS_FILE: Path = CONFIG_DIR / "paths.yaml"
PARAMS_FILE: Path = CONFIG_DIR / "params.yaml"

# Prefix used for environment overrides of individual configuration leaves.
ENV_PREFIX = "GYN_"

_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


# --- Variable expansion ------------------------------------------------------


_VAR_OPEN = "${"


def _resolve_spec(spec: str, tree: dict, depth: int) -> str:
    """Resolve the inside of a ``${...}`` specification into a string.

    Accepts three forms:
        ``name``               -- another entry in the same configuration tree
        ``env:NAME``           -- the environment variable NAME, required
        ``env:NAME:-default``  -- the environment variable NAME, or a default;
                                  the default may itself contain ``${...}``
    """
    if spec.startswith("env:"):
        body = spec[4:]
        if ":-" in body:
            name, default = body.split(":-", 1)
            if name in os.environ:
                return os.environ[name]
            # The default may contain further references, so expand it.
            return _scan_expand(default, tree, depth + 1)
        if body in os.environ:
            return os.environ[body]
        raise KeyError(
            f"Environment variable {body!r} referenced by the configuration "
            f"is not set and has no default."
        )
    found = _find_leaf(tree, spec)
    if found is None:
        raise KeyError(f"Configuration reference {spec!r} not found.")
    return str(_expand(found, tree, depth + 1))


def _scan_expand(text: str, tree: dict, depth: int) -> str:
    """Expand every ``${...}`` in ``text``, matching braces so that defaults may
    themselves contain references (``${env:A:-${b}/x}``)."""
    out: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith(_VAR_OPEN, i):
            j = i + 2
            level = 1
            while j < len(text) and level > 0:
                if text.startswith(_VAR_OPEN, j):
                    level += 1
                    j += 2
                    continue
                if text[j] == "}":
                    level -= 1
                j += 1
            if level != 0:
                raise ValueError(f"Unbalanced ${{...}} in configuration: {text!r}")
            out.append(_resolve_spec(text[i + 2:j - 1], tree, depth))
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _expand(value: Any, tree: dict, depth: int = 0) -> Any:
    """Recursively expand ``${...}`` references inside a configuration value.

    ``tree`` supplies the values for plain-name references and is searched
    depth-first, so an entry may reference any other entry regardless of the
    order in which the two appear in the file.
    """
    if depth > 12:  # guards against a reference cycle
        raise RecursionError("Configuration ${...} references form a cycle.")
    if isinstance(value, dict):
        return {k: _expand(v, tree, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, tree, depth + 1) for v in value]
    if not isinstance(value, str):
        return value
    return _scan_expand(value, tree, depth)


def _find_leaf(tree: dict, name: str) -> Any | None:
    """Return the first value stored under ``name`` anywhere in ``tree``."""
    for key, value in tree.items():
        if key == name:
            return value
        if isinstance(value, dict):
            hit = _find_leaf(value, name)
            if hit is not None:
                return hit
    return None


# --- Environment overrides ---------------------------------------------------


def _apply_env_overrides(tree: dict, path: tuple[str, ...] = ()) -> dict:
    """Replace leaves whose ``GYN_<DOTTED_PATH>`` environment variable is set."""
    out = {}
    for key, value in tree.items():
        here = path + (key,)
        if isinstance(value, dict):
            out[key] = _apply_env_overrides(value, here)
        else:
            env_name = ENV_PREFIX + "_".join(here).upper()
            if env_name in os.environ:
                raw = os.environ[env_name]
                out[key] = _coerce(raw, value)
            else:
                out[key] = value
    return out


def _coerce(raw: str, template: Any) -> Any:
    """Cast an environment override to the type of the value it replaces."""
    if isinstance(template, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(template, int) and not isinstance(template, bool):
        try:
            return int(raw)
        except ValueError:
            return raw
    if isinstance(template, float):
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw


# --- Load --------------------------------------------------------------------


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required configuration file missing: {path}")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must contain a mapping at the top level.")
    return loaded


_paths_raw = _read_yaml(PATHS_FILE)
_params_raw = _read_yaml(PARAMS_FILE)

# project_root is injected so that paths.yaml can refer to it.
_paths_raw.setdefault("project_root", str(PROJECT_ROOT))

CONFIG: dict = {
    "paths": _apply_env_overrides(_expand(_paths_raw, _paths_raw)),
    "params": _apply_env_overrides(_expand(_params_raw, _params_raw)),
}

SEED: int = int(CONFIG["params"]["seed"])

# Label vocabularies for the non-English keys carried by the stage artefacts.
# Declared in config/legacy_label_map.yaml, written with \uXXXX escapes so the
# file stays ASCII. Kept separate from CONFIG because it is a lookup table for
# display, not a parameter of the analysis.
LABELS_FILE: Path = CONFIG_DIR / "legacy_label_map.yaml"
LABELS: dict = yaml.safe_load(LABELS_FILE.open(encoding="utf-8")) if LABELS_FILE.exists() else {}


def label(section: str, key: str, default: str | None = None) -> str | None:
    """Return the display name recorded for an artefact key.

    Args:
        section: section of legacy_label_map.yaml, e.g. "layer".
        key: the artefact's own key, which may be non-ASCII.
        default: value returned when the section or key is absent.

    Returns:
        The display name, or ``default``.
    """
    return LABELS.get(section, {}).get(key, default)



# --- Accessors ---------------------------------------------------------------


def _resolve(section: str, keys: tuple[str, ...], base: Path | None = None) -> Path:
    """Walk ``keys`` into a configuration section and return a resolved path."""
    node: Any = CONFIG[section]
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"No configuration entry {section}:{'.'.join(keys)}")
        node = node[key]
    path = Path(str(node))
    if not path.is_absolute():
        if base is None:
            raise ValueError(
                f"{section}:{'.'.join(keys)} is relative but has no base "
                f"directory. Use data(), work() or results() for such entries."
            )
        path = base / path
    return path


def data(*keys: str) -> Path:
    """Return a resolved path from the ``paths.data`` section."""
    return _resolve("paths", ("data",) + keys)


def work(*keys: str) -> Path:
    """Return a resolved path inside the work root.

    ``work("benchmark_summary")`` returns
    ``<work_root>/bench_summary.json``; directory keys resolve to directories.
    """
    return _resolve("paths", ("work",) + keys, base=work_root())


def results(*keys: str) -> Path:
    """Return a resolved path inside the results root."""
    return _resolve("paths", ("results",) + keys, base=results_root())


def work_root() -> Path:
    """Return the resolved work root, creating it on first use."""
    value = CONFIG["paths"]["work_root"]
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def results_root() -> Path:
    """Return the resolved results root."""
    value = CONFIG["paths"]["results_root"]
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def param(*keys: str) -> Any:
    """Return an analysis parameter, e.g. ``param("features", "top_mad_genes")``."""
    node: Any = CONFIG["params"]
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"No parameter params:{'.'.join(keys)}")
        node = node[key]
    return node


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (including parents) if it does not exist; return it."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def describe() -> str:
    """Return a short human-readable summary of the resolved configuration."""
    return (
        f"project_root = {PROJECT_ROOT}\n"
        f"data_root    = {CONFIG['paths']['data_root']}\n"
        f"work_root    = {work_root()}\n"
        f"results_root = {results_root()}\n"
        f"seed         = {SEED}"
    )


if __name__ == "__main__":
    print(describe())
