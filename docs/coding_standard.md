# Coding standard

Every file under `src/` follows the rules below. They exist so that the package
can be read, re-run and audited by someone who did not write it.

## 1. Language

All identifiers, comments, docstrings, log messages, exception messages, printed
output and generated filenames are **English**. No CJK characters appear in any
file under `src/`, `config/`, `docs/` or `results/`. The only exception is the
manuscript sources, which are typeset separately.

## 2. File header

Every source file opens with a header comment in the block form shown below.
The field names are literal and must appear in this order.

Python:

```python
"""
<One-line statement of what this module does.>

Purpose
    <Two to four sentences: what problem the module solves, and where it sits in
    the pipeline. State the analysis stage and what it consumes and produces.>

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    <Every file the module reads, as a config key or an argument, with the type
    and a note on what it must contain. Write "None." if the module reads no
    file.>

Outputs
    <Every file the module writes, as a config key, with the type. Write "None."
    if the module only prints or returns.>

Usage
    <The exact command line, e.g. python 12_02_build_figure_1_and_2.py>
"""
```

R:

```r
# <One-line statement of what this script does.>
#
# Purpose
#     <...>
# Author
#     Tao Zhu
# Created
#     2026-09-13
# Inputs
#     <...>
# Outputs
#     <...>
# Usage
#     Rscript 05_02_benchmark_r_packages.R
```

## 3. Naming

- Stage directories: `NN_english_stage_name`, numbered from `01` in execution
  order.
- Source files: `NN_MM_english_function_name.ext`, where `NN` is the stage number
  and `MM` the position within the stage. Sorting the directory therefore yields
  the execution order, and the name states the function.
- Functions: `snake_case`, verb first (`compute_transfer_ari`, not `ari_calc`).
- Module-level constants: `UPPER_SNAKE_CASE`, and only for values that are
  genuinely constant across runs.

## 4. No hard-coded business data

A module may not contain a literal directory, filename, cohort name, layer name,
threshold, `K`, gene count, sample count or alpha level. All of these come from
the configuration:

```python
from common.config import data, work, results, param, ensure_dir

xena_dir   = data("xena_ucec_dir")          # not "/Volumes/..."
out_path   = work("benchmark_summary")      # not "/tmp/..."
fig_dir    = results("figures_dir")
k_primary  = param("clustering", "k_primary")
top_genes  = param("features", "top_mad_genes")
layers     = param("layers")
cohorts    = param("cohorts")
```

The **only** permitted numeric literal of this kind is the random seed, and even
that should be read once from `common.config.SEED` rather than repeated:

```python
from common.config import SEED
import random
rng = random.Random(SEED)
```

Numeric literals are still allowed for genuine mathematics that is not a tunable
parameter: a 0/1 flag, an index, a rounding precision, a matrix dimension, a
threshold of exact equality (`== 0.0`).

Mock, synthetic or placeholder data is forbidden. Every value in an output must
trace back to a real input file.

## 5. Path handling

- Never build a path by string concatenation. Use `pathlib.Path`.
- Never assume the current working directory. All configuration paths are
  resolved to absolute paths by `common.config`.
- Create output directories with `ensure_dir(path)` before writing.
- A module is runnable from any directory:
  `python src/12_figures/12_02_build_figure_1_and_2.py` must work from the
  repository root and from anywhere else.

To let a module import `common`, start it with:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import data, work, param, SEED
```

## 6. Comments

- Every function and class carries a docstring stating its purpose, arguments,
  and return value.
- Inside a function, comment the *why*, not the *what*. Any non-obvious choice, a
  threshold that comes from the pre-registration, a workaround for a library
  quirk, or a subtle ordering requirement must be commented.
- Every block implementing a published method must cite the method's source in a
  comment so that the implementation can be checked against the description.
- Do not leave commented-out code. Delete it; version control remembers.

## 7. Output

- Console output is English and states what was computed. Prefer one line per
  artefact written, with the artefact path.
- Any figure or table written to `results/` must be reproducible from the
  configuration alone, with the same seed, on the same inputs.
