#!/usr/bin/env python3
"""
Mark superseded pre-registration plans and check their version status.

Purpose
    Stage 07, documentation step. The direction-F pre-registration series exists
    as four plans (v0.1, v0.2, v0.3 and the frozen v1.0) in the analysis
    workspace, and the frozen version is the authoritative one. This script makes
    that unambiguous in the documents themselves: every superseded plan receives
    a banner directly below its changelog paragraph, stating that the frozen
    version supersedes it, and the script then reports for every plan of the
    series whether its status is annotated. It is idempotent - a plan that
    already carries the banner is left untouched - so it can be re-run at any
    time. The plans are located by version label rather than by file name,
    because their titles are not ASCII and no non-ASCII character may appear in
    this package.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    data.preregistration_dir
        Directory holding the pre-registration plans; addressed through the file
        pattern in params preregistration.file_pattern, which must resolve to
        exactly one plan per version label.
    params preregistration.version_labels, preregistration.superseded_versions,
    params preregistration.superseded_banner, preregistration.banner_marker,
    params preregistration.frozen_status_marker, preregistration.changelog_prefix.

Outputs
    None. The superseded plan files are rewritten in place with the banner
    inserted; the status of every plan is printed.

Usage
    python src/07_batch_testbed/07_05_decode_crossed_design.py
"""

import sys
from pathlib import Path

# The package root (one level above src/) so that "common" can be imported from
# any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import data, param

# --- Configuration -----------------------------------------------------------

PLANS_DIR = data("preregistration_dir")
FILE_PATTERN = param("preregistration", "file_pattern")
VERSION_LABELS = [str(label) for label in param("preregistration", "version_labels")]
SUPERSEDED_VERSIONS = [str(label) for label in param("preregistration", "superseded_versions")]
BANNER = param("preregistration", "superseded_banner")
BANNER_MARKER = param("preregistration", "banner_marker")
FROZEN_STATUS_MARKER = param("preregistration", "frozen_status_marker")
CHANGELOG_PREFIX = param("preregistration", "changelog_prefix")

# Line the banner is inserted after when a plan has no changelog paragraph: the
# second line, i.e. directly below the title. Plans without a changelog are the
# earliest drafts, which carry no change summary at all.
DEFAULT_INSERTION_POSITION = 1


def locate_plan(version: str) -> Path:
    """Resolve one pre-registration plan from its version label.

    Args:
        version: version label, e.g. "v0.2".

    Returns:
        The path of the unique file in PLANS_DIR that matches the configured
        pattern with the label substituted.

    Raises:
        FileNotFoundError: when no file matches the pattern.
        ValueError: when more than one file matches, because the label would then
            be ambiguous and the wrong document could be edited.
    """
    matches = sorted(PLANS_DIR.glob(str(FILE_PATTERN).format(version=version)))
    if not matches:
        raise FileNotFoundError(
            f"No pre-registration plan for version {version!r} in {PLANS_DIR}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Version label {version!r} matches {len(matches)} files in {PLANS_DIR}; "
            "the label must identify exactly one plan."
        )
    return matches[0]


def banner_position(lines: list) -> int:
    """Return the line index the banner has to be inserted at.

    Args:
        lines: the document split into lines.

    Returns:
        One past the changelog paragraph when the plan has one, otherwise
        DEFAULT_INSERTION_POSITION. The changelog paragraph is found by its
        configured prefix: the reference implementation also required a
        non-ASCII keyword in the line (the word for "changes"), which cannot
        appear in this package, and the prefix alone selects the same line in
        every plan of the series.
    """
    for index, line in enumerate(lines):
        if line.startswith(CHANGELOG_PREFIX):
            return index + 1
    return DEFAULT_INSERTION_POSITION


def annotate_plan(version: str) -> int | None:
    """Insert the superseded banner into one plan unless it is already there.

    Args:
        version: version label of the plan.

    Returns:
        The 1-based line number of the inserted banner, or None when the plan
        already carries the banner and was left unchanged.
    """
    path = locate_plan(version)
    text = path.read_text(encoding="utf-8")
    if BANNER_MARKER in text:
        print(f"{version}: banner already present, skipped")
        return None
    lines = text.split("\n")
    position = banner_position(lines)
    lines.insert(position, BANNER)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"{version}: banner inserted at line {position + 1}")
    return position + 1


def report_status(version: str) -> bool:
    """Report whether one plan carries a status annotation.

    Args:
        version: version label of the plan.

    Returns:
        True when the plan is annotated: either it carries the superseded banner,
        or it carries the frozen-status marker, which is how the authoritative
        frozen plan is recognised. The original check used a non-ASCII status
        phrase for the latter; the ASCII marker is the equivalent token that the
        frozen plan's status line contains.
    """
    text = locate_plan(version).read_text(encoding="utf-8")
    annotated = BANNER_MARKER in text or FROZEN_STATUS_MARKER in text
    print(f"  {version}: {'status annotated' if annotated else 'NO status annotation'}")
    return annotated


def main() -> dict:
    """Annotate the superseded plans and report the status of the whole series.

    Returns:
        Mapping version label -> True when the plan is annotated.
    """
    for version in SUPERSEDED_VERSIONS:
        annotate_plan(version)
    print("\nStatus of the pre-registration series:")
    status = {version: report_status(version) for version in VERSION_LABELS}
    return status


if __name__ == "__main__":
    main()
