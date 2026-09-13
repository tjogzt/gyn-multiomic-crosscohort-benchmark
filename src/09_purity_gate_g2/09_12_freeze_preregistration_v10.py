#!/usr/bin/env python3
"""
Freeze the v1.0 pre-registration plan by applying the recorded edits to its draft.

Purpose
    Stage 09, pre-registration step. The direction-F study was pre-registered as a
    series of plans, and the authoritative one is the frozen v1.0: it merges the
    completed G2 evidence and fixes the G2 threshold from the measured noise floor
    instead of the placeholder value. This module builds that frozen plan from the
    still unfrozen draft. It reads the draft plan named by
    params:preregistration.source_version, applies the anchor-and-replacement edits
    of data("preregistration_edits") in order, writes the result as the plan named
    by params:preregistration.frozen_version, and then reports whether every token
    the frozen plan must carry is present. The edit texts are the document's own
    prose and are Chinese, so they live in a data file next to the plans rather
    than in this package.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    data("preregistration_dir") with params preregistration.file_pattern,
    preregistration.source_version
        Directory of the plans and the draft plan inside it. A version label has to
        resolve to exactly one file, otherwise the wrong document could be edited.
    data("preregistration_edits")
        YAML sidecar holding the ordered edits and, for each one, the anchor text,
        its replacement text and whether the anchor is required to be present. It
        also lists the tokens the frozen plan must contain. Its texts are Chinese
        because the plan is; see docs/coding_standard.md, section 1.
    work("g2_refined")
        JSON recording the measured G2 noise floor. Its "sd_relevant" field, the
        decision-relevant paired-difference SD, is substituted into the replacement
        text of the F-55 to F-58 edit as the {sd_relevant:.4f} format field, so the
        frozen prose quotes the measured number instead of a copy of it.

Outputs
    The frozen plan, data("preregistration_dir") resolved with
    params:preregistration.frozen_version, rewritten in place.

Usage
    python src/09_purity_gate_g2/09_12_freeze_preregistration_v10.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

# Allow the module to be run from any directory: put src/ on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import data, param, work  # noqa: E402

# --- Configuration -----------------------------------------------------------

PLANS_DIR = data("preregistration_dir")
FILE_PATTERN = str(param("preregistration", "file_pattern"))
SOURCE_VERSION = str(param("preregistration", "source_version"))
FROZEN_VERSION = str(param("preregistration", "frozen_version"))
EDITS_FILE = data("preregistration_edits")
NOISE_ARTEFACT = work("g2_refined")

# Field of work("g2_refined") that is substituted into the edit texts. The package
# records that number under this key and the edit sidecar uses the same name as its
# format field, so the producer of the number and its consumer cannot drift apart.
SD_FIELD = "sd_relevant"

# Print width of the token column of the final report. It is a formatting choice of
# this module, not a parameter of the analysis.
TOKEN_COLUMN_WIDTH = 18

# Number of leading characters of an anchor quoted when the anchor is not found.
# The anchor texts are long paragraphs, and the opening characters are enough to
# identify which edit failed.
ANCHOR_PREVIEW_LENGTH = 70


def existing_plan(version: str) -> Path | None:
    """Resolve one pre-registration plan from its version label, if it exists.

    The plans are addressed by version label rather than by file name because
    their titles are not ASCII. The label must select exactly one file.

    Args:
        version: version label, e.g. "v0.3".

    Returns:
        The path of the unique plan matching the configured pattern with the label
        substituted, or None when no plan matches. None is how the not-yet-built
        frozen plan is probed.

    Raises:
        ValueError: when the label matches more than one file, because the label
            would then be ambiguous and the wrong document could be edited.
    """
    matches = sorted(PLANS_DIR.glob(FILE_PATTERN.format(version=version)))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"Version label {version!r} matches {len(matches)} files in {PLANS_DIR}; "
            "the label must identify exactly one plan."
        )
    return matches[0]


def resolve_plan(version: str) -> Path:
    """Resolve one pre-registration plan, which must exist.

    Args:
        version: version label, e.g. "v0.3".

    Returns:
        The path of the unique plan matching the configured pattern with the label
        substituted.

    Raises:
        FileNotFoundError: when no plan matches the label.
        ValueError: when the label matches more than one file.
    """
    plan = existing_plan(version)
    if plan is None:
        raise FileNotFoundError(
            f"No pre-registration plan for version {version!r} in {PLANS_DIR}."
        )
    return plan


def frozen_plan_path(source: Path) -> Path:
    """Return the path the frozen plan is written to.

    Args:
        source: path of the draft plan.

    Returns:
        The frozen plan that already exists, when there is one, because re-running
        the module has to rewrite the document in place. Otherwise the draft's
        file name with the version label swapped, which is the output name of the
        reference implementation.
    """
    existing = existing_plan(FROZEN_VERSION)
    if existing is not None:
        return existing
    return source.with_name(source.name.replace(SOURCE_VERSION, FROZEN_VERSION, 1))


def load_edits() -> tuple[list, list]:
    """Load the ordered edits and the verification tokens from the sidecar.

    Returns:
        A pair ``(edits, tokens)``. ``edits`` is the ordered list of mappings with
        the keys ``anchor``, ``replacement`` and ``required``; ``tokens`` is the
        list of substrings the frozen plan must contain.
    """
    with EDITS_FILE.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload["edits"], payload["verify_tokens"]


def noise_field_value() -> float:
    """Return the measured noise value substituted into the edit texts.

    The value is read from the artefact rather than copied into the edit sidecar,
    so the frozen prose cannot quote a stale number.

    Returns:
        The decision-relevant paired-difference SD measured in stage 09.
    """
    with NOISE_ARTEFACT.open(encoding="utf-8") as handle:
        artefact = json.load(handle)
    return float(artefact[SD_FIELD])


def apply_edits(document: str, edits: list, fields: dict) -> str:
    """Apply every edit to the document, in order, and return the result.

    Args:
        document: the full text of the draft plan.
        edits: the ordered edits from the sidecar.
        fields: values for the format fields of the replacement texts, keyed by
            field name.

    Returns:
        The edited document text.

    Raises:
        SystemExit: when a required anchor is absent, which means the draft is not
            the revision the sidecar was written against.
    """
    for edit in edits:
        anchor = edit["anchor"]
        # The replacement prose is data: it may carry the measured noise value as a
        # format field, which is the only dynamic piece of the sidecar.
        replacement = edit["replacement"].format(**fields)
        if anchor not in document:
            if edit.get("required", True):
                raise SystemExit(
                    f"Anchor not found: {anchor[:ANCHOR_PREVIEW_LENGTH]}"
                )
            # Optional anchors belong to revisions the draft may no longer carry; a
            # missing one is reported and skipped rather than being fatal.
            print(f"  optional anchor absent: {edit.get('name', '?')}")
            continue
        # The first occurrence only: the draft places each anchor once, and a line
        # repeated elsewhere in a quoted block must not be rewritten.
        document = document.replace(anchor, replacement, 1)
    return document


def report_tokens(document: str, tokens: list) -> list:
    """Print whether every required token is present in the frozen plan.

    Args:
        document: the frozen plan text.
        tokens: the tokens the document must contain.

    Returns:
        The tokens that were not found, in the order in which they are declared.
    """
    missing = []
    for token in tokens:
        present = token in document
        if not present:
            missing.append(token)
        print(f"  {token:{TOKEN_COLUMN_WIDTH}s} {'ok' if present else 'MISSING'}")
    return missing


def main() -> None:
    """Build the frozen plan from the draft and report the result.

    Returns:
        None. The frozen plan is written in place and the outcome is printed.
    """
    source = resolve_plan(SOURCE_VERSION)
    destination = frozen_plan_path(source)
    document = source.read_text(encoding="utf-8")
    original = document

    edits, tokens = load_edits()
    fields = {SD_FIELD: noise_field_value()}
    document = apply_edits(document, edits, fields)

    # A plan byte-identical to its draft means no edit found its anchor, which would
    # silently publish an unfrozen document as the frozen one.
    assert document != original, "No edit took effect: the result equals the draft."

    destination.write_text(document, encoding="utf-8")
    print(f"Wrote {destination}")
    print(f"lines {len(document.splitlines())} | bytes {len(document.encode())}")
    missing = report_tokens(document, tokens)
    if missing:
        print(f"{len(missing)} token(s) missing from the frozen plan.")


if __name__ == "__main__":
    main()
