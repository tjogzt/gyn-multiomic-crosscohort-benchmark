"""
Verify that every reference of the Discussion traces back to a real record.

Purpose
    Stage 14, fifth script. No bibliographic entry may be written from memory:
    each DOI printed in the reference list of the manuscript has to be backed by
    a record returned by a public registry. The script rebuilds the set of
    verified records from the Crossref and PubMed lookups harvested into the
    disc_refs work files, adds the lookups that were confirmed by hand, extracts
    every DOI from the reference section, and reports the DOIs that no record
    covers. It then checks that the reference numbering of the list is
    continuous and counts the bracketed in-text citations, and writes the DOI
    list together with the unresolved DOIs to the disc_verify artefact.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    manuscript("discussion")            the Discussion source of the manuscript.
    work("disc_refs")                   JSON list, Crossref lookups, first batch.
    work("disc_refs_second")            JSON list, Crossref lookups, second batch.
    work("disc_refs_third")             JSON list, PubMed lookups, third batch.
    data("reference_check_manifest")    data file: the heading that opens the
                                        reference list, the DOI pattern and the
                                        records confirmed by hand.

Outputs
    work("disc_verify")                 JSON, {"dois": [...], "bad": [...]}: every
                                        DOI found in the reference section and the
                                        subset that has no verified record.

Usage
    python 14_05_verify_references.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import CONFIG, PROJECT_ROOT, data, ensure_dir, work

# Source label attached to a record harvested from the first Crossref batch; the
# remaining labels are read from the manifest and from the batch keys below.
CROSSREF_SOURCES = [("disc_refs", "crossref-r1"), ("disc_refs_second", "crossref-r2")]
PUBMED_SOURCE = ("disc_refs_third", "pubmed-r3")


def manuscript(key: str) -> Path:
    """Resolve a manuscript source declared under ``paths.manuscript``.

    Args
        key: entry name inside the manuscript section of config/paths.yaml.

    Returns
        Absolute path of the manuscript source.
    """
    root = Path(str(CONFIG["paths"]["manuscript_root"]))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root / CONFIG["paths"]["manuscript"][key]


def load_reference_data() -> dict:
    """Read the reference-check data file declared in the configuration.

    Returns
        dict with the keys ``section_heading``, ``doi_pattern`` and
        ``confirmed_records``.
    """
    with data("reference_check_manifest").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_verified(reference_data: dict) -> dict:
    """Build the map of DOI -> (source, memory aid) that counts as verified.

    Args
        reference_data: the decoded reference-check data file.

    Returns
        dict keyed by the lower-case DOI; the value is the pair (source label,
        short descriptor) printed when a reference is matched.
    """
    verified: dict[str, tuple[str, str]] = {}

    def add(doi, source: str, meta: str) -> None:
        """Record one verified DOI; an empty DOI is a lookup that found nothing."""
        if doi:
            verified[doi.lower()] = (source, meta)

    # Crossref batches: the descriptor is the first author, year and journal.
    for key, source in CROSSREF_SOURCES:
        path = work(key)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for record in json.load(handle):
                add(
                    record.get("doi"),
                    source,
                    f"{record.get('first_author')} {record.get('year')} {record.get('journal')}",
                )

    # PubMed batch: the DOI sits among the article identifiers, and the
    # descriptor is the opening of the title.
    key, source = PUBMED_SOURCE
    path = work(key)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for record in json.load(handle):
                top = record.get("top") or {}
                dois = [
                    item["value"]
                    for item in (top.get("articleids") or [])
                    if item.get("idtype") == "doi"
                ]
                add(dois[0] if dois else None, source, str(top.get("title"))[:60])

    # The records confirmed by hand are read last, so that a hand check overrides
    # a machine lookup of the same DOI.
    for record in reference_data["confirmed_records"]:
        add(record["doi"], record["source"], record["note"])

    return verified


def main() -> None:
    """Trace every reference DOI, check the numbering and write the artefact."""
    reference_data = load_reference_data()
    verified = load_verified(reference_data)

    md = manuscript("discussion").read_text(encoding="utf-8")
    # Everything after the first reference heading is the bibliographic list; if
    # the manuscript carries no such heading the whole document is searched.
    delimiter = reference_data["section_heading"]
    refsec = md.split(delimiter, 1)[1] if delimiter in md else md

    # --- DOIs printed in the reference section -------------------------------
    dois = re.findall(reference_data["doi_pattern"], refsec)
    unique_dois = len(set(doi.lower() for doi in dois))
    print(f"DOIs in the reference section: {len(dois)}  (unique {unique_dois})\n")

    bad = []
    # dict.fromkeys keeps the first-seen order, so the report follows the list.
    for doi in dict.fromkeys(dois):
        # A DOI that closed a sentence keeps the period in the capture; strip it
        # before the lookup so that a trailing "." cannot mask a verified DOI.
        key = doi.lower().rstrip(".")
        if key in verified:
            print(f"  \u2713 {doi:44s} [{verified[key][0]}] {verified[key][1]}")
        else:
            bad.append(doi)
            print(f"  \u2717 {doi:44s} not in the verified set")
    print(f"\nnot traceable to a record: {len(bad)}")

    # --- Consistency of the reference numbering and the citations ------------
    nums = re.findall(r"^(\d+)\. ", refsec, re.M)
    continuous = nums == [str(index) for index in range(1, len(nums) + 1)]
    print(
        f"reference numbers: {len(nums)} entries, range {nums[0]}-{nums[-1]}: "
        + ("continuous" if continuous else "not continuous")
    )
    cited = re.findall(r"\[([0-9,\s\-\u2013]+)\]", md)
    print(f"bracketed in-text citations: {len(cited)}")

    out_path = work("disc_verify")
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump({"dois": dois, "bad": bad}, handle, ensure_ascii=False, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
