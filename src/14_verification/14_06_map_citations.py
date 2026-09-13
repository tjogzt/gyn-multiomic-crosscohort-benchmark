"""
Map the in-text citations of the Discussion onto its reference list.

Purpose
    Stage 14, sixth script. A citation that points at a reference the list does
    not carry, or a listed reference that no sentence ever cites, is a silent
    defect that survives copy-editing. The script splits the Discussion at the
    heading that opens its reference list, collects the reference numbers from
    the list and the bracketed citation numbers from the narrative, and prints
    both directions of the mismatch: listed but never cited, and cited but never
    listed. Every reference that is never cited is echoed with its opening text,
    so that a stale entry can be removed without opening the manuscript.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    manuscript("discussion")            the Discussion source of the manuscript.
    data("reference_check_manifest")    data file: the heading that opens the
                                        reference list.

Outputs
    None. Both mismatch directions are printed.

Usage
    python 14_06_map_citations.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import CONFIG, PROJECT_ROOT, data

# Number of characters of a reference entry echoed for an uncited reference.
ENTRY_ECHO_LENGTH = 96


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


def load_section_heading() -> str:
    """Return the heading that opens the reference list of the Discussion.

    Returns
        The heading text, read from the reference-check data file declared in the
        configuration. It is kept out of the code because it is written in the
        manuscript's own script.
    """
    with data("reference_check_manifest").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["section_heading"]


def main() -> None:
    """Print the citation-list mismatches in both directions."""
    md = manuscript("discussion").read_text(encoding="utf-8")
    # Everything before the first reference heading is the narrative that cites;
    # everything after it is the list that is cited.
    body, refsec = md.split(load_section_heading(), 1)

    refs = set()
    for match in re.finditer(r"^(\d+)\. ", refsec, re.M):
        refs.add(int(match.group(1)))

    cited = set()
    for match in re.finditer(r"\[([0-9,\s]+)\]", body):
        # A bracket may carry a group such as "[1, 3, 7]"; each field is one
        # citation number, and a non-numeric field (a numeric range, say) is
        # ignored rather than guessed at.
        for part in match.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                cited.add(int(part))

    print(f"reference list: {len(refs)} entries  ({min(refs)}-{max(refs)})")
    print(f"in-text citation numbers: {len(cited)}  -> {sorted(cited)}")
    print()
    never = sorted(refs - cited)
    undefined = sorted(cited - refs)
    print(f"listed but never cited: {never if never else 'none'}")
    print(f"cited but not listed: {undefined if undefined else 'none'}")
    print()
    if never:
        for number in never:
            line = re.search(rf"^{number}\. (.*)$", refsec, re.M)
            print(f"   [{number}] {line.group(1)[:ENTRY_ECHO_LENGTH] if line else '?'}")


if __name__ == "__main__":
    main()
