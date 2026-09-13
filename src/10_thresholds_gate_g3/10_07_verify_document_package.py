"""
Check every built report PDF against the identifier tokens of its source.

Purpose
    Stage 10, seventh script: the standalone verification pass of the document
    package. For every document the manifest marks for re-checking, it extracts
    from the Markdown source each whitespace-free identifier token that is at
    least params:reports.token_min_length long and looks it up in the text layer
    of the built PDF. No hand-written token list is used, so the pass is
    independent of the per-document expectations that 10_06 applies: it reports a
    token that was dropped or broken at line breaking even when no one wrote it
    down. Page count and the number of replacement characters are reported next
    to the missing tokens, because a document that lost pages or glyphs would
    otherwise pass.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    data("report_documents_manifest") -- JSON manifest of the document package.
        Only its ``verification_documents`` entry is read here: that is the
        subset of documents this pass re-checks on its own.
    data("report_output_dir") -- directory holding the Markdown sources and the
        PDFs built from them by 10_06.
    params:reports.token_min_length -- minimum length of an identifier token
        extracted from a source.
    params:reports.markdown_extension, reports.pdf_extension -- file extensions
        of the source and of the PDF of a document.

Outputs
    None. The pass prints, per document, the page count, the number of source
    tokens, the number of tokens missing from the PDF, the number of replacement
    characters and the first few missing tokens, then the total.

Usage
    python src/10_thresholds_gate_g3/10_07_verify_document_package.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF; used to read the text layer of a built PDF.

# Put src/ on the import path so that the module runs from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import data, param  # noqa: E402

# Character class of an identifier token: the characters that make up an
# accession, a data-file name, a DOI or a path fragment. A run of these with no
# whitespace in it is what the typesetter cannot break, so it is what can be lost
# at line breaking and what this pass therefore compares.
IDENTIFIER_CHARACTERS = r"[A-Za-z0-9_.:/\-\[\]<>+=]"

# U+FFFD REPLACEMENT CHARACTER: what the text layer carries for a character that
# was typeset as a box.
REPLACEMENT_CHARACTER = "\ufffd"

# Width of the rule printed above the report.
RULE_WIDTH = 84

# Number of missing tokens printed per document; the count above the list is the
# full count, the list only names the first few so that the report stays short.
MISSING_LIST_LIMIT = 8


def load_document_manifest() -> dict:
    """Load the JSON manifest of the document package.

    Returns:
        The parsed manifest; this pass reads its ``verification_documents``
        entry and its per-document file-name stems.
    """
    with data("report_documents_manifest").open(encoding="utf-8") as handle:
        return json.load(handle)


def verification_documents(manifest: dict) -> list[tuple[str, str]]:
    """Return the documents this pass re-checks, in the declared order.

    Args:
        manifest: the parsed document manifest.

    Returns:
        A list of (document key, file-name stem) pairs. The subset and its order
        come from the manifest, so which documents are verified is data and not
        a decision taken here.
    """
    return [
        (key, manifest["documents"][key]["file_stem"])
        for key in manifest["verification_documents"]
    ]


def identifier_pattern() -> re.Pattern:
    """Return the regular expression that extracts identifier tokens.

    Returns:
        A compiled pattern matching a whitespace-free run of identifier
        characters at least params:reports.token_min_length long. The length
        bound comes from the configuration because it decides how many tokens the
        pass compares and is quoted in the manuscript methods.
    """
    return re.compile(
        IDENTIFIER_CHARACTERS + r"{" + str(param("reports", "token_min_length")) + r",}"
    )


def source_path(report_dir: Path, stem: str) -> Path:
    """Return the Markdown source path of one document.

    Args:
        report_dir: the directory declared as data:report_output_dir.
        stem: file-name stem of the document, without extension.

    Returns:
        The path of the Markdown source, which is the reference text of the
        comparison.
    """
    return report_dir / f"{stem}{param('reports', 'markdown_extension')}"


def pdf_path(report_dir: Path, stem: str) -> Path:
    """Return the PDF path of one document.

    Args:
        report_dir: the directory declared as data:report_output_dir.
        stem: file-name stem of the document, without extension.

    Returns:
        The path of the PDF built by 10_06.
    """
    return report_dir / f"{stem}{param('reports', 'pdf_extension')}"


def read_pdf_text(path: Path):
    """Return the page count and the text layer of a PDF.

    Args:
        path: the PDF to read.

    Returns:
        A (page count, text) pair; the text is the concatenation of every page in
        page order.
    """
    with fitz.open(path) as document:
        pages = [str(document[index].get_text()) for index in range(document.page_count)]
        return document.page_count, "".join(pages)


def extract_identifiers(text: str) -> list[str]:
    """Return the distinct identifier tokens of a text, in sorted order.

    Args:
        text: the Markdown source, read as it stands.

    Returns:
        The sorted set of tokens at or above the configured minimum length. The
        source is scanned as it is, link targets and fenced code blocks
        included: the original pass did the same, and both sides of the
        comparison then see the same text, so a token is only reported when it
        truly disappeared from the typeset PDF.

    Notes:
        The comparison against the PDF drops whitespace, because a line break
        inserted inside a long identifier is not part of the identifier.
    """
    return sorted(set(identifier_pattern().findall(text)))


def main() -> None:
    """Check every document this pass owns and print the totals.

    Returns:
        None. The pass only reports; a missing token does not raise, so that the
        state of the whole package is visible from one run.
    """
    manifest = load_document_manifest()
    documents = verification_documents(manifest)
    report_dir = data("report_output_dir")

    print("=" * RULE_WIDTH)
    print("Extract long identifier tokens from every Markdown source and look each")
    print("one up in the built PDF; no hand-written token list is used.")
    print("=" * RULE_WIDTH)

    total_missing = 0
    for _, stem in documents:
        source = source_path(report_dir, stem)
        pdf = pdf_path(report_dir, stem)
        if not (source.exists() and pdf.exists()):
            print(f"  !! {stem}: source or PDF missing at {report_dir}")
            continue
        tokens = extract_identifiers(source.read_text(encoding="utf-8"))
        pages, text = read_pdf_text(pdf)
        flattened = re.sub(r"\s", "", text)
        missing = [token for token in tokens if token not in flattened]
        tofu = text.count(REPLACEMENT_CHARACTER)
        total_missing += len(missing)
        flag = "OK " if (not missing and tofu == 0) else "!! "
        print(f"  {flag}{stem}")
        print(
            f"      {pages} pages | source tokens {len(tokens)} | "
            f"missing {len(missing)} | missing glyphs {tofu}"
        )
        for item in missing[:MISSING_LIST_LIMIT]:
            print(f"      x {item}")
    print(f"\nmissing identifiers over all documents: {total_missing}")


if __name__ == "__main__":
    main()
