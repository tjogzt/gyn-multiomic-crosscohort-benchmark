"""
Build the PDF document package from the Markdown report sources.

Purpose
    Stage 10, sixth script: the builder of the document package. It renders to
    PDF, with pandoc and XeLaTeX, every report document the manifest declares,
    and then checks each built PDF against the identifier tokens the manifest
    requires that document to retain. The check exists because two faults of
    this pipeline are otherwise silent: a glyph the typeset fonts do not carry,
    which pandoc mentions only on stderr as "no <glyph> (U+XXXX)" and which
    reaches the printed page as an empty box, and an identifier broken or
    dropped at line breaking, which leaves the PDF text missing a string the
    Markdown source contains. Both faults are counted per document.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    data("report_documents_manifest") -- JSON manifest of the document package:
        one entry per document giving its file-name stem and its description, in
        the declared build order, plus the identifier tokens that document must
        still contain after rendering.
    data("report_output_dir") -- directory holding the Markdown sources; a
        document is read from <stem><markdown_extension>.
    work("pdfbuild_header_tex") -- XeLaTeX preamble applied to every build: the
        CJK fonts, the symbol font that replaces the emoji of the Markdown
        sources, and the running head.
    params:reports.pdf_engine, reports.page_geometry, reports.font_size,
    reports.toc_depth -- the pandoc invocation shared by every build here.
    params:reports.symbol_replacements -- emoji-to-symbol pairs applied to a
        source before pandoc sees it, because the typeset fonts have no emoji
        glyph.
    params:reports.markdown_extension, reports.pdf_extension -- file extensions
        of the sources and of the built PDFs.

Outputs
    data("report_output_dir")/<stem><pdf_extension> for every document of the
        manifest -- the rendered report PDFs; the deliverable of the stage.
    work("pdfbuild_dir")/<stem><markdown_extension> -- the preprocessed copy of
        each source that was actually handed to pandoc, kept so that a rendering
        difference can be traced back to the exact text that produced it.

Usage
    python src/10_thresholds_gate_g3/10_06_build_document_package.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import fitz  # PyMuPDF; used to read the text layer of a built PDF.

# Put src/ on the import path so that the module runs from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import data, ensure_dir, param, work  # noqa: E402

# pandoc emits one line of this form per glyph the typeset font does not carry,
# for example "Missing character: There is no <glyph> (U+1F534) in font ...".
# Pulling the name and the codepoint out of stderr is what turns an otherwise
# silent typesetting fault into the count this module reports.
MISSING_GLYPH_PATTERN = re.compile(r"no (.+?) \(U\+([0-9A-F]+)\)")

# U+FFFD REPLACEMENT CHARACTER is what the text layer of a PDF carries for a
# character that was typeset as a box; counting it is the second glyph-level
# check, and it catches a font fault that pandoc did not report at all.
REPLACEMENT_CHARACTER = "\ufffd"

# Width of the rule printed between two passes of the run.
RULE_WIDTH = 78


def load_document_manifest() -> dict:
    """Load the JSON manifest of the document package.

    Returns:
        The parsed manifest, with its ``documents`` mapping in the declared
        build order, and its per-document token expectations.
    """
    with data("report_documents_manifest").open(encoding="utf-8") as handle:
        return json.load(handle)


def document_list(manifest: dict) -> list[tuple[str, str]]:
    """Return the documents of the manifest in their declared build order.

    Args:
        manifest: the parsed document manifest.

    Returns:
        A list of (document key, file-name stem) pairs. The manifest is ordered,
        so the list is both the build order and the report order.
    """
    return [(key, entry["file_stem"]) for key, entry in manifest["documents"].items()]


def substitution_pairs() -> list[tuple[str, str]]:
    """Return the emoji-to-symbol substitutions, in their declared order.

    Returns:
        The (search, replacement) pairs of params:reports.symbol_replacements.
        The order is significant and is the order of the configuration list.
    """
    return [(str(old), str(new)) for old, new in param("reports", "symbol_replacements")]


def apply_substitutions(text: str, pairs: list[tuple[str, str]]) -> str:
    """Replace the symbols the typeset fonts cannot carry.

    Args:
        text: the Markdown source as read from disk.
        pairs: the (search, replacement) pairs to apply, in order.

    Returns:
        The text with every emoji replaced by a plain symbol. The text is not
        otherwise modified: the substitutions are purely typographic, so that
        the built PDF differs from a naive build only where a glyph was missing.
    """
    for old, new in pairs:
        text = text.replace(old, new)
    return text


def source_path(report_dir: Path, stem: str) -> Path:
    """Return the Markdown source path of one document.

    Args:
        report_dir: the directory declared as data:report_output_dir.
        stem: file-name stem of the document, without extension.

    Returns:
        The path of the Markdown source.
    """
    return report_dir / f"{stem}{param('reports', 'markdown_extension')}"


def pdf_path(report_dir: Path, stem: str) -> Path:
    """Return the PDF path of one document.

    Args:
        report_dir: the directory declared as data:report_output_dir.
        stem: file-name stem of the document, without extension.

    Returns:
        The path of the PDF, the file that is built and afterwards checked.
    """
    return report_dir / f"{stem}{param('reports', 'pdf_extension')}"


def stage_source(source: Path, stem: str, pairs: list[tuple[str, str]]) -> Path:
    """Write the preprocessed copy of a source into the pdfbuild directory.

    The build never renders the source in place: pandoc is given a staged copy
    under work("pdfbuild_dir"), so that what was rendered is inspectable after
    the fact and the report directory only ever receives the PDF.

    Args:
        source: the Markdown source in the report directory.
        stem: file-name stem of the document.
        pairs: the emoji-to-symbol substitutions to apply.

    Returns:
        The path of the staged Markdown file.
    """
    staged = work("pdfbuild_dir") / f"{stem}{param('reports', 'markdown_extension')}"
    text = apply_substitutions(source.read_text(encoding="utf-8"), pairs)
    staged.write_text(text, encoding="utf-8")
    return staged


def run_pandoc(staged_markdown: Path, output_pdf: Path) -> subprocess.CompletedProcess:
    """Render one staged Markdown source to PDF.

    Args:
        staged_markdown: the preprocessed source handed to pandoc.
        output_pdf: path of the PDF to write.

    Returns:
        The finished process. Its return code decides success, and its stderr is
        where pandoc reports the glyphs the fonts could not typeset.
    """
    toc_depth = param("reports", "toc_depth")
    return subprocess.run(
        [
            "pandoc",
            str(staged_markdown),
            "-o",
            str(output_pdf),
            f"--pdf-engine={param('reports', 'pdf_engine')}",
            "-V",
            f"geometry:{param('reports', 'page_geometry')}",
            "-V",
            f"fontsize={param('reports', 'font_size')}",
            # The preamble supplies the CJK and symbol fonts, without which the
            # documents render as missing-glyph boxes.
            "-H",
            str(work("pdfbuild_header_tex")),
            "--toc",
            f"--toc-depth={toc_depth}",
        ],
        capture_output=True,
        text=True,
    )


def glyphs_missing_from(stderr: str) -> list[tuple[str, str]]:
    """Return the glyphs pandoc could not typeset.

    Args:
        stderr: the standard error of one pandoc run.

    Returns:
        The sorted set of (glyph name, hex codepoint) pairs pandoc reported. It
        is a set because the same glyph is reported once per occurrence.
    """
    return sorted(set(MISSING_GLYPH_PATTERN.findall(stderr)))


def read_pdf_text(path: Path):
    """Return the page count and the text layer of a PDF.

    Args:
        path: the PDF to read.

    Returns:
        A (page count, text) pair, the text being the concatenation of every
        page. The text layer is what an identifier check can be run against: a
        string that was dropped or broken at line breaking is absent from it.
    """
    with fitz.open(path) as document:
        # Indexed access rather than iteration: the pages are concatenated in
        # page order, which is the order the reader sees them in.
        pages = [str(document[index].get_text()) for index in range(document.page_count)]
        return document.page_count, "".join(pages)


def missing_identifiers(tokens: list[str], text: str) -> list[str]:
    """Return the tokens the PDF text does not contain.

    Args:
        tokens: the identifier tokens the document must retain.
        text: the extracted text of the built PDF.

    Returns:
        The tokens that are absent. Whitespace is removed from both sides before
        the comparison, because a line break inside a long identifier inserts
        one that is not part of the identifier itself.
    """
    flattened = re.sub(r"\s", "", text)
    return [token for token in tokens if token.replace(" ", "") not in flattened]


def main() -> None:
    """Build every document of the manifest and check each built PDF.

    Returns:
        None. The PDFs are written into the report directory; the run prints one
        line per document for the build pass and one for the check pass.
    """
    manifest = load_document_manifest()
    documents = document_list(manifest)
    report_dir = data("report_output_dir")
    pairs = substitution_pairs()
    ensure_dir(work("pdfbuild_dir"))

    print("=" * RULE_WIDTH)
    print("BUILD: render every report document to PDF")
    print("=" * RULE_WIDTH)
    built = 0
    for _, stem in documents:
        source = source_path(report_dir, stem)
        if not source.exists():
            print(f"  MISSING {stem}: no source at {source}")
            continue
        staged = stage_source(source, stem, pairs)
        output = pdf_path(report_dir, stem)
        process = run_pandoc(staged, output)
        missing_glyphs = glyphs_missing_from(process.stderr)
        if process.returncode == 0:
            built += 1
        print(
            f"  {stem}: rc={process.returncode} "
            f"missing_glyphs={len(missing_glyphs)} {missing_glyphs[:3]}"
        )
    print(f"\nbuilt {built}/{len(documents)}")

    # Second pass: the PDF that was just written is checked against the tokens
    # the manifest requires of it, so that a document cannot pass the build with
    # an identifier silently lost.
    print("\n" + "=" * RULE_WIDTH)
    print("VERIFY: missing identifiers / missing glyphs / page count")
    print("=" * RULE_WIDTH)
    all_missing = 0
    for key, stem in documents:
        path = pdf_path(report_dir, stem)
        if not path.exists():
            print(f"  {stem}: no PDF at {path}")
            continue
        pages, text = read_pdf_text(path)
        bad = missing_identifiers(manifest["required_tokens"].get(key, []), text)
        tofu = text.count(REPLACEMENT_CHARACTER)
        all_missing += len(bad)
        flag = "OK " if (not bad and tofu == 0) else "!! "
        print(
            f"  {flag}{stem}: {pages} pages "
            f"missing={len(bad)} missing_glyphs={tofu} {bad}"
        )
    print(f"\nmissing identifiers in total: {all_missing}")


if __name__ == "__main__":
    main()
