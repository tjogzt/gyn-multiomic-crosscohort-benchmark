"""
Rebuild every report document and abort the run if any build fails.

Purpose
    Stage 10, eighth script: the build guard of the report documents. A pandoc
    run can fail -- a font that is not installed, a preamble that is missing, a
    source the typesetter chokes on -- while the PDF of an earlier successful run
    stays on disk and still looks current, so a stale document is silently
    published. This guard therefore rebuilds every Markdown source of the report
    directory unconditionally and exits with a non-zero status as soon as one
    build fails, which makes the failure impossible to mistake for a refresh.
    Before the rebuild it scans the sources for the two faults that break the
    typesetting of these documents (a CJK strikethrough, which crashes the soul
    package, and an over-long token that cannot be broken at a "/"), and after
    the rebuild it re-checks tokens, glyphs, page count and freshness.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    Every Markdown source in data("report_output_dir") -- the guard builds the
        whole directory, not a declared subset, because the point is that no
        document can be skipped and left stale.
    work("pdfbuild_header_tex") -- XeLaTeX preamble applied to every build.
    params:reports.pdf_engine, reports.page_geometry, reports.font_size,
    reports.toc_depth -- the pandoc invocation shared by every build here.
    params:reports.symbol_replacements -- emoji-to-symbol pairs applied to a
        source before pandoc sees it.
    params:reports.token_min_length -- minimum length of an identifier token
        extracted from a source for the post-build check.
    params:reports.unbreakable_token_length -- length above which a token that
        cannot be broken at a "/" risks being clipped.
    params:reports.strikethrough_max_span -- span of the strikethrough pattern
        used to find "~~...~~" deletions.
    params:reports.code_fence -- marker of a fenced code block, inside which a
        long token is typeset on its own line and cannot be broken.
    params:reports.markdown_extension, reports.pdf_extension -- file extensions
        of the sources and of the built PDFs.

Outputs
    data("report_output_dir")/<stem><pdf_extension> for every source in the
        directory -- the rebuilt PDFs, which replace any earlier build.
    work("pdfbuild_dir")/<stem><markdown_extension> -- the preprocessed copy of
        each source that was actually handed to pandoc.
    Exit status 1 when at least one build failed; 0 when all of them succeeded.

Usage
    python src/10_thresholds_gate_g3/10_08_guard_latex_builds.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import fitz  # PyMuPDF; used to read the text layer of a built PDF.

# Put src/ on the import path so that the module runs from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import data, ensure_dir, param, work  # noqa: E402

# Character class of an identifier token: the characters that make up an
# accession, a data-file name, a DOI or a path fragment. A whitespace-free run of
# these is what the typesetter cannot break, so it is what can be clipped and
# what the post-build check looks for.
IDENTIFIER_CHARACTERS = r"[A-Za-z0-9_.:/\-\[\]<>+=]"

# The CJK ideograph range. Used only to decide whether a strikethrough span
# contains CJK text: "~~...~~" around CJK crashes the soul LaTeX package that
# pandoc loads for strikethrough, whereas an ASCII-only deletion is rendered
# without trouble and is accepted.
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")

# U+FFFD REPLACEMENT CHARACTER: what the text layer carries for a character that
# was typeset as a box.
REPLACEMENT_CHARACTER = "\ufffd"

# Width of the rule printed between two steps of the guard.
RULE_WIDTH = 90

# Trailing lines and per-line length of a failed pandoc run that are echoed: the
# last lines of stderr name the failing file and character, and the full stream
# is far too long to print.
STDERR_TAIL_LINES = 6
STDERR_LINE_LIMIT = 150

# Number of missing tokens printed per document; the count above the list is the
# full count.
MISSING_LIST_LIMIT = 5


def report_document_stems() -> list[str]:
    """Return the file-name stem of every Markdown source of the report directory.

    Returns:
        The stems of the sources, sorted, which is the order the guard builds and
        reports them in. The set is discovered from the directory rather than
        declared, so that a newly added document is guarded without any other
        change.
    """
    extension = str(param("reports", "markdown_extension"))
    return sorted(
        entry.stem
        for entry in data("report_output_dir").iterdir()
        if entry.name.endswith(extension)
    )


def markdown_path(report_dir: Path, stem: str) -> Path:
    """Return the Markdown source path of one document.

    Args:
        report_dir: the directory declared as data:report_output_dir.
        stem: file-name stem of the document.

    Returns:
        The path of the source, whose modification time is the reference of the
        freshness check.
    """
    return report_dir / f"{stem}{param('reports', 'markdown_extension')}"


def pdf_path(report_dir: Path, stem: str) -> Path:
    """Return the PDF path of one document.

    Args:
        report_dir: the directory declared as data:report_output_dir.
        stem: file-name stem of the document.

    Returns:
        The path of the PDF that is rebuilt and afterwards checked.
    """
    return report_dir / f"{stem}{param('reports', 'pdf_extension')}"


def substitution_pairs() -> list[tuple[str, str]]:
    """Return the emoji-to-symbol substitutions, in their declared order.

    Returns:
        The (search, replacement) pairs of params:reports.symbol_replacements,
        whose order is significant and is the order of the configuration list.
    """
    return [(str(old), str(new)) for old, new in param("reports", "symbol_replacements")]


def apply_substitutions(text: str, pairs: list[tuple[str, str]]) -> str:
    """Replace the symbols the typeset fonts cannot carry.

    Args:
        text: the Markdown source as read from disk.
        pairs: the (search, replacement) pairs to apply, in order.

    Returns:
        The text with every emoji replaced by a plain symbol; nothing else is
        modified.
    """
    for old, new in pairs:
        text = text.replace(old, new)
    return text


def strikethrough_pattern() -> re.Pattern:
    """Return the pattern that finds a strikethrough span.

    Returns:
        A compiled pattern matching "~~...~~" over at most
        params:reports.strikethrough_max_span characters. The bound keeps the
        pattern from matching across a whole paragraph when a source has an
        unmatched "~~".
    """
    span = param("reports", "strikethrough_max_span")
    return re.compile(r"~~[^~]{0," + str(span) + r"}~~")


def cjk_strikethrough_spans(text: str) -> list[str]:
    """Return the strikethrough spans of a source that contain CJK text.

    Args:
        text: the Markdown source.

    Returns:
        The offending spans; an empty list means the source is safe to typeset.
    """
    return [hit for hit in strikethrough_pattern().findall(text) if CJK_PATTERN.search(hit)]


def token_pattern() -> re.Pattern:
    """Return the pattern that extracts identifier tokens from a source.

    Returns:
        A compiled pattern matching a whitespace-free run of identifier
        characters at least params:reports.token_min_length long. The bound is
        the same one the standalone verification pass of 10_07 uses, so that both
        scripts check the same tokens.

    Notes:
        This is deliberately a different bound from long_token_pattern(): almost
        every token is short enough never to be broken, and only the ones above
        the unbreakable bound are the ones at risk. Keeping the two apart means a
        clipped short identifier is still caught here.
    """
    length = param("reports", "token_min_length")
    return re.compile(IDENTIFIER_CHARACTERS + r"{" + str(length) + r",}")


def long_token_pattern() -> re.Pattern:
    """Return the pattern that finds a token that may be clipped.

    Returns:
        A compiled pattern matching a whitespace-free run of identifier
        characters at least params:reports.unbreakable_token_length long, i.e.
        one long enough for the typesetter to push a fragment off the page.
    """
    length = param("reports", "unbreakable_token_length")
    return re.compile(IDENTIFIER_CHARACTERS + r"{" + str(length) + r",}")


def code_block_contents(text: str) -> str:
    """Return the contents of every fenced code block of a source.

    Args:
        text: the Markdown source.

    Returns:
        The blocks joined into one string, which is what a token is looked up in
        to decide whether it is typeset on a line of its own. The fence marker
        comes from params:reports.code_fence and is escaped, so that a marker
        that happens to be a regex metacharacter still matches literally.
    """
    fence = re.escape(str(param("reports", "code_fence")))
    block = re.compile(fence + r"[^\n]*\n(.*?)\n" + fence, re.S)
    return "\n".join(set(block.findall(text)))


def longest_unbreakable_segment(token: str) -> str:
    """Return the part of a token the typesetter cannot break.

    Args:
        token: a long identifier token taken from a source.

    Returns:
        The longest fragment between two "/" characters. XeLaTeX may break a line
        at a "/", so only a fragment with no "/" in it can be pushed off the page;
        the longest such fragment is therefore the worst case of the token.
    """
    segments = [part for part in token.split("/") if part]
    if not segments:
        return token
    return max(segments, key=len)


def risky_tokens(text: str) -> list[str]:
    """Return the unbreakable fragments of a source that risk being clipped.

    Args:
        text: the Markdown source.

    Returns:
        One entry per offending fragment, over the whole source. A token that
        occurs inside a fenced code block is skipped: there it is typeset on a
        line of its own and cannot be broken, which is exactly the accepted
        remedy.
    """
    pattern = long_token_pattern()
    limit = param("reports", "unbreakable_token_length")
    inside_code = code_block_contents(text)
    found = []
    for token in sorted(set(pattern.findall(text))):
        if token in inside_code:
            continue
        fragment = longest_unbreakable_segment(token)
        if len(fragment) >= limit:
            found.append(fragment)
    return found


def build_document(stem: str, report_dir: Path, pairs: list[tuple[str, str]]) -> subprocess.CompletedProcess:
    """Stage one source and render it to PDF.

    Args:
        stem: file-name stem of the document.
        report_dir: the directory declared as data:report_output_dir.
        pairs: the emoji-to-symbol substitutions to apply.

    Returns:
        The finished pandoc process, whose return code decides whether the run
        may continue. Its stderr is echoed by the caller when it fails.
    """
    staged = work("pdfbuild_dir") / f"{stem}{param('reports', 'markdown_extension')}"
    text = apply_substitutions(
        markdown_path(report_dir, stem).read_text(encoding="utf-8"), pairs
    )
    staged.write_text(text, encoding="utf-8")
    output = pdf_path(report_dir, stem)
    return subprocess.run(
        [
            "pandoc",
            str(staged),
            "-o",
            str(output),
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
            f"--toc-depth={param('reports', 'toc_depth')}",
        ],
        capture_output=True,
        text=True,
    )


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


def main() -> None:
    """Run the two source scans, the full rebuild and the post-build check.

    Returns:
        None on success. The run exits with status 1 as soon as one build failed,
        and then no verification is printed: a failed rebuild means the PDFs on
        disk are a mixture of new and stale content and must not be treated as a
        refreshed package.
    """
    report_dir = data("report_output_dir")
    stems = report_document_stems()
    pairs = substitution_pairs()
    ensure_dir(work("pdfbuild_dir"))

    # Step 0: a strikethrough containing CJK crashes the soul package, which
    # pandoc loads whenever a source uses "~~...~~", so it is found before the
    # build rather than diagnosed from a LaTeX error afterwards.
    print("=" * RULE_WIDTH)
    print("[0] Strikethrough scan (~~...~~ around CJK crashes the soul package)")
    print("=" * RULE_WIDTH)
    bad_sources: list[str] = []
    for stem in stems:
        text = markdown_path(report_dir, stem).read_text(encoding="utf-8")
        spans = cjk_strikethrough_spans(text)
        if spans:
            bad_sources.append(stem)
            print(f"  !!! {stem}: {len(spans)} CJK strikethrough spans {spans[:2]}")
        else:
            hits = strikethrough_pattern().findall(text)
            if hits:
                print(f"  ok  {stem}: {len(hits)} ASCII-only strikethrough spans (acceptable)")
    print(f"  documents with CJK strikethrough: {bad_sources if bad_sources else 'none'}")

    # Step 0.5: a token long enough and unbreakable enough to be pushed off the
    # page is reported before the build, because the failure it causes is a
    # silently truncated identifier and not an error the typesetter raises.
    print("\n" + "=" * RULE_WIDTH)
    print("[0.5] Over-long token warning (xelatex breaks a line at '/', so only a")
    print("      fragment that offers no break can be clipped)")
    print("=" * RULE_WIDTH)
    long_hits: list[tuple[str, str]] = []
    for stem in stems:
        text = markdown_path(report_dir, stem).read_text(encoding="utf-8")
        for fragment in risky_tokens(text):
            long_hits.append((stem, fragment))
    if long_hits:
        for stem, fragment in long_hits:
            print(f"  !! {stem}: unbreakable fragment of {len(fragment)} characters -> {fragment[:60]}...")
        print(f"  -> {len(long_hits)} fragment(s) risk being clipped (move them into a code block)")
    else:
        print("  none (every token at or above the limit sits in a code block,")
        print("  or contains a '/' the typesetter may break at)")

    # Step 1: the unconditional rebuild. This is the guard proper: the failure of
    # one build is fatal, so that a PDF left over from an earlier run can never
    # be mistaken for a document that was refreshed by this run.
    print("\n" + "=" * RULE_WIDTH)
    print("[1] Full rebuild (a non-zero pandoc exit code aborts the run)")
    print("=" * RULE_WIDTH)
    failed: list[str] = []
    for stem in stems:
        process = build_document(stem, report_dir, pairs)
        if process.returncode != 0:
            failed.append(stem)
            print(f"  !!! {stem}: rc={process.returncode}")
            for line in process.stderr.strip().split("\n")[-STDERR_TAIL_LINES:]:
                print("        ", line[:STDERR_LINE_LIMIT])
        else:
            print(f"  OK  {stem}")

    if failed:
        print(f"\n{len(failed)} build(s) failed: {failed}")
        sys.exit(1)

    # Step 2: the post-build check. Tokens, glyphs and page count say whether the
    # rebuild produced a complete document; the freshness check says whether the
    # PDF on disk is the one this run produced, which is the property the whole
    # guard exists to protect.
    print("\n" + "=" * RULE_WIDTH)
    print("[2] Verification (long tokens / missing glyphs / page count / freshness)")
    print("=" * RULE_WIDTH)
    total_missing = 0
    token_re = token_pattern()
    for stem in stems:
        source = markdown_path(report_dir, stem)
        pdf = pdf_path(report_dir, stem)
        tokens = sorted(set(token_re.findall(source.read_text(encoding="utf-8"))))
        pages, text = read_pdf_text(pdf)
        flattened = re.sub(r"\s", "", text)
        missing = [token for token in tokens if token not in flattened]
        tofu = text.count(REPLACEMENT_CHARACTER)
        # A PDF older than its source was not rebuilt by this run.
        current = pdf.stat().st_mtime >= source.stat().st_mtime
        total_missing += len(missing)
        flag = "OK " if (not missing and tofu == 0 and current) else "!! "
        print(
            f"  {flag}{stem}: {pages} pages token={len(tokens)} missing={len(missing)} "
            f"missing_glyphs={tofu} freshness={'current' if current else 'STALE!'}"
        )
        for item in missing[:MISSING_LIST_LIMIT]:
            print("        x", item)
    print(f"\nmissing identifiers in total: {total_missing}")


if __name__ == "__main__":
    main()
