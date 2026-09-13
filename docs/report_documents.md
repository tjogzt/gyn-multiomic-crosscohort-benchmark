# Report documents

The document package of stage 10 (`10_06`, `10_07`, `10_08`) renders the project's
report documents to PDF with pandoc and then checks each built PDF against its
Markdown source. The documents themselves are not part of this repository: they
are written in Chinese and belong to the direction-F workspace, which is declared
as `paths.data.report_output_dir` (default `${project_root}/../retyping`, and
redirectable with `GYN_PREREGISTRATION_DIR`).

## The manifest

`report_documents.json`, at the repository root and declared as
`paths.data.report_documents_manifest`, is the data file of the package. It holds:

- `documents` — one entry per report document, keyed by an English document key.
  Each entry gives the `file_stem` (the file name of the Markdown source and of
  the PDF, without extension) and an English `description`. **The declared order
  is the build order** of the document package.
- `verification_documents` — the document keys that `10_07` verifies on their own
  by re-extracting long identifier tokens from the Markdown source, instead of
  comparing against a written-down list.
- `required_tokens` — the identifier tokens a document must still contain after
  rendering, keyed by document key. A token that is missing from the PDF text was
  broken or dropped at line breaking, which is the failure `10_06` reports.

### Why the titles are escaped

The `file_stem` values are the exact file names on disk and are not ASCII.
`docs/coding_standard.md` keeps every file of the repository ASCII, so each of
them is written as a JSON `\uXXXX` escape; `json.load` decodes them back to the
exact names. Nothing else in the repository carries a document title, so
translating or renaming a document means editing this one data file and the
documents themselves.

## Rendering settings

The rendering settings are parameters, not data, and live in
`config/params.yaml` under `reports`: the pandoc PDF engine, the page geometry,
the base type size, the table-of-contents depth, the token-length thresholds of
the two verification passes, and the extension of the sources and of the PDFs.
