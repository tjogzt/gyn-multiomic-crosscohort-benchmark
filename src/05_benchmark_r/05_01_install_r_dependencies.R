# Install the R integration packages required by the R benchmark.
#
# Purpose
#     Install every R package the R-side integration benchmark imports, from the
#     repositories declared in config/params.yaml. Bioconductor packages are
#     installed through BiocManager; a package that CRAN has archived (currently
#     intNMF) is built from its CRAN Archive source tarball; packages whose build
#     is heavy are only verified. The script records the result per package and
#     prints a summary of which versions are available, so the benchmark can
#     report how many of its method arms were actually runnable.
#
# Author
#     Tao Zhu
# Created
#     2026-09-13
#
# Inputs
#     config/params.yaml  r_repositories (mirrors and archive roots) and
#                         r_dependencies (package groups by source).
#     config/paths.yaml   loaded through the same helper so that both scripts
#                         share one configuration entry point.
#
# Outputs
#     None. Packages are installed into the active R library; the console reports
#     one line per attempt and a final availability summary.
#
# Usage
#     Rscript 05_01_install_r_dependencies.R

# --- Configuration access -----------------------------------------------------
# The helper resolves the repository root from the running script, reads the YAML
# configuration with the yaml package and, when yaml is unavailable, falls back
# to a JSON export of the same file (config/<stem>.json).

find_repo_root <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  hit <- grep("^--file=", args, value = TRUE)
  start <- if (length(hit) > 0) {
    dirname(normalizePath(sub("^--file=", "", hit[1])))
  } else {
    normalizePath(getwd())
  }
  d <- start
  for (i in seq_len(12)) {
    if (file.exists(file.path(d, "config", "params.yaml"))) return(d)
    parent <- dirname(d)
    if (identical(parent, d)) break
    d <- parent
  }
  stop("Cannot locate the repository root: no config/params.yaml above ", start)
}

read_cfg <- function(root, stem) {
  yaml_path <- file.path(root, "config", paste0(stem, ".yaml"))
  if (requireNamespace("yaml", quietly = TRUE)) {
    return(yaml::read_yaml(yaml_path))
  }
  json_path <- file.path(root, "config", paste0(stem, ".json"))
  if (file.exists(json_path) && requireNamespace("jsonlite", quietly = TRUE)) {
    return(jsonlite::fromJSON(json_path, simplifyVector = TRUE))
  }
  stop("Neither the yaml package nor a JSON export (", json_path,
       ") is available to read the configuration.")
}

# Return the first value stored under `name` anywhere in the tree.
find_leaf <- function(tree, name) {
  for (key in names(tree)) {
    if (identical(key, name)) return(tree[[key]])
    if (is.list(tree[[key]])) {
      hit <- find_leaf(tree[[key]], name)
      if (!is.null(hit)) return(hit)
    }
  }
  NULL
}

# Resolve a ${name}, ${env:NAME} or ${env:NAME:-default} specification.
resolve_spec <- function(spec, tree, depth) {
  if (startsWith(spec, "env:")) {
    body <- substring(spec, 5L)
    if (grepl(":-", body, fixed = TRUE)) {
      parts <- strsplit(body, ":-", fixed = TRUE)[[1]]
      name <- parts[1]
      default <- paste(parts[-1], collapse = ":-")
      val <- Sys.getenv(name, unset = NA_character_)
      if (!is.na(val)) return(val)
      return(expand_refs(default, tree, depth))
    }
    val <- Sys.getenv(body, unset = NA_character_)
    if (is.na(val)) {
      stop("Environment variable '", body,
           "' referenced by the configuration is not set and has no default.")
    }
    return(val)
  }
  found <- find_leaf(tree, spec)
  if (is.null(found)) stop("Configuration reference '", spec, "' not found.")
  as.character(expand_refs(found, tree, depth))
}

# Expand every ${...} inside a configuration value, matching nested braces.
expand_refs <- function(value, tree, depth = 0L) {
  if (depth > 12L) stop("Configuration ${...} references form a cycle.")
  if (is.list(value)) {
    return(lapply(value, expand_refs, tree = tree, depth = depth + 1L))
  }
  if (!is.character(value)) return(value)
  vapply(value, function(txt) {
    out <- ""
    i <- 1L
    n <- nchar(txt)
    while (i <= n) {
      if (substr(txt, i, i + 1L) == "${") {
        j <- i + 2L
        level <- 1L
        while (j <= n && level > 0L) {
          if (substr(txt, j, j + 1L) == "${") {
            level <- level + 1L
            j <- j + 2L
            next
          }
          if (substr(txt, j, j) == "}") level <- level - 1L
          j <- j + 1L
        }
        if (level != 0L) stop("Unbalanced ${...} in configuration: ", txt)
        out <- paste0(out, resolve_spec(substr(txt, i + 2L, j - 2L), tree, depth + 1L))
        i <- j
      } else {
        out <- paste0(out, substr(txt, i, i))
        i <- i + 1L
      }
    }
    out
  }, character(1), USE.NAMES = FALSE)
}

# Walk a dotted key path into a configuration section.
cfg_get <- function(tree, keys) {
  node <- tree
  for (key in keys) {
    if (!is.list(node) || is.null(node[[key]])) {
      stop("No configuration entry ", paste(keys, collapse = "."))
    }
    node <- node[[key]]
  }
  node
}

ROOT <- find_repo_root()
PARAMS <- read_cfg(ROOT, "params")
# paths.yaml is read for a single, shared configuration entry point; this script
# needs no filesystem location from it.
PATHS_RAW <- read_cfg(ROOT, "paths")
PATHS_RAW$project_root <- ROOT
invisible(expand_refs(PATHS_RAW, PATHS_RAW))

REPOS <- cfg_get(PARAMS, c("r_repositories"))
DEPS <- cfg_get(PARAMS, c("r_dependencies"))
CRAN_MIRROR <- as.character(REPOS$cran)
BIOC_MIRROR <- as.character(REPOS$bioconductor)
ARCHIVE_ROOTS <- as.character(unlist(REPOS$archive_roots))
BIOC_PKGS <- as.character(unlist(DEPS$bioc))
ARCHIVE_PKGS <- as.character(unlist(DEPS$archive))
PREINSTALLED_PKGS <- as.character(unlist(DEPS$preinstalled))
# Summary order: archive packages, then the packages verified but not installed,
# then the Bioconductor packages.
ALL_PKGS <- c(ARCHIVE_PKGS, PREINSTALLED_PKGS, BIOC_PKGS)

# --- Timing and library setup -------------------------------------------------

lg <- function(...) cat(format(Sys.time(), "[%H:%M:%S]"), ..., "\n")

# A generous download timeout is set because Bioconductor source builds over a
# mirror can be slow; the parallel build count leaves one core for the system.
options(timeout = 3600, Ncpus = max(1, parallel::detectCores() - 1))
lib <- .libPaths()[1]
lg("library:", lib)

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  options(repos = c(CRAN = CRAN_MIRROR))
  install.packages("BiocManager", lib = lib, quiet = TRUE)
}
options(BioC_mirror = BIOC_MIRROR)
repos <- BiocManager::repositories()
repos["CRAN"] <- CRAN_MIRROR
options(repos = repos)
lg("Bioconductor version:", as.character(BiocManager::version()))


# --- Package helpers ----------------------------------------------------------

have <- function(p) requireNamespace(p, quietly = TRUE)
ver <- function(p) if (have(p)) as.character(packageVersion(p)) else "not installed"

# Install one package by name, taking the package name from the caller's list.
install_bioc <- function(pkg) {
  if (have(pkg)) {
    lg(sprintf("[skip] %-13s %s", pkg, ver(pkg)))
    return(TRUE)
  }
  lg(sprintf("installing %-13s ...", pkg))
  ok <- tryCatch(
    {
      BiocManager::install(pkg, lib = lib, ask = FALSE, update = FALSE, quiet = TRUE)
      have(pkg)
    },
    error = function(e) {
      lg("  error:", substr(conditionMessage(e), 1, 160))
      FALSE
    }
  )
  lg(sprintf("  %s %-13s %s", if (isTRUE(ok)) "[OK]" else "[FAIL]", pkg, ver(pkg)))
  isTRUE(ok)
}

install_archive <- function(pkg) {
  # Build a package CRAN has archived from its source tarball under an Archive
  # index. Each configured root is tried in turn because the mirrors do not all
  # carry the same Archive snapshot.
  if (have(pkg)) {
    lg(sprintf("[skip] %s %s", pkg, ver(pkg)))
    return(TRUE)
  }
  pattern <- paste0(pkg, "_[0-9.]+(?:-[0-9]+)?\\.tar\\.gz")
  index_urls <- paste0(ARCHIVE_ROOTS, "/", pkg, "/")
  for (u in index_urls) {
    idx <- tryCatch(readLines(u, warn = FALSE), error = function(e) character(0))
    gz <- if (length(idx)) {
      unique(regmatches(idx, regexpr(pattern, idx)))
    } else {
      character(0)
    }
    if (!length(gz)) {
      lg("  archive index unavailable:", u)
      next
    }
    # The shortest name is the newest version, since the version string is the
    # only variable-length part of the file name.
    gz <- gz[order(nchar(gz))][length(gz)]
    lg("  archive found:", gz, "@", u)
    ok <- tryCatch(
      {
        install.packages(paste0(u, gz), lib = lib, repos = NULL, type = "source", quiet = TRUE)
        have(pkg)
      },
      error = function(e) {
        lg("  source install error:", substr(conditionMessage(e), 1, 160))
        FALSE
      }
    )
    if (isTRUE(ok)) {
      lg(sprintf("  [OK] %s %s", pkg, ver(pkg)))
      return(TRUE)
    }
  }
  lg(sprintf("  [FAIL] %s could not be installed (archived, missing dependency or unavailable)", pkg))
  FALSE
}


# --- Install every method arm -------------------------------------------------

LOG <- list()
n_total <- length(ARCHIVE_PKGS) + length(BIOC_PKGS)
step <- 0L
for (pkg in ARCHIVE_PKGS) {
  step <- step + 1L
  lg(sprintf("=== %d/%d %s (CRAN Archive) ===", step, n_total, pkg))
  LOG[[pkg]] <- install_archive(pkg)
}
for (pkg in BIOC_PKGS) {
  step <- step + 1L
  lg(sprintf("=== %d/%d %s (Bioconductor) ===", step, n_total, pkg))
  LOG[[pkg]] <- install_bioc(pkg)
}

lg("=== summary ===")
for (p in ALL_PKGS) lg(sprintf("  %-14s %-14s", p, ver(p)))
lg(sprintf("R-side method arms available: %d / %d",
           sum(vapply(ALL_PKGS, have, logical(1))), length(ALL_PKGS)))
