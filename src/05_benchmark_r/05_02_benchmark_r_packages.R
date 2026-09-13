# Benchmark six R multi-omic integration packages with the transfer-ARI protocol.
#
# Purpose
#     Run the R side of the integration benchmark with the same protocol, the same
#     cached layer matrices and the same output columns as the Python side. Each
#     method is applied to cohort A and cohort B of a pair, the A clustering is
#     transferred to B by nearest centroid, and the transfer is scored with the
#     adjusted Rand index against a reference partition of B derived from the raw
#     concatenated representation. It reads the compressed-CSV cache produced by
#     the second Python benchmark driver and writes one row per
#     (domain, layer subset, cohort pair, method) to the R results file.
#
# Author
#     Tao Zhu
# Created
#     2026-09-13
#
# Inputs
#     work("hcsv_dir")                Compressed-CSV layer matrices written by
#                                     04_02_benchmark_python_methods_extended.py.
#     config/params.yaml              seed, feature and clustering parameters, the
#                                     benchmark gates, the cohort and layer tags,
#                                     the evaluation domains and the R package list.
#     config/paths.yaml               the work root and the output file location.
#
# Outputs
#     work("benchmark_r_matrix")  CSV with columns domain, subset, pair, method,
#                                 nA, nB, ari_<K> and deg_<K> for every K.
#
# Usage
#     Rscript 05_02_benchmark_r_packages.R

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
# Separator inside a cohort-pair label, e.g. "ind\xd7dis". Declared in
# params.yaml because downstream stages split records on it; see the note there.
PAIR_SEPARATOR <- cfg_get(PARAMS, c("benchmark", "pair_separator"))
PATHS_RAW <- read_cfg(ROOT, "paths")
# paths.yaml resolves ${project_root} and ${data_root}; inject the repository
# root the same way src/common/config.py does before expanding the references.
PATHS_RAW$project_root <- ROOT
PATHS <- expand_refs(PATHS_RAW, PATHS_RAW)

WORK_ROOT <- as.character(PATHS$work_root)
HCSV_DIR <- file.path(WORK_ROOT, as.character(cfg_get(PATHS, c("work", "hcsv_dir"))))
OUT <- file.path(WORK_ROOT, as.character(cfg_get(PATHS, c("work", "benchmark_r_matrix"))))

SEED <- as.integer(cfg_get(PARAMS, c("seed")))
NGENE <- as.integer(cfg_get(PARAMS, c("features", "top_mad_genes")))
KGRID <- as.integer(unlist(cfg_get(PARAMS, c("clustering", "k_values"))))
K_PRIMARY <- as.integer(cfg_get(PARAMS, c("clustering", "k_primary")))
MIN_COMMON_GENES <- as.integer(cfg_get(PARAMS, c("benchmark", "min_common_genes")))
MIN_COMMON_MAD_GENES <- as.integer(cfg_get(PARAMS, c("benchmark", "min_common_mad_genes")))
MIN_SAMPLES <- as.integer(cfg_get(PARAMS, c("benchmark", "min_samples")))
MIN_SAMPLES_R <- as.integer(cfg_get(PARAMS, c("benchmark", "min_samples_r")))
# Per-layer feature cap applied before the MOFA2 fit, a recorded cost control.
MOFA_FEATURE_CAP <- as.integer(cfg_get(PARAMS, c("benchmark", "r_mofa_feature_cap")))

COH <- cfg_get(PARAMS, c("cohort_tags"))
LAY <- cfg_get(PARAMS, c("layer_tags"))
DOM <- cfg_get(PARAMS, c("domains"))
REQUIRED <- as.character(unlist(cfg_get(PARAMS, c("r_dependencies", "required"))))

dir.create(dirname(OUT), recursive = TRUE, showWarnings = FALSE)
set.seed(SEED)

suppressPackageStartupMessages({
  library(jsonlite)
})

lg <- function(...) cat(format(Sys.time(), "[%H:%M:%S]"), ..., "\n")


# --- Utilities ----------------------------------------------------------------


read_layer <- function(coh, lay) {
  # Read one compressed-CSV layer matrix, returning NULL when it is absent so the
  # caller can skip the combination instead of failing.
  p <- file.path(HCSV_DIR, paste0(coh, "__", lay, ".csv.gz"))
  if (!file.exists(p)) return(NULL)
  M <- as.matrix(read.csv(gzfile(p), row.names = 1, check.names = FALSE))
  rownames(M) <- as.character(rownames(M))
  colnames(M) <- as.character(colnames(M))
  M
}

zgene <- function(X) {
  # Standardise every column (gene) to mean 0 and unit standard deviation; a
  # degenerate column is protected by a unit denominator and non-finite entries
  # are set to 0.
  m <- colMeans(X, na.rm = TRUE)
  s <- apply(X, 2, sd, na.rm = TRUE)
  s[!is.finite(s) | s < 1e-9] <- 1
  Z <- sweep(sweep(X, 2, m, "-"), 2, s, "/")
  Z[!is.finite(Z)] <- 0
  Z
}

topmad_k <- function(X, ng) {
  # Column indices of the `ng` genes with the largest median absolute deviation,
  # restricted to those with strictly positive MAD.
  med <- apply(X, 2, median, na.rm = TRUE)
  mad <- apply(abs(sweep(X, 2, med, "-")), 2, median, na.rm = TRUE)
  mad[!is.finite(mad)] <- -1
  k <- order(-mad)[seq_len(min(ng, length(mad)))]
  k[mad[k] > 0]
}

ari <- function(a, b) {
  # Adjusted Rand index between two partitions, computed from the contingency
  # table so that no external package is needed.
  a <- as.integer(factor(a))
  b <- as.integer(factor(b))
  n <- length(a)
  tab <- table(a, b)
  c2 <- function(x) sum(x * (x - 1) / 2)
  sij <- c2(tab)
  si <- c2(rowSums(tab))
  sj <- c2(colSums(tab))
  sn <- n * (n - 1) / 2
  exp_ <- si * sj / sn
  den <- 0.5 * (si + sj) - exp_
  if (den == 0) return(0)
  (sij - exp_) / den
}

kmeans_fit <- function(X, K, seed = SEED) {
  # One shared KMeans configuration for every fit in the protocol.
  kmeans(X, centers = K, nstart = 20, iter.max = 100, algorithm = "Lloyd")
}

transfer_ari <- function(Xa, Xb, K) {
  # Cluster A, assign every B sample to its nearest A centroid, and score the
  # assignment against the clustering of B. A record is flagged degenerate when
  # either partition has fewer than two distinct labels.
  ka <- kmeans_fit(Xa, K)
  kb <- kmeans_fit(Xb, K)
  d <- as.matrix(dist(rbind(ka$centers, Xb)))[seq_len(K), -(seq_len(K)), drop = FALSE]
  pred <- apply(d, 2, which.min)
  list(
    ari = ari(kb$cluster, pred),
    deg = as.integer(length(unique(kb$cluster)) < 2 || length(unique(pred)) < 2)
  )
}


# --- Methods: each returns a sample x feature matrix --------------------------


m_snf <- function(ls, K) {
  # Similarity network fusion (Wang et al., Nature Methods 2014) followed by a
  # spectral embedding of the fused network, so no hard labels are returned.
  if (!requireNamespace("SNFtool", quietly = TRUE)) stop("SNFtool is not installed")
  n <- nrow(ls[[1]])
  if (n < MIN_SAMPLES) return(ls[[1]])
  kk <- max(3, min(20, n %/% 10))
  Wl <- lapply(ls, function(X) {
    X[!is.finite(X)] <- 0
    # Drop zero-variance columns: they make dist() return NaN/Inf.
    v <- apply(X, 2, var, na.rm = TRUE)
    v[!is.finite(v)] <- 0
    if (sum(v > 0) >= 2) X <- X[, v > 0, drop = FALSE]
    D <- as.matrix(dist(X))
    D[!is.finite(D)] <- max(D[is.finite(D)], 1)
    mu <- median(D[D > 0])
    if (!is.finite(mu) || mu <= 0) mu <- 1
    S <- exp(-D^2 / (2 * mu^2))
    S[!is.finite(S)] <- 0
    K0 <- matrix(0, n, n)
    for (i in seq_len(n)) {
      idx <- order(-S[i, ])[seq_len(kk)]
      K0[i, idx] <- S[i, idx]
    }
    K0 <- (K0 + t(K0)) / 2
    rs <- rowSums(K0)
    rs[rs < 1e-12] <- 1
    K0 / rs
  })
  Wf <- SNFtool::SNF(Wl, K = kk, t = 20)
  Wf[!is.finite(Wf)] <- 0
  Wf[Wf < 0] <- 0
  Wf <- (Wf + t(Wf)) / 2
  ev <- tryCatch(eigen(Wf, symmetric = TRUE), error = function(e) NULL)
  if (is.null(ev)) return(do.call(cbind, ls))
  d <- min(8, n - 2)
  val <- pmax(ev$values[seq_len(d)], 0)
  emb <- ev$vectors[, seq_len(d), drop = FALSE] * sqrt(val)
  emb[!is.finite(emb)] <- 0
  emb
}

m_intnmf_equiv <- function(ls, K) {
  # Alternating-least-squares multiple NMF sharing a sample factor W, the
  # equivalent implementation of intNMF (Chalise & Fridley 2016, intNMF).
  L <- length(ls)
  n <- nrow(ls[[1]])
  k <- max(2, K)
  Xs <- lapply(ls, function(X) {
    Y <- X - min(X)
    Y[Y < 0] <- 0
    Y + 1e-6
  })
  set.seed(SEED)
  W <- matrix(runif(n * k), n, k)
  Hs <- lapply(Xs, function(X) matrix(runif(ncol(X) * k), ncol(X), k))
  for (it in seq_len(120)) {
    num <- matrix(0, n, k)
    den <- matrix(0, n, k)
    for (l in seq_len(L)) {
      Hl <- Hs[[l]]
      num <- num + Xs[[l]] %*% Hl
      den <- den + W %*% crossprod(Hl)
    }
    den[den < 1e-9] <- 1e-9
    W <- W * sqrt(pmax(num, 0) / den)
    W[!is.finite(W)] <- 0
    W[W < 1e-9] <- 1e-9
    for (l in seq_len(L)) {
      Hl <- Hs[[l]]
      n2 <- crossprod(Xs[[l]], W) # p_l x k
      d2 <- Hl %*% crossprod(W) # (p_l x k)(k x k); the dimensions must agree
      d2[d2 < 1e-9] <- 1e-9
      Hl <- Hl * sqrt(pmax(n2, 0) / d2)
      Hl[!is.finite(Hl)] <- 0
      Hl[Hl < 1e-9] <- 1e-9
      Hs[[l]] <- Hl
    }
  }
  W
}

m_mcia_equiv <- function(ls, K) {
  # MFA/MCoA, the equivalent form of MCIA (Abdi 2010; the MCIA paper states its
  # duality diagram as MFA's). Each layer is PCA-reduced and rescaled by
  # 1/sqrt(first eigenvalue), then a global PCA is applied.
  Sc <- lapply(ls, function(X) {
    p <- prcomp(X, center = TRUE, scale. = FALSE)
    lam <- max(p$sdev[1]^2, 1e-9)
    p$x / sqrt(lam)
  })
  X <- do.call(cbind, Sc)
  P <- prcomp(X, center = TRUE, scale. = FALSE)
  P$x[, seq_len(min(5, ncol(P$x))), drop = FALSE]
}

m_mcia <- function(ls, K) {
  # Use the MCIA package when it is visible and fall back to the MFA equivalent.
  if (requireNamespace("MCIA", quietly = TRUE)) {
    return(as.matrix(MCIA::mcia(ls, nf = min(5, min(sapply(ls, nrow)) - 1),
                                ncx = 1, cia = FALSE, verbose = FALSE)$mcoa$Tli))
  }
  m_mcia_equiv(ls, K)
}

m_mixomics <- function(ls, K) {
  if (!requireNamespace("mixOmics", quietly = TRUE)) stop("mixOmics is not installed")
  n <- nrow(ls[[1]])
  nc <- min(5, n - 1, min(sapply(ls, ncol)))
  Xl <- lapply(seq_along(ls), function(i) {
    X <- ls[[i]]
    colnames(X) <- paste0("b", i, "_g", seq_len(ncol(X)))
    X
  })
  names(Xl) <- paste0("block", seq_along(Xl)) # mixOmics needs a unique block name
  if (length(Xl) < 2) {
    # A single-block subset makes block.pls degenerate (indY = 1); use the
    # single-view limit of the same family, mixOmics' own PCA.
    return(as.matrix(mixOmics::pca(X = Xl[[1]], ncomp = nc, center = TRUE, scale = FALSE)$x))
  }
  # mixOmics >= 6.32 requires Y or indY on block.pls: the multi-block framework
  # needs one block designated as the outcome block.
  r <- mixOmics::block.pls(X = Xl, indY = 1, ncomp = nc, scale = FALSE, verbose = FALSE)
  do.call(cbind, lapply(r$variates, function(V) as.matrix(V[, seq_len(nc), drop = FALSE])))
}

m_icluster <- function(ls, K) {
  if (!requireNamespace("iClusterPlus", quietly = TRUE)) stop("iClusterPlus is not installed")
  # iClusterPlus requires the same number of rows (samples) in every dt, so the
  # blocks are passed untransposed.
  dts <- lapply(ls, function(X) {
    Z <- X
    storage.mode(Z) <- "double"
    Z
  })
  args <- c(list(K = K, maxiter = 30), setNames(dts, paste0("dt", seq_along(dts))))
  args$type <- rep("gaussian", length(dts))
  fit <- do.call(iClusterPlus::iClusterPlus, args)
  as.matrix(fit$meanZ)
}

m_mofa <- function(ls, K) {
  if (!requireNamespace("MOFA2", quietly = TRUE)) stop("MOFA2 is not installed")
  # Cost control, recorded as a deviation: each layer is deterministically capped
  # at its top-MAD features and trained in fast-convergence mode.
  Xl <- lapply(ls, function(X) {
    if (ncol(X) > MOFA_FEATURE_CAP) {
      med <- apply(X, 2, median)
      md <- apply(abs(sweep(X, 2, med, "-")), 2, median)
      X <- X[, order(-md)[seq_len(MOFA_FEATURE_CAP)], drop = FALSE]
    }
    t(X)
  })
  m <- MOFA2::create_mofa(Xl)
  dop <- MOFA2::get_default_data_options(m)
  mop <- MOFA2::get_default_model_options(m)
  mop$num_factors <- min(5, max(2, K))
  top <- MOFA2::get_default_training_options(m)
  top$verbose <- FALSE
  top$seed <- SEED
  top$maxiter <- 100
  top$convergence_mode <- "fast"
  m <- MOFA2::prepare_mofa(m, data_options = dop, model_options = mop, training_options = top)
  f <- MOFA2::run_mofa(m, use_basilisk = TRUE)
  as.matrix(MOFA2::get_factors(f)[[1]])
}

METHODS <- list(
  "1_SNF" = m_snf,
  "2_intNMF_equiv" = m_intnmf_equiv,
  "3_MCIA_equiv(MFA)" = m_mcia,
  "4_mixOmics" = m_mixomics,
  "5_iClusterPlus" = m_icluster,
  "6_MOFA2" = m_mofa
)


# --- Environment guard --------------------------------------------------------
# Declare the library explicitly so that a second R installation cannot make a
# package appear or disappear between runs.

cat("R:", R.version.string, "| lib:", .libPaths()[1], "\n")
for (p in REQUIRED) {
  if (!requireNamespace(p, quietly = TRUE)) {
    stop(sprintf("required package %s is not visible in the active library", p))
  }
}
cat("Required package visibility check: passed\n")


# --- Evaluation domains -------------------------------------------------------
# Domain A: the three CPTAC cohorts on three layers.
# Domain B: TCGA-UCEC plus the two CPTAC-UCEC cohorts on four layers.

DOMAINS <- list()
DOMAINS[[DOM$A]] <- list(
  cohorts = c(COH$discovery, COH$independent, COH$ov),
  layers = c(LAY$mRNA, LAY$CNA, LAY$protein)
)
DOMAINS[[DOM$B]] <- list(
  cohorts = c(COH$tcga, COH$discovery, COH$independent),
  layers = c(LAY$mRNA, LAY$miRNA, LAY$CNA, LAY$methylation)
)

rows <- list()
t0 <- Sys.time()
for (dname in names(DOMAINS)) {
  D <- DOMAINS[[dname]]
  cohs <- D$cohorts
  lays <- D$layers
  CACHE <- new.env()
  getL <- function(c, l) {
    # Memoised layer lookup: each CSV is read at most once per domain.
    key <- paste0(c, "|", l)
    if (!exists(key, envir = CACHE)) assign(key, read_layer(c, l), envir = CACHE)
    get(key, envir = CACHE)
  }
  lg("=== domain", dname, "===")
  subs <- unlist(lapply(seq_along(lays), function(r) combn(lays, r, simplify = FALSE)),
                 recursive = FALSE)
  for (sub in subs) {
    sub <- as.character(unlist(sub))
    for (ai in seq_along(cohs)) {
      for (bi in seq_along(cohs)) {
        # Evaluate each unordered cohort pair once.
        if (bi <= ai) next
        a <- cohs[ai]
        b <- cohs[bi]
        Ma <- lapply(sub, function(l) getL(a, l))
        Mb <- lapply(sub, function(l) getL(b, l))
        if (any(sapply(Ma, is.null)) || any(sapply(Mb, is.null))) next
        # Sample intersection within each cohort, then the gene intersection
        # between the two cohorts, layer by layer.
        sa <- Reduce(intersect, lapply(Ma, colnames))
        sb <- Reduce(intersect, lapply(Mb, colnames))
        if (length(sa) < MIN_SAMPLES_R || length(sb) < MIN_SAMPLES_R) next
        Asel <- list()
        Bsel <- list()
        ok <- TRUE
        for (li in seq_along(sub)) {
          gk <- intersect(rownames(Ma[[li]]), rownames(Mb[[li]]))
          if (length(gk) < MIN_COMMON_GENES) {
            ok <- FALSE
            break
          }
          Xa <- zgene(t(Ma[[li]][gk, sa, drop = FALSE]))
          Xb <- zgene(t(Mb[[li]][gk, sb, drop = FALSE]))
          ka <- topmad_k(Xa, NGENE)
          kb <- topmad_k(Xb, NGENE)
          k <- sort(intersect(ka, kb))
          if (length(k) < MIN_COMMON_MAD_GENES) k <- seq_len(min(NGENE, ncol(Xa)))
          Asel[[length(Asel) + 1]] <- Xa[, k, drop = FALSE]
          Bsel[[length(Bsel) + 1]] <- Xb[, k, drop = FALSE]
        }
        if (!ok) next
        # Method-independent reference partition of B, from the raw concatenation.
        REF <- list()
        Bcat <- zgene(do.call(cbind, Bsel))
        for (K in KGRID) REF[[as.character(K)]] <- kmeans_fit(Bcat, K)$cluster
        for (mn in names(METHODS)) {
          rec <- data.frame(domain = dname, subset = paste(sub, collapse = "+"),
                            pair = paste0(a, PAIR_SEPARATOR, b), method = mn,
                            nA = nrow(Asel[[1]]), nB = nrow(Bsel[[1]]),
                            stringsAsFactors = FALSE)
          # run the method on both cohorts; a failure is recorded, not raised
          Fa <- tryCatch(zgene(METHODS[[mn]](Asel, K_PRIMARY)), error = function(e) {
            rec$err <<- substr(conditionMessage(e), 1, 60)
            NULL
          })
          Fb <- tryCatch(zgene(METHODS[[mn]](Bsel, K_PRIMARY)), error = function(e) NULL)
          if (is.null(Fa) || is.null(Fb) || ncol(Fa) == 0 || ncol(Fb) == 0) {
            rec$err <- if (is.null(rec$err)) "dim/method" else rec$err
            rows[[length(rows) + 1]] <- rec
            next
          }
          dmin <- min(ncol(Fa), ncol(Fb)) # align to the common component prefix
          if (dmin < 1) {
            rec$err <- "dim/method"
            rows[[length(rows) + 1]] <- rec
            next
          }
          if (ncol(Fa) != ncol(Fb)) {
            Fa <- Fa[, seq_len(dmin), drop = FALSE]
            Fb <- Fb[, seq_len(dmin), drop = FALSE]
          }
          for (K in KGRID) {
            r <- tryCatch(transfer_ari(Fa, Fb, K),
                          error = function(e) list(ari = NA, deg = NA))
            rec[[paste0("ari_", K)]] <- round(r$ari, 4)
            rec[[paste0("deg_", K)]] <- r$deg
          }
          rows[[length(rows) + 1]] <- rec
        }
      }
    }
    lg(sprintf("  %s done (%d rows, %.0fs)", paste(sub, collapse = "+"), length(rows),
               as.numeric(difftime(Sys.time(), t0, units = "secs"))))
    # Incremental write: fill missing columns so rbind succeeds, then persist.
    fill_cols <- c(paste0("ari_", KGRID), paste0("deg_", KGRID), "err")
    df <- do.call(rbind, lapply(rows, function(x) {
      for (cc in fill_cols) if (is.null(x[[cc]])) x[[cc]] <- NA
      x
    }))
    write.csv(df, OUT, row.names = FALSE)
  }
}
lg("done:", nrow(df), "rows ->", OUT)
