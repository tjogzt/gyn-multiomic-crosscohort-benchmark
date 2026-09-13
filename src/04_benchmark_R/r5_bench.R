#!/usr/bin/env Rscript
# R 侧多组学整合基准：与 Python 侧同协议、同数据缓存、同输出格式
# 方法: SNFtool / intNMF等价实现 / MCIA / mixOmics(block.pls) / iClusterPlus / MOFA2
suppressPackageStartupMessages({ library(jsonlite) })

H <- "/tmp/gyn_retyping/hcsv"
OUT <- "/tmp/gyn_retyping/bench_r_matrix.csv"
NGENE <- 1500; KGRID <- c(3,4,5); SEED <- 49
set.seed(SEED)

lg <- function(...) cat(format(Sys.time(), "[%H:%M:%S]"), ..., "\n")

# ---------- 工具 ----------
read_layer <- function(coh, lay) {
  p <- file.path(H, paste0(coh, "__", lay, ".csv.gz"))
  if (!file.exists(p)) return(NULL)
  M <- as.matrix(read.csv(gzfile(p), row.names = 1, check.names = FALSE))
  rownames(M) <- as.character(rownames(M)); colnames(M) <- as.character(colnames(M))
  M
}
zgene <- function(X) {
  m <- colMeans(X, na.rm = TRUE); s <- apply(X, 2, sd, na.rm = TRUE)
  s[!is.finite(s) | s < 1e-9] <- 1
  Z <- sweep(sweep(X, 2, m, "-"), 2, s, "/")
  Z[!is.finite(Z)] <- 0
  Z
}
topmad_k <- function(X, ng) {
  med <- apply(X, 2, median, na.rm = TRUE)
  mad <- apply(abs(sweep(X, 2, med, "-")), 2, median, na.rm = TRUE)
  mad[!is.finite(mad)] <- -1
  k <- order(-mad)[seq_len(min(ng, length(mad)))]
  k[mad[k] > 0]
}
ari <- function(a, b) {
  a <- as.integer(factor(a)); b <- as.integer(factor(b))
  n <- length(a); tab <- table(a, b)
  c2 <- function(x) sum(x * (x - 1) / 2)
  sij <- c2(tab); si <- c2(rowSums(tab)); sj <- c2(colSums(tab)); sn <- n * (n - 1) / 2
  exp_ <- si * sj / sn
  den <- 0.5 * (si + sj) - exp_
  if (den == 0) return(0)
  (sij - exp_) / den
}
kmeans_fit <- function(X, K, seed = SEED) kmeans(X, centers = K, nstart = 20, iter.max = 100, algorithm = "Lloyd")

transfer_ari <- function(Xa, Xb, K) {
  ka <- kmeans_fit(Xa, K); kb <- kmeans_fit(Xb, K)
  d <- as.matrix(dist(rbind(ka$centers, Xb)))[seq_len(K), -(seq_len(K)), drop = FALSE]
  pred <- apply(d, 2, which.min)
  list(ari = ari(kb$cluster, pred),
       deg = as.integer(length(unique(kb$cluster)) < 2 || length(unique(pred)) < 2))
}

# ---------- 方法：每个返回 样本×特征 矩阵 ----------
m_snf <- function(ls, K) {
  if (!requireNamespace("SNFtool", quietly = TRUE)) stop("SNFtool 未装")
  n <- nrow(ls[[1]]); if (n < 15) return(ls[[1]])
  kk <- max(3, min(20, n %/% 10))
  Wl <- lapply(ls, function(X) {
    X[!is.finite(X)] <- 0
    # 去掉零方差列，避免 dist 出现 NaN/Inf
    v <- apply(X, 2, var, na.rm = TRUE); v[!is.finite(v)] <- 0
    if (sum(v > 0) >= 2) X <- X[, v > 0, drop = FALSE]
    D <- as.matrix(dist(X)); D[!is.finite(D)] <- max(D[is.finite(D)], 1)
    mu <- median(D[D > 0]); if (!is.finite(mu) || mu <= 0) mu <- 1
    S <- exp(-D^2 / (2 * mu^2)); S[!is.finite(S)] <- 0
    K0 <- matrix(0, n, n)
    for (i in seq_len(n)) { idx <- order(-S[i, ])[seq_len(kk)]; K0[i, idx] <- S[i, idx] }
    K0 <- (K0 + t(K0)) / 2; rs <- rowSums(K0); rs[rs < 1e-12] <- 1; K0 / rs
  })
  Wf <- SNFtool::SNF(Wl, K = kk, t = 20)
  Wf[!is.finite(Wf)] <- 0; Wf[Wf < 0] <- 0
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
  # 共享样本因子 W 的交替最小二乘 mNMF（Chalise & Fridley 2016, intNMF）
  L <- length(ls); n <- nrow(ls[[1]]); k <- max(2, K)
  Xs <- lapply(ls, function(X) { Y <- X - min(X); Y[Y < 0] <- 0; Y + 1e-6 })
  set.seed(SEED); W <- matrix(runif(n * k), n, k)
  Hs <- lapply(Xs, function(X) matrix(runif(ncol(X) * k), ncol(X), k))
  for (it in seq_len(120)) {
    num <- matrix(0, n, k); den <- matrix(0, n, k)
    for (l in seq_len(L)) { Hl <- Hs[[l]]; num <- num + Xs[[l]] %*% Hl; den <- den + W %*% crossprod(Hl) }
    den[den < 1e-9] <- 1e-9
    W <- W * sqrt(pmax(num, 0) / den)
    W[!is.finite(W)] <- 0; W[W < 1e-9] <- 1e-9
    for (l in seq_len(L)) {
      Hl <- Hs[[l]]
      n2 <- crossprod(Xs[[l]], W)          # p_l x k
      d2 <- Hl %*% crossprod(W)            # (p_l x k)(k x k) —— 维度必须一致
      d2[d2 < 1e-9] <- 1e-9
      Hl <- Hl * sqrt(pmax(n2, 0) / d2); Hl[!is.finite(Hl)] <- 0; Hl[Hl < 1e-9] <- 1e-9
      Hs[[l]] <- Hl
    }
  }
  W
}
m_mcia_equiv <- function(ls, K) {
  # MFA/MCoA —— MCIA 的等价形式（Abdi 2010；MCIA 原文即 MFA 的 duality-diagram 表述）
  # 每层 PCA 后按 1/sqrt(第一特征值) 归一，再全局 PCA
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
  if (requireNamespace("MCIA", quietly = TRUE))
    return(as.matrix(MCIA::mcia(ls, nf = min(5, min(sapply(ls, nrow)) - 1), ncx = 1, cia = FALSE, verbose = FALSE)$mcoa$Tli))
  m_mcia_equiv(ls, K)
}
m_mixomics <- function(ls, K) {
  if (!requireNamespace("mixOmics", quietly = TRUE)) stop("mixOmics 未装")
  n <- nrow(ls[[1]]); nc <- min(5, n - 1, min(sapply(ls, ncol)))
  Xl <- lapply(seq_along(ls), function(i) {
    X <- ls[[i]]; colnames(X) <- paste0("b", i, "_g", seq_len(ncol(X))); X
  })
  names(Xl) <- paste0("block", seq_along(Xl))   # mixOmics 要求各块有唯一“块名”
  if (length(Xl) < 2) {
    # 单块子集：block.pls 会因 indY=1 退化 —— 改用 mixOmics 自身的 PCA（同族、单视图极限）
    return(as.matrix(mixOmics::pca(X = Xl[[1]], ncomp = nc, center = TRUE, scale = FALSE)$x))
  }
  # mixOmics >=6.32 的 block.pls 强制要求 Y 或 indY（以某块为结局块的多块框架）
  r <- mixOmics::block.pls(X = Xl, indY = 1, ncomp = nc, scale = FALSE, verbose = FALSE)
  do.call(cbind, lapply(r$variates, function(V) as.matrix(V[, seq_len(nc), drop = FALSE])))
}
m_icluster <- function(ls, K) {
  if (!requireNamespace("iClusterPlus", quietly = TRUE)) stop("iClusterPlus 未装")
  # iClusterPlus 要求各 dt 行数相同（行=样本），故不转置
  dts <- lapply(ls, function(X) { Z <- X; storage.mode(Z) <- "double"; Z })
  args <- c(list(K = K, maxiter = 30), setNames(dts, paste0("dt", seq_along(dts))))
  args$type <- rep("gaussian", length(dts))
  fit <- do.call(iClusterPlus::iClusterPlus, args)
  as.matrix(fit$meanZ)
}
m_mofa <- function(ls, K) {
  if (!requireNamespace("MOFA2", quietly = TRUE)) stop("MOFA2 未装")
  # 成本控制（如实记录为偏离）：每层按 MAD 确定性截到 <=400 特征，快速收敛模式
  CAP <- 400
  Xl <- lapply(ls, function(X) {
    if (ncol(X) > CAP) { med <- apply(X,2,median); md <- apply(abs(sweep(X,2,med,"-")),2,median); X <- X[, order(-md)[seq_len(CAP)], drop=FALSE] }
    t(X)
  })
  m <- MOFA2::create_mofa(Xl)
  dop <- MOFA2::get_default_data_options(m)
  mop <- MOFA2::get_default_model_options(m); mop$num_factors <- min(5, max(2, K))
  top <- MOFA2::get_default_training_options(m)
  top$verbose <- FALSE; top$seed <- SEED; top$maxiter <- 100; top$convergence_mode <- "fast"
  m <- MOFA2::prepare_mofa(m, data_options = dop, model_options = mop, training_options = top)
  f <- MOFA2::run_mofa(m, use_basilisk = TRUE)
  as.matrix(MOFA2::get_factors(f)[[1]])
}

METHODS <- list(
  "1_SNF"          = m_snf,
  "2_intNMF等价"    = m_intnmf_equiv,
  "3_MCIA等价(MFA)" = m_mcia,
  "4_mixOmics"     = m_mixomics,
  "5_iClusterPlus" = m_icluster,
  "6_MOFA2"        = m_mofa
)

# 显式声明库路径，避免两套 R 造成"包时有时无"
cat("R:", R.version.string, "| 库:", .libPaths()[1], "\n")
for (p in c("SNFtool","mixOmics","iClusterPlus","MOFA2")) {
  if (!requireNamespace(p, quietly = TRUE)) stop(sprintf("必装包 %s 在当前库不可见（请用 /opt/homebrew/bin/Rscript）", p))
}
cat("必需包可见性检查: 通过\n")

DOMAINS <- list(
  "A_CPTAC三队列" = list(cohorts = c("ind","dis","OV"),                layers = c("mRNA","CNA","protein")),
  "B_EC四层"      = list(cohorts = c("TCGA","ind","dis"),               layers = c("mRNA","miRNA","CNA","meth"))
)

rows <- list(); t0 <- Sys.time()
for (dname in names(DOMAINS)) {
  D <- DOMAINS[[dname]]; cohs <- D$cohorts; lays <- D$layers
  CACHE <- new.env()
  getL <- function(c, l) {
    key <- paste0(c, "|", l)
    if (!exists(key, envir = CACHE)) assign(key, read_layer(c, l), envir = CACHE)
    get(key, envir = CACHE)
  }
  lg("=== 域", dname, "===")
  subs <- unlist(lapply(seq_along(lays), function(r) combn(lays, r, simplify = FALSE)), recursive = FALSE)
  for (sub in subs) {
    sub <- as.character(unlist(sub))
    for (ai in seq_along(cohs)) for (bi in seq_along(cohs)) {
      if (bi <= ai) next
      a <- cohs[ai]; b <- cohs[bi]
      Ma <- lapply(sub, function(l) getL(a, l)); Mb <- lapply(sub, function(l) getL(b, l))
      if (any(sapply(Ma, is.null)) || any(sapply(Mb, is.null))) next
      sa <- Reduce(intersect, lapply(Ma, colnames)); sb <- Reduce(intersect, lapply(Mb, colnames))
      if (length(sa) < 40 || length(sb) < 40) next
      Asel <- list(); Bsel <- list(); ok <- TRUE
      for (li in seq_along(sub)) {
        gk <- intersect(rownames(Ma[[li]]), rownames(Mb[[li]]))
        if (length(gk) < 100) { ok <- FALSE; break }
        Xa <- zgene(t(Ma[[li]][gk, sa, drop = FALSE]))
        Xb <- zgene(t(Mb[[li]][gk, sb, drop = FALSE]))
        ka <- topmad_k(Xa, NGENE); kb <- topmad_k(Xb, NGENE)
        k <- sort(intersect(ka, kb)); if (length(k) < 50) k <- seq_len(min(NGENE, ncol(Xa)))
        Asel[[length(Asel)+1]] <- Xa[, k, drop = FALSE]; Bsel[[length(Bsel)+1]] <- Xb[, k, drop = FALSE]
      }
      if (!ok) next
      REF <- list()
      Bcat <- zgene(do.call(cbind, Bsel))
      for (K in KGRID) REF[[as.character(K)]] <- kmeans_fit(Bcat, K)$cluster
      for (mn in names(METHODS)) {
        rec <- data.frame(domain = dname, subset = paste(sub, collapse = "+"),
                          pair = paste0(a, "\u00d7", b), method = mn,
                          nA = nrow(Asel[[1]]), nB = nrow(Bsel[[1]]), stringsAsFactors = FALSE)
        Fa <- tryCatch(zgene(METHODS[[mn]](Asel, 4)), error = function(e) { rec$err <<- substr(conditionMessage(e),1,60); NULL })
        Fb <- tryCatch(zgene(METHODS[[mn]](Bsel, 4)), error = function(e) NULL)
        if (is.null(Fa) || is.null(Fb) || ncol(Fa) == 0 || ncol(Fb) == 0) { rec$err <- if (is.null(rec$err)) "dim/method" else rec$err; rows[[length(rows)+1]] <- rec; next }
        dmin <- min(ncol(Fa), ncol(Fb))          # 成分数不同时对齐到公共前缀
        if (dmin < 1) { rec$err <- "dim/method"; rows[[length(rows)+1]] <- rec; next }
        if (ncol(Fa) != ncol(Fb)) { Fa <- Fa[, seq_len(dmin), drop = FALSE]; Fb <- Fb[, seq_len(dmin), drop = FALSE] }
        for (K in KGRID) {
          r <- tryCatch(transfer_ari(Fa, Fb, K), error = function(e) list(ari = NA, deg = NA))
          rec[[paste0("ari_", K)]] <- round(r$ari, 4); rec[[paste0("deg_", K)]] <- r$deg
        }
        rows[[length(rows)+1]] <- rec
      }
    }
    lg(sprintf("  %s 完成 (%d 条, %.0fs)", paste(sub, collapse="+"), length(rows), as.numeric(difftime(Sys.time(), t0, units="secs"))))
    # 增量落盘
    df <- do.call(rbind, lapply(rows, function(x) { for (cc in c("ari_3","ari_4","ari_5","deg_3","deg_4","deg_5","err")) if (is.null(x[[cc]])) x[[cc]] <- NA; x }))
    write.csv(df, OUT, row.names = FALSE)
  }
}
lg("完成:", nrow(df), "条 →", OUT)
