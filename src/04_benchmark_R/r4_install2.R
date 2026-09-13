# R 侧多组学方法安装（正式）
lg <- function(...) cat(format(Sys.time(), "[%H:%M:%S]"), ..., "\n")
options(timeout = 3600, Ncpus = max(1, parallel::detectCores() - 1))
lib <- .libPaths()[1]; lg("库:", lib)

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
  install.packages("BiocManager", lib = lib, quiet = TRUE)
}
options(BioC_mirror = "https://bioconductor.org")
r <- BiocManager::repositories(); r["CRAN"] <- "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"
options(repos = r)
lg("Bioc 版本:", as.character(BiocManager::version()))

have <- function(p) requireNamespace(p, quietly = TRUE)
ver  <- function(p) if (have(p)) as.character(packageVersion(p)) else "未装"

install_bioc <- function(pkg) {
  if (have(pkg)) { lg(sprintf("[跳过] %-13s %s", pkg, ver(pkg))); return(TRUE) }
  lg(sprintf("装 %-13s …", pkg))
  ok <- tryCatch({ BiocManager::install(pkg, lib = lib, ask = FALSE, update = FALSE, quiet = TRUE); have(pkg) },
                 error = function(e) { lg("  错误:", substr(conditionMessage(e), 1, 160)); FALSE })
  lg(sprintf("  %s %-13s %s", if (isTRUE(ok)) "✓" else "✗", pkg, ver(pkg))); isTRUE(ok)
}

# ---------- 1) intNMF：CRAN 已归档，改走 Archive 源码包 ----------
install_intnmf <- function() {
  if (have("intNMF")) { lg(sprintf("[跳过] intNMF %s", ver("intNMF"))); return(TRUE) }
  urls <- c(
    "https://cran.r-project.org/src/contrib/Archive/intNMF/",
    "https://mirrors.tuna.tsinghua.edu.cn/CRAN/src/contrib/Archive/intNMF/",
    "https://packagemanager.posit.co/cran/latest/src/contrib/Archive/intNMF/"
  )
  for (u in urls) {
    idx <- tryCatch(readLines(u, warn = FALSE), error = function(e) character(0))
    gz <- if (length(idx)) unique(regmatches(idx, regexpr("intNMF_[0-9.]+(?:-[0-9]+)?\\.tar\\.gz", idx))) else character(0)
    if (!length(gz)) { lg("  归档索引未取到:", u); next }
    gz <- gz[order(nchar(gz))][length(gz)]   # 取最新版本
    lg("  归档找到:", gz, "@", u)
    ok <- tryCatch({ install.packages(paste0(u, gz), lib = lib, repos = NULL, type = "source", quiet = TRUE); have("intNMF") },
                   error = function(e) { lg("  源码装错误:", substr(conditionMessage(e), 1, 160)); FALSE })
    if (isTRUE(ok)) { lg(sprintf("  ✓ intNMF %s", ver("intNMF"))); return(TRUE) }
  }
  lg("  ✗ intNMF 未能装上（该包已归档，缺依赖或已不可得）"); FALSE
}

LOG <- list()
lg("=== 1/5 intNMF (CRAN Archive) ===");        LOG[["intNMF"]]      <- install_intnmf()
lg("=== 2/5 MCIA (Bioc) ===");                  LOG[["MCIA"]]        <- install_bioc("MCIA")
lg("=== 3/5 mixOmics (Bioc) ===");              LOG[["mixOmics"]]    <- install_bioc("mixOmics")
lg("=== 4/5 iClusterPlus (Bioc) ===");          LOG[["iClusterPlus"]]<- install_bioc("iClusterPlus")
lg("=== 5/5 MOFA2 (Bioc，依赖重) ===");         LOG[["MOFA2"]]       <- install_bioc("MOFA2")

lg("=== 汇总 ===")
for (p in c("intNMF","SNFtool","MCIA","mixOmics","iClusterPlus","MOFA2"))
  lg(sprintf("  %-14s %-10s", p, ver(p)))
lg(sprintf("R 侧可用方法数: %d / 6",
           sum(sapply(c("intNMF","SNFtool","MCIA","mixOmics","iClusterPlus","MOFA2"), have))))
