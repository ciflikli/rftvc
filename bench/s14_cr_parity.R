# S14 parity: randomForestSRC competing-risks forest on survival::pbc, fixed folds (bench/s14_cr_parity.py).
# Run: RL=<R lib with randomForestSRC> Rscript bench/s14_cr_parity.R <data.csv> <out.csv>
suppressMessages(library(randomForestSRC, lib.loc = Sys.getenv("RL")))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1])
times <- seq(365, 3650, by = 365)
feats <- setdiff(names(d), c("id", "time", "status", "fold"))
out <- list()
for (f in sort(unique(d$fold))) {
  train <- d[d$fold != f, c("time", "status", feats)]
  test <- d[d$fold == f, ]
  fit <- rfsrc(Surv(time, status) ~ ., data = train, ntree = 500, nodesize = 15, nsplit = 0,
               mtry = ceiling(sqrt(length(feats))), samptype = "swor",
               sampsize = function(x) x * 0.632, splitrule = "logrank", seed = -1)
  pr <- predict(fit, newdata = test[, c("time", "status", feats)])
  for (k in 1:2) {
    cif <- pr$cif[, , k]                 # n_test x length(time.interest)
    idx <- findInterval(times, pr$time.interest)
    vals <- sapply(idx, function(i) if (i == 0) rep(0, nrow(cif)) else cif[, i])
    out[[length(out) + 1]] <- data.frame(id = test$id, fold = f, cause = k, vals)
  }
}
res <- do.call(rbind, out)
names(res)[4:ncol(res)] <- paste0("t", times)
write.csv(res, args[2], row.names = FALSE)
