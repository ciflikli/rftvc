args <- commandArgs(trailingOnly = TRUE)
arm <- args[[1]]
package <- if (arm == "ranger") "ranger" else "randomForestSRC"
if (!requireNamespace(package, quietly = TRUE)) {
  cat(paste0("R package ", package, " is unavailable\n"), file = stderr())
  quit(status = 2)
}
df <- read.csv(args[[2]], check.names = FALSE)
library(survival)
held_out <- df$id %% 5L == 0L
train <- df[!held_out, , drop = FALSE]
test <- df[held_out, , drop = FALSE]
n_trees <- as.integer(args[[3]])
n_threads <- as.integer(args[[4]])
Sys.setenv(OMP_NUM_THREADS = n_threads)
set.seed(0)
features <- grep("^x[0-9]+$", names(df), value = TRUE)
fit_data <- train[, c("stop", "event", features)]
fit_data$event <- as.logical(fit_data$event)
test_data <- test[seq_len(min(1000L, nrow(test))), , drop = FALSE]
if (arm == "ranger") {
  t0 <- proc.time()[["elapsed"]]
  forest <- ranger::ranger(
    Surv(stop, event) ~ ., data = fit_data,
    num.trees = n_trees, mtry = max(1L, floor(sqrt(length(features)))),
    min.bucket = 15L, replace = TRUE, sample.fraction = 0.632,
    num.threads = n_threads, write.forest = TRUE
  )
  fit_s <- proc.time()[["elapsed"]] - t0
  t0 <- proc.time()[["elapsed"]]
  pred <- rowSums(predict(forest, data = test_data[, features, drop = FALSE],
                          num.threads = n_threads)$chf)
  predict_s <- proc.time()[["elapsed"]] - t0
} else {
  t0 <- proc.time()[["elapsed"]]
  forest <- randomForestSRC::rfsrc(
    Surv(stop, event) ~ ., data = fit_data,
    ntree = n_trees, mtry = max(1L, floor(sqrt(length(features)))),
    nodesize = 15L, samptype = "swr", sampsize = max(1L, round(0.632 * nrow(train))),
    nsplit = 0L, importance = "none", perf.type = "none", ntime = 0L,
    forest = TRUE
  )
  fit_s <- proc.time()[["elapsed"]] - t0
  t0 <- proc.time()[["elapsed"]]
  pred <- predict(forest, newdata = test_data[, features, drop = FALSE])$predicted
  predict_s <- proc.time()[["elapsed"]] - t0
}
test_c <- survival::concordance(
  survival::Surv(test_data$stop, as.logical(test_data$event)) ~ pred,
  reverse = TRUE
)$concordance
cat(sprintf('{"fit_s":%.8f,"predict_1000_s":%.8f,"n_train":%d,"n_test":%d,"n_eval":%d,"test_c":%.8f,"package_version":"%s"}\n',
            fit_s, predict_s, nrow(train), nrow(test), nrow(test_data), test_c,
            as.character(utils::packageVersion(package))))
