# randomForestSRC check (2026-09-26): start-stop rows with competing risks. Run: RL=<R lib with randomForestSRC> Rscript bench/cr_rfsrc_check.R
suppressMessages(library(randomForestSRC, lib.loc=Sys.getenv("RL"))); set.seed(1)
n <- 200; rows <- do.call(rbind, lapply(1:n, function(i){ k <- sample(2:4,1); st <- cumsum(c(0, runif(k,0.5,1.5)))
 data.frame(id=i, start=st[-(k+1)], stop=st[-1], x1=rnorm(k), x2=rnorm(1), event=c(rep(0,k-1), sample(0:2,1)))}))
for (ev in list(rows$event, as.integer(rows$event>0))) { d <- rows; d$event <- ev
 cat("-- causes:", sort(unique(ev)), " splitrule=random\n")
 tryCatch({f <- rfsrc(Surv(id, start, stop, event) ~ ., d, ntree=20, splitrule="random"); cat("family", f$family, "\n"); print(names(f)[grepl("cif|chf|surv", names(f))]); if(!is.null(f$cif)) print(dim(f$cif))}, error=function(e) cat("ERROR:", conditionMessage(e), "\n")) }
print(body(randomForestSRC:::get.grow.event.info))
