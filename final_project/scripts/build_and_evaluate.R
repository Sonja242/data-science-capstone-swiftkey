# Run from the final_project folder. Author: Sonja Sahebzad.
library(data.table)
library(jsonlite)
setDTthreads(2L)
source("app/predictor.R")
project <- normalizePath("..", winslash = "/")
dir.create("results", showWarnings = FALSE)
private <- file.path(project, "data", "final_product")
dir.create(private, recursive = TRUE, showWarnings = FALSE)
if (file.exists("results/metrics.json")) stop("Final results already exist. Preserve this completed experiment.")

# Selection and reporting rules are fixed before any model comparison.
protocol <- list(version = "1.0", date = as.character(Sys.Date()), author = "Sonja Sahebzad",
  objective = "A CPU next-word product with measured quality, resource use and reproducible inference.",
  selection = "Highest development top-3 among models below 200 MiB in-memory and 50 MiB compressed; ties use top-1 then smaller size.",
  development = "Existing 600 validation examples, 200 per source. No quiz questions.",
  test = "900 previously unused original test lines, 300 per source; one random next-word position per line. Fixed seed 20260925.",
  metrics = "Top-1, top-3, vocabulary coverage, prediction availability, per-source results, Wilson 95% intervals, local CPU inference time and size.",
  confidence = "Model scores are not calibrated confidence. No confidence percentage will be displayed to users.",
  limitations = "Equal source weighting is an evaluation design, not an estimate of real user traffic. Archived web text may differ from current usage.")
write_json(protocol, "results/protocol.json", auto_unbox = TRUE, pretty = TRUE)

message("Reserve fresh final cases before tuning")
rows <- readRDS(file.path(project, "models/expanded_partitions_v3.rds"))
setDT(rows)
rows[, row_id := .I]
eligible <- rows[split == "test" & stringi::stri_count_fixed(text, " ") >= 2L]
set.seed(20261103L)
old_v3 <- eligible[, .SD[sample.int(.N, min(.N, 1000L))], by = source]
pool <- eligible[!row_id %in% old_v3$row_id]
hash_files <- c(
  "data/neural_evaluation/test.csv", "data/adaptation/test.csv",
  "data/learning_curve/final_test.csv", "models/learning_curve_reserved_case_hashes.csv",
  "models/word_generation_reserved_case_hashes.csv", "models/ranking_study_reserved_hashes.csv")
excluded <- unique(unlist(lapply(hash_files, function(f) {
  path <- file.path(project, f)
  if (!file.exists(path)) return(character())
  x <- fread(path)
  if (!"line_hash" %in% names(x)) stop("Missing hash in ", f)
  x$line_hash
})))
set.seed(20260925L)
reserved <- pool[, .SD[sample.int(.N, min(.N, 1200L))], by = source]
reserved[, line_hash := vapply(text, digest::digest, character(1), algo = "sha256", serialize = FALSE)]
reserved <- reserved[!line_hash %in% excluded, head(.SD, 300L), by = source]
stopifnot(nrow(reserved) == 900L, !anyDuplicated(reserved$text),
          !any(reserved$text %in% rows[split == "train", text]),
          !any(reserved$text %in% rows[split == "validation", text]))
set.seed(20260926L)
tokens <- strsplit(reserved$text, " ", fixed = TRUE)
positions <- vapply(tokens, function(w) sample(2:length(w), 1L), integer(1))
reserved[, prefix := mapply(function(w,k) paste(head(w,k-1L), collapse=" "), tokens, positions)]
reserved[, actual := mapply(function(w,k) w[k], tokens, positions)]
fwrite(reserved[, .(source,line_hash,prefix,actual)], file.path(private, "final_test.csv"))
fwrite(reserved[, .(source,line_hash)], "results/reserved_hashes.csv")
rm(rows, eligible, old_v3, pool, tokens); gc(FALSE)
dev <- fread(file.path(project, "data/adaptation/development.csv"))[, .(source,line_hash,prefix,actual)]
stopifnot(!any(dev$line_hash %in% reserved$line_hash), nrow(dev) == 600L)

message("Load the frozen trained n-gram model")
original <- readRDS(file.path(project, "models/selected_predictor_v3.rds"))
# This modest explicit output blocklist is documented, not a comprehensive safety classifier.
blocked <- c("fuck","fucking","fucked","fucker","fuckers","motherfucker","shit","shits","shitty",
 "bullshit","bitch","bitches","cunt","cunts","nigger","niggers","faggot","faggots","asshole","assholes",
 "cock","cocks","dick","dicks","pussy","porn","porno","pornography")

compact <- function(vocab_size, max_order, support, followers) {
  keep <- head(order(-original$base_probability, original$vocabulary)[
    !original$vocabulary[order(-original$base_probability, original$vocabulary)] %in% c("<unk>", blocked)], vocab_size)
  vocabulary <- original$vocabulary[keep]
  mapping <- rep.int(-1L, length(original$vocabulary) + 1L)
  mapping[1L] <- 0L
  mapping[keep + 1L] <- seq_along(keep)
  base <- original$base_probability[keep]
  base <- base / sum(base)
  tables <- vector("list", max_order)
  for (n in 2:max_order) {
    ctx <- paste0("w", seq_len(n - 1L))
    tab <- copy(original$tables[[n]][total >= support,
                  c(ctx, "word_id", "weight"), with = FALSE])
    for (col in c(ctx, "word_id")) set(tab, j = col, value = mapping[tab[[col]] + 1L])
    valid <- tab$word_id > 0L
    for (col in ctx) valid <- valid & tab[[col]] >= 0L
    tab <- tab[valid]
    setorderv(tab, c(ctx, "weight", "word_id"), c(rep(1L, length(ctx)), -1L, 1L))
    tab <- tab[, head(.SD, followers), by = ctx]
    tab[, backoff := pmax(0, 1 - sum(weight)), by = ctx]
    setkeyv(tab, ctx)
    tables[[n]] <- tab
    message("  ",n,"-grams retained: ",nrow(tab))
  }
  lookup <- data.table(word=vocabulary,id=seq_along(vocabulary)); setkey(lookup,word)
  list(version="shiny-1.0", vocabulary=vocabulary, lookup=lookup,
       base_probability=base, base_order=order(-base,vocabulary), tables=tables,
       maximum_order=max_order, support=support, followers=followers,
       training_lines=original$training_lines, training_tokens=original$training_tokens,
       method="Pruned interpolated Kneser-Ney", blocked_terms=blocked)
}

evaluate <- function(model, cases) {
  invisible(predict_word(model, "a simple example"))
  rbindlist(lapply(seq_len(nrow(cases)), function(i) {
    start <- as.numeric(Sys.time())
    p <- predict_word(model, cases$prefix[i])
    elapsed <- (as.numeric(Sys.time()) - start) * 1000
    data.table(source=cases$source[i], line_hash=cases$line_hash[i],
      rank=match(cases$actual[i], p$words, nomatch=0L),
      available=length(p$words)>0L, in_vocabulary=cases$actual[i] %in% model$vocabulary,
      elapsed_ms=elapsed, order=p$order)
  }))
}
configs <- data.table(name=c("Light","Balanced","Extended"),
  vocab=c(20000L,40000L,50000L), max_order=c(3L,5L,5L),
  support=c(20L,10L,5L), followers=c(5L,5L,8L))
comparisons <- list()
for (i in seq_len(nrow(configs))) {
  cfg <- configs[i]; message("Build ", cfg$name)
  model <- compact(cfg$vocab,cfg$max_order,cfg$support,cfg$followers)
  model_path <- file.path(private, paste0(cfg$name,".rds"))
  saveRDS(model, model_path, compress="xz")
  details <- evaluate(model,dev)
  comparisons[[i]] <- cbind(cfg, data.table(cases=nrow(details),
    top1=sum(details$rank==1L), top3=sum(details$rank>0L),
    object_mib=as.numeric(object.size(model))/1024^2,
    file_mib=file.info(model_path)$size/1024^2,
    median_ms=median(details$elapsed_ms), p95_ms=unname(quantile(details$elapsed_ms,.95))))
  fwrite(details,file.path(private,paste0(cfg$name,"_development.csv")))
  print(comparisons[[i]])
  rm(model); gc(FALSE)
}
comparison <- rbindlist(comparisons)
fwrite(comparison,"results/development_comparison.csv")
eligible <- comparison[object_mib < 200 & file_mib < 50]
stopifnot(nrow(eligible)>0L)
setorderv(eligible,c("top3","top1","object_mib"),c(-1L,-1L,1L))
winner <- eligible$name[1L]
file.copy(file.path(private,paste0(winner,".rds")),"app/model.rds",overwrite=FALSE)
stopifnot(file.exists("app/model.rds"))
write_json(list(selected=winner,model_md5=unname(tools::md5sum("app/model.rds")),
  selected_before_final_test=TRUE),"results/selection.json",pretty=TRUE,auto_unbox=TRUE)
rm(original); gc(FALSE)
load_seconds <- system.time(model <- readRDS("app/model.rds"))[[3L]]

message("Check exact sparse inference against dense distributions")
check_phrases <- c("", "xqzz", "the", "the weather is", "I\u2019m looking forward to", head(dev$prefix,95L))
for (phrase in check_phrases) {
  a <- predict_word(model,phrase)
  b <- predict_word(model,phrase,dense=TRUE)
  stopifnot(identical(a$words,b$words), max(abs(a$scores-b$scores))<1e-12,
            abs(sum(b$distribution)-1)<1e-10, !any(a$words %in% blocked))
}
message("Frozen winner: ",winner,". Evaluate reserved final cases once.")
details <- evaluate(model,reserved)
fwrite(details,"results/final_case_metrics.csv")
wilson <- function(k,n) {
  z <- qnorm(.975); p <- k/n; den <- 1+z*z/n
  mid <- (p+z*z/(2*n))/den; half <- z*sqrt(p*(1-p)/n+z*z/(4*n*n))/den
  c(low=mid-half,high=mid+half)
}
summarize <- function(d) {
  n <- nrow(d); t1 <- sum(d$rank==1L); t3 <- sum(d$rank>0L)
  data.table(cases=n, top1_count=t1,top1=t1/n,top1_low=wilson(t1,n)[1],top1_high=wilson(t1,n)[2],
    top3_count=t3,top3=t3/n,top3_low=wilson(t3,n)[1],top3_high=wilson(t3,n)[2],
    available=mean(d$available), vocabulary_coverage=mean(d$in_vocabulary),
    median_ms=median(d$elapsed_ms),p95_ms=unname(quantile(d$elapsed_ms,.95)),mean_ms=mean(d$elapsed_ms))
}
overall <- summarize(details)
by_source <- details[,summarize(.SD),by=source]
fwrite(by_source,"results/by_source.csv")
metadata <- list(title="Sonja Next Word",author="Sonja Sahebzad",brand="Sonja Projects",
  selected=winner,overall=as.list(overall[1]),by_source=by_source,
  vocabulary=length(model$vocabulary),max_order=model$maximum_order,
  model_mib=file.info("app/model.rds")$size/1024^2,
  memory_mib=as.numeric(object.size(model))/1024^2,load_seconds=load_seconds,
  training_lines=model$training_lines,training_tokens=model$training_tokens,
  model_md5=unname(tools::md5sum("app/model.rds")),
  test_seed=20260925L,position_seed=20260926L,development_cases=nrow(dev),
  confidence="Not calibrated; no confidence percentage is displayed.",
  date=as.character(Sys.Date()),R=R.version.string,platform=R.version$platform,
  cpu=Sys.getenv("PROCESSOR_IDENTIFIER"))
write_json(metadata,"results/metrics.json",pretty=TRUE,auto_unbox=TRUE,digits=10)
file.copy("results/metrics.json","app/metrics.json",overwrite=FALSE)
writeLines(capture.output(sessionInfo()),"results/session-info.txt")
fwrite(data.table(file=c("app/model.rds","app/predictor.R","scripts/build_and_evaluate.R"),
  md5=unname(tools::md5sum(c("app/model.rds","app/predictor.R","scripts/build_and_evaluate.R")))),
  "results/checksums.csv")
print(overall)
message("Completed. Final test results are frozen.")
