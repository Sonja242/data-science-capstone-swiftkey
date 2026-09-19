# Larger training data and continuation smoothing. Author: Sonja Sahebzad.
if (!file.exists("models/selected_predictor_v2.rds")) source("14_rebuild_prediction_v2.R")
library(data.table)
setDTthreads(8L)
source("R/predictive_model_v3.R")
paths <- setNames(file.path("data", "final", "en_US", paste0("en_US.",
  c("blogs", "news", "twitter"), ".txt")), c("blogs", "news", "twitter"))
stopifnot(all(file.exists(paths)))
if (file.exists("models/expanded_evaluation_v3.rds")) {
  previous_run <- readRDS("models/expanded_evaluation_v3.rds")
  if (!identical(unname(previous_run$corpus_md5), unname(tools::md5sum(paths))))
    stop("The corpus has changed. Preserve old artifacts and rebuild the v3 caches before proceeding.")
  if (!identical(unname(previous_run$code_md5), unname(tools::md5sum(names(previous_run$code_md5)))))
    stop("Model code has changed. Preserve old artifacts and rebuild the v3 caches before proceeding.")
  rm(previous_run)
}
dir.create("models", showWarnings = FALSE)
partition_path <- "models/expanded_partitions_v3.rds"
if (file.exists(partition_path)) rows <- readRDS(partition_path) else {
  rows <- expanded_partitions(paths)
  saveRDS(rows, partition_path, compress = "gzip")
}
partitions <- rows[, .(lines = .N), by = .(source, split)]
print(partitions)
validation <- prediction_cases(rows[split == "validation"], per_source = 600L, seed = 20261102L)
test <- prediction_cases(rows[split == "test"], per_source = 1000L, seed = 20261103L)
frequency <- NULL
all_results <- list()
best_score <- -Inf
best_model <- NULL
selection <- NULL
for (size in c("small", "large")) {
  counts_path <- paste0("models/expanded_counts_", size, "_v3.rds")
  if (file.exists(counts_path)) counts <- readRDS(counts_path) else {
    training <- if (size == "small") rows[split == "train" & small == TRUE, text] else rows[split == "train", text]
    message("Building ", size, " model from ", length(training), " training lines.")
    counts <- integer_ngram_counts(training)
    saveRDS(counts, counts_path, compress = "gzip")
    rm(training)
  }
  if (is.null(frequency)) {
    frequency <- data.table(word = counts$vocabulary[counts$tables[[1]]$w1], count = counts$tables[[1]]$count)
    # The same synthetic options are reused for every candidate and the baseline.
    validation_options <- make_choice_cases(validation, frequency, 20261104L)
    test_options <- make_choice_cases(test, frequency, 20261105L)
  }
  configs <- if (size == "small") data.table(method = "kneser_ney", order = c(3L,5L), discount = "0.75") else
    data.table(method = c("absolute","kneser_ney","kneser_ney","kneser_ney"),
      order = c(5L,3L,5L,5L), discount = c("0.75","0.75","0.75","estimated"))
  for (i in seq_len(nrow(configs))) {
    config <- configs[i]
    id <- paste(size, config$method, config$order, config$discount, sep = "_")
    message("Validation: ", id)
    d <- if (config$discount == "estimated") "estimated" else as.numeric(config$discount)
    model <- prepare_expanded_model(counts, maximum_order = config$order, method = config$method, discount = d)
    evaluation <- evaluate_expanded(model, validation, validation_options)
    result <- copy(evaluation$overall)
    result[, `:=`(id = id, training_lines = counts$training_lines, training_tokens = counts$training_tokens)]
    all_results[[length(all_results) + 1L]] <- result
    print(result)
    # Pick solely by free-text validation top-3 accuracy. Ties keep the earlier smaller candidate.
    if (result$top3 > best_score) {
      best_score <- result$top3
      best_model <- model
      selection <- result
    }
    rm(model)
    gc(FALSE)
  }
  rm(counts)
  gc(FALSE)
}
comparison <- rbindlist(all_results)
fwrite(comparison, "models/expanded_validation_v3.csv")
message("Selection fixed before computing test scores: ", selection$id)
saveRDS(best_model, "models/selected_predictor_v3.rds", compress = "gzip")
baseline <- readRDS("models/selected_predictor_v2.rds")
baseline_test <- evaluate_expanded(baseline, test, test_options, version = 2L)
selected_test <- evaluate_expanded(best_model, test, test_options)
set.seed(20261106L)
paired <- as.integer(selected_test$details$rank > 0L) - as.integer(baseline_test$details$rank > 0L)
ci <- quantile(replicate(2000L, mean(sample(paired, replace = TRUE))), c(.025,.975))
result <- list(partitions = partitions, validation = comparison, selection = selection,
  baseline_test = baseline_test, selected_test = selected_test, improvement_ci = unname(ci),
  choices_description = "Correct observed word plus three frequency-matched synthetic distractors (within 250 vocabulary ranks); not real quiz accuracy.",
  corpus_md5 = tools::md5sum(paths), code_md5 = tools::md5sum(c("R/predictive_model_v3.R", "R/predictive_model_v2.R")),
  seeds = 20261101:20261106, built_at = Sys.time(), environment = sessionInfo())
saveRDS(result, "models/expanded_evaluation_v3.rds", compress = "gzip")
fwrite(rbindlist(list(cbind(model = "v2 unchanged", baseline_test$overall),
                     cbind(model = selection$id, selected_test$overall))), "models/expanded_test_v3.csv")
fwrite(selected_test$by_source, "models/expanded_by_source_v3.csv")
print(selected_test$overall)
print(baseline_test$overall)
print(ci)
cat("Expanded experiment complete.\n")
