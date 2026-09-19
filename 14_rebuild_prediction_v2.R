# Run from the Data Science Capstone project. Author: Sonja Sahebzad.
library(data.table)
source("R/predictive_model_v2.R")
data.table::setDTthreads(2L)
dir.create("models", showWarnings = FALSE)
paths <- setNames(file.path("data", "final", "en_US", paste0("en_US.",
  c("blogs", "news", "twitter"), ".txt")), c("blogs", "news", "twitter"))
stopifnot(all(file.exists(paths)))
metadata <- list(version = 2L, probability = .02, seed = 20260924L,
                 sample_function = paste(deparse(body(sample_prediction_data)), collapse = "\n"),
                 sizes = file.info(paths)$size, md5 = unname(tools::md5sum(paths)))
artifact_path <- "models/prediction_counts_v2.rds"
if (file.exists(artifact_path)) {
  artifacts <- readRDS(artifact_path)
  if (!identical(artifacts$metadata, metadata)) stop("Cached inputs changed. Preserve and rebuild the v2 cache.")
} else {
  dataset <- sample_prediction_data(paths)
  train <- dataset[split == "train"]
  validation <- dataset[split == "validation"]
  test <- dataset[split == "test"]
  stopifnot(!any(train$text %in% validation$text), !any(train$text %in% test$text),
            !any(validation$text %in% test$text))
  artifacts <- list(metadata = metadata, counts = build_prediction_counts(train$text),
    partitions = dataset[, .(lines = .N), by = .(source, split)],
    validation = prediction_cases(validation, seed = 20261001L),
    test = prediction_cases(test, seed = 20261002L))
  saveRDS(artifacts, artifact_path, compress = "gzip")
}
configurations <- data.table(
  id = c("unigram", "legacy_2", "interpolation_1", "interpolation_2", "witten_bell_1", "witten_bell_2"),
  method = c("unigram", "legacy", "interpolated", "interpolated", "witten_bell", "witten_bell"),
  threshold = c(1L, 2L, 1L, 2L, 1L, 2L))
evaluations <- lapply(seq_len(nrow(configurations)), function(i) {
  config <- configurations[i]
  message("Validation: ", config$id)
  model <- prepare_prediction_model(artifacts$counts, config$method, config$threshold)
  result <- evaluate_predictions(model, artifacts$validation, repeats = 1L)$overall
  result[, id := config$id]
  result
})
comparison <- rbindlist(evaluations)
# Determine hyperparameters using validation only, never the test partition.
eligible <- comparison[method %in% c("interpolated", "witten_bell")]
leader <- max(eligible$top3)
eligible <- eligible[top3 >= leader - .005]
setorder(eligible, memory_mib, -top3, id)
selected <- eligible[1]
selected_model <- prepare_prediction_model(artifacts$counts, selected$method, selected$threshold)
legacy_model <- prepare_prediction_model(artifacts$counts, "legacy", 2L)
message("Final test, fixed selection: ", selected$id)
test_selected <- evaluate_predictions(selected_model, artifacts$test, repeats = 3L)
test_legacy <- evaluate_predictions(legacy_model, artifacts$test, repeats = 1L)
pair_difference <- as.integer(test_selected$details$rank > 0L) -
  as.integer(test_legacy$details$rank > 0L)
set.seed(20261003L)
bootstrap <- replicate(2000L, mean(sample(pair_difference, replace = TRUE)))
result <- list(version = 2L, input_metadata = metadata,
  source_code_md5 = unname(tools::md5sum("R/predictive_model_v2.R")),
  partitions = artifacts$partitions, validation = comparison, selection = selected,
  test_selected = test_selected, test_legacy = test_legacy,
  improvement_ci = unname(quantile(bootstrap, c(.025, .975))),
  environment = sessionInfo(), built_at = Sys.time())
saveRDS(selected_model, "models/selected_predictor_v2.rds", compress = "gzip")
saveRDS(result, "models/prediction_evaluation_v2.rds", compress = "gzip")
fwrite(comparison, "models/prediction_validation_v2.csv")
fwrite(rbindlist(list(test_selected$overall, test_legacy$overall)), "models/prediction_test_v2.csv")
fwrite(test_selected$by_source, "models/prediction_test_by_source_v2.csv")
print(comparison[, .(id, top1, top3, perplexity, milliseconds, memory_mib)])
print(test_selected$overall)
print(test_legacy$overall)
print(result$improvement_ci)
cat("Model and evaluation saved. Knit 12_predictive_model_evaluation.Rmd.\n")
