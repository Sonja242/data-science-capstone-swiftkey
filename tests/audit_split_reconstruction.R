# Independent reconstruction audit of the prior experiments. Sonja Sahebzad.
# No access to any newly reserved learning-curve final test.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
project <- normalizePath(args[1L], winslash = "/", mustWork = TRUE)
output <- args[2L]
setwd(project)
library(data.table)
source("R/predictive_model_v3.R")
setDTthreads(4L)
rows <- readRDS("models/expanded_partitions_v3.rds")
rows[, row_id := .I]
stopifnot(uniqueN(rows$text) == nrow(rows), all(rows$split %in% c("train", "validation", "test")))
eligible <- rows[stringi::stri_count_fixed(text, " ") >= 2L]
pick <- function(x, n, seed) {
  set.seed(seed)
  x[, .SD[sample.int(.N, min(.N, n))], by = source]
}
hashes <- function(x) unname(vapply(x$text, digest::digest, character(1), algo = "sha256", serialize = FALSE))
old_val <- pick(eligible[split == "validation"], 600L, 20261102L)
old_test <- pick(eligible[split == "test"], 1000L, 20261103L)
# Validate equivalence of the original sampling helper and reconstructed indices.
words <- strsplit(old_test$text, " ", fixed = TRUE)
positions <- vapply(words, function(w) sample(2:length(w), 1L), integer(1))
reconstructed <- data.table(source = old_test$source,
  prefix = mapply(function(w, k) paste(head(w, k - 1L), collapse = " "), words, positions),
  actual = mapply(function(w, k) w[k], words, positions))
original <- prediction_cases(rows[split == "test"], 1000L, 20261103L)
stopifnot(identical(reconstructed, original))
neural_val <- pick(eligible[split == "validation" & !row_id %in% old_val$row_id], 100L, 20261201L)
neural_test <- pick(eligible[split == "test" & !row_id %in% old_test$row_id], 300L, 20261202L)
adaptation_val <- pick(eligible[split == "validation" & !row_id %in% c(old_val$row_id, neural_val$row_id)], 200L, 20261301L)
adaptation_test <- pick(eligible[split == "test" & !row_id %in% c(old_test$row_id, neural_test$row_id)], 300L, 20261302L)
selected <- list(neural_development = neural_val, neural_test = neural_test,
                 adaptation_development = adaptation_val, adaptation_test = adaptation_test)
paths <- c("data/neural_evaluation/development.csv", "data/neural_evaluation/test.csv",
           "data/adaptation/development.csv", "data/adaptation/test.csv")
for (i in seq_along(paths)) {
  exported <- fread(paths[i])
  stopifnot(identical(exported$source, selected[[i]]$source),
            identical(exported$line_hash, hashes(selected[[i]])))
}
all_eval <- rbindlist(c(list(old_val, old_test), selected))
stopifnot(uniqueN(all_eval$row_id) == nrow(all_eval), !any(all_eval$split == "train"))
local <- readLines("data/adaptation/local_training.txt", encoding = "UTF-8", warn = FALSE)
stopifnot(length(local) == 48000L, all(local %chin% rows[split == "train", text]),
          !any(local %chin% all_eval$text))
reserved <- fread("models/learning_curve_reserved_case_hashes.csv")
stopifnot(nrow(reserved) == 900L, uniqueN(reserved$line_hash) == 900L,
          identical(reserved[, .N, by = source]$N, rep(300L, 3L)),
          !any(reserved$line_hash %chin% hashes(all_eval)))
# This reads prior training text and only hashes from the new final set.
external <- readLines("data/adaptation/taskmaster_training.txt", encoding = "UTF-8", warn = FALSE)
training_hashes <- hashes(data.table(text = c(local, external)))
stopifnot(!any(reserved$line_hash %chin% training_hashes))
manifest <- list(passed = TRUE, author = "Sonja Sahebzad",
  partition_lines = nrow(rows), partition_normalized_lines_unique = TRUE,
  original_v3_sampling_reconstruction_identical = TRUE,
  four_prior_exports_reconstructed_exactly = TRUE,
  prior_evaluation_cases = nrow(all_eval), prior_evaluation_row_overlap = 0L,
  local_training_lines = length(local), local_training_all_in_train_partition = TRUE,
  new_learning_curve_holdout_text_or_outcomes_inspected = FALSE, reserved_hash_manifest_cases = 900L, reserved_hash_overlap_prior_evaluations = 0L, reserved_hash_overlap_prior_adaptation_training = 0L,
  limitations = "Exact normalized-line separation only; near duplicates, related documents and unknown pretrained-model overlap remain possible.")
jsonlite::write_json(manifest, output, pretty = TRUE, auto_unbox = TRUE)
print(manifest)