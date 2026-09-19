source("R/predictive_model_v3.R")
library(data.table)
setDTthreads(4L)
counts <- integer_ngram_counts(normalize_prediction_text(c(rep("birds can fly very high", 20),
  "little birds can swim in water", rep("boats can float in water", 10), "people read books at home")))
for (method in c("absolute", "kneser_ney")) for (pruning in c(1L, 2L)) {
  model <- prepare_expanded_model(counts, method = method, minimum_count = pruning)
  for (phrase in c("birds can", "little birds can", "unseen lexemes", "", "boats")) {
    dist <- expanded_distribution(model, phrase)
    stopifnot(all(is.finite(dist$probability)), all(dist$probability > 0),
              abs(sum(dist$probability) - 1) < 1e-10)
  }
  stopifnot(predict_next_v3(model, "birds can")$word[1] == "fly")
  options <- predict_next_v3(model, "birds can", choices = c("float", "fly", "unknownlexeme"))
  stopifnot(options$word[1] == "fly", is.na(options[word == "unknownlexeme", model_probability]))
}
stopifnot(!any(counts$tables[[2]]$w2 == 0L),
          counts$training_tokens == sum(lengths(strsplit(normalize_prediction_text(c(rep("birds can fly very high",20),
            "little birds can swim in water",rep("boats can float in water",10),"people read books at home"))," "))))
cat("PASS: line boundaries, counts, pruned normalized smoothing, continuation counts, unknown options.\n")
