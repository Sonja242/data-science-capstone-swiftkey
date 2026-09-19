source(if (file.exists("R/predictive_model_v2.R")) "R/predictive_model_v2.R" else "_capstone_stage/predictive_model_v2.R")
library(data.table)
toy <- normalize_prediction_text(c(rep("birds can fly", 20), "small birds can swim",
  rep("boats can float", 8), "people read books", "people write books"))
counts <- build_prediction_counts(toy)
for (method in c("interpolated", "witten_bell", "unigram")) {
  model <- prepare_prediction_model(counts, method)
  for (input in c("small birds can", "unseen words here", "", "BIRDS, can")) {
    dist <- prediction_distribution(model, input)
    stopifnot(abs(sum(dist$probability) - 1) < 1e-12,
      all(is.finite(dist$probability)), all(dist$probability > 0))
    p <- predict_next_v2(model, input)
    stopifnot(!anyDuplicated(p$word), all(diff(p$model_probability) <= 0),
      identical(p$model_probability, dist$probability[match(p$word, model$vocabulary)]))
  }
}
model <- prepare_prediction_model(counts)
stopifnot(predict_next_v2(model, "small birds can")$word[1] == "fly")
choices <- predict_next_v2(model, "birds can", choices = c("float", "fly", "unseenlexeme"))
stopifnot(choices$word[1] == "fly", is.na(choices[word == "unseenlexeme", model_probability]))
stopifnot(normalize_prediction_text("You're here; I\u2019m too!") == "you're here i'm too")
cases <- data.table(source = "test", prefix = c("birds can", "unknown"), actual = c("fly", "unseenlexeme"))
ev <- evaluate_predictions(model, cases, repeats = 1L)
stopifnot(ev$overall$top1 == .5, ev$overall$oov_percent == 50, is.finite(ev$overall$perplexity),
          is.finite(ev$overall$milliseconds), ev$overall$milliseconds >= 0)
cat("PASS: probability mass, shared ranking/probability, rare-context smoothing, OOV, punctuation, metrics.\n")
