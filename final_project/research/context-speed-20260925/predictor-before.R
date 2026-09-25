# Sonja Projects | Author: Sonja Sahebzad
# Standalone CPU inference. No corpus, Python, GPU or remote service is needed.
library(data.table)
data.table::setDTthreads(1L)

normalize_phrase <- function(text) {
  text <- stringi::stri_enc_toutf8(text, is_unknown_8bit = FALSE, validate = TRUE)
  text[is.na(text)] <- ""
  text <- stringi::stri_trans_tolower(text)
  text <- stringi::stri_replace_all_fixed(text, c("\u2018", "\u2019"), c("'", "'"), vectorize_all = FALSE)
  text <- stringi::stri_replace_all_regex(text, "https?://\\S+|www\\.\\S+|@[\\p{L}\\p{N}_]+|&[a-z]+;", " ")
  words <- stringi::stri_extract_all_regex(text, "[a-z]+(?:'[a-z]+)*", omit_no_match = TRUE)
  vapply(words, paste, character(1), collapse = " ")
}

predict_word <- function(model, phrase, top_n = 3L, dense = FALSE) {
  stopifnot(is.character(phrase), length(phrase) == 1L, top_n >= 1L,
            top_n <= length(model$vocabulary))
  clean <- normalize_phrase(phrase)
  words <- if (nzchar(clean)) strsplit(clean, " ", fixed = TRUE)[[1L]] else character()
  ids <- model$lookup[list(tail(words, model$maximum_order - 1L)), id]
  ids[is.na(ids)] <- -1L # OOV never matches a trained context; shorter contexts still work.
  ids <- c(rep.int(0L, model$maximum_order - 1L), ids)
  hits <- vector("list", model$maximum_order)
  used_order <- 1L
  for (n in 2:model$maximum_order) {
    hits[[n]] <- model$tables[[n]][as.list(tail(ids, n - 1L)), nomatch = 0L]
    if (nrow(hits[[n]])) used_order <- n
  }
  # Non-hit words only receive a common multiplier. The highest base words plus
  # all positive context contributions therefore contain the exact top N.
  candidate_ids <- if (dense) seq_along(model$vocabulary) else
    unique(c(head(model$base_order, top_n), unlist(lapply(hits, function(x) x$word_id))))
  p <- model$base_probability[candidate_ids]
  for (n in 2:model$maximum_order) {
    tab <- hits[[n]]
    if (!nrow(tab)) next
    p <- p * tab$backoff[1L]
    at <- match(tab$word_id, candidate_ids)
    p[at] <- p[at] + tab$weight
  }
  ranking <- order(-p, model$vocabulary[candidate_ids])
  take <- head(ranking, top_n)
  # Start-of-text padding contributes to n-gram order, but is not a typed word.
  context_words <- min(length(words), used_order - 1L)
  list(words = model$vocabulary[candidate_ids[take]], scores = p[take],
       order = used_order, context = paste(tail(words, used_order - 1L), collapse = " "),
       context_words = context_words, input_words = length(words),
       start_markers = used_order - 1L - context_words,
       normalized = clean,
       distribution = if (dense) p else NULL)
}
