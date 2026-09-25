# Research only. Author: Sonja Sahebzad | Sonja Projects
predict_cached <- function(model, phrase, lambda = 0.05, top_n = 3L, dense = FALSE) {
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
    if (is.null(model$search_index)) {
      hits[[n]] <- model$tables[[n]][as.list(tail(ids, n - 1L)), nomatch = 0L]
    } else {
      at <- context_rows(model, n, tail(ids, n - 1L))
      tab <- model$tables[[n]]
      hits[[n]] <- list(word_id = tab$word_id[at], weight = tab$weight[at],
                        backoff = tab$backoff[at])
    }
    if (length(hits[[n]]$word_id)) used_order <- n
  }
  # Non-hit words only receive a common multiplier. The highest base words plus
  # all positive context contributions therefore contain the exact top N.
  previous <- tail(words, 100L)
  cache_ids <- model$lookup[list(previous), id]
  cache_ids <- cache_ids[!is.na(cache_ids) & !cache_ids %in% head(model$base_order, 100L)]
  cache_counts <- table(cache_ids)
  cache_words <- as.integer(names(cache_counts))
  if (!length(cache_words)) lambda <- 0
  candidate_ids <- if (dense) seq_along(model$vocabulary) else
    unique(c(head(model$base_order, top_n), cache_words, unlist(lapply(hits, function(x) x$word_id))))
  p <- model$base_probability[candidate_ids]
  for (n in 2:model$maximum_order) {
    tab <- hits[[n]]
    if (!length(tab$word_id)) next
    p <- p * tab$backoff[1L]
    at <- match(tab$word_id, candidate_ids)
    p[at] <- p[at] + tab$weight
  }
  if (lambda > 0) {
    p <- p * (1 - lambda)
    at <- match(cache_words, candidate_ids)
    p[at] <- p[at] + lambda * as.numeric(cache_counts) / sum(cache_counts)
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
