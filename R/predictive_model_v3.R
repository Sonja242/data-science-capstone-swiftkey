# Expanded corpus model. Author: Sonja Sahebzad.
# Reuses the documented tokenizer; all model changes are independently evaluated.
source("R/predictive_model_v2.R")

expanded_partitions <- function(paths) {
  # Exclude every line in the previous experiment, including its training data.
  # This also permits a fair test of the unchanged v2 model on fresh examples.
  old <- sample_prediction_data(paths, probability = .02, seed = 20260924L)
  excluded <- old$text
  rm(old)
  pieces <- lapply(seq_along(paths), function(i) {
    con <- file(paths[[i]], "rb")
    on.exit(close(con))
    chunks <- list()
    repeat {
      raw <- readLines(con, 20000L, warn = FALSE, skipNul = TRUE, encoding = "UTF-8")
      if (!length(raw)) break
      chunks[[length(chunks) + 1L]] <- normalize_prediction_text(raw)
    }
    data.table::data.table(source = names(paths)[i], text = unlist(chunks, use.names = FALSE))
  })
  rows <- unique(data.table::rbindlist(pieces), by = "text")
  rows <- rows[nzchar(text) & !text %in% excluded]
  set.seed(20261101L)
  rows[, split := sample(c("train", "validation", "test"), .N, TRUE,
    prob = c(.8, .1, .1)), by = source]
  rows[, small := runif(.N) < .25]
  rows
}

integer_ngram_counts <- function(lines, maximum_order = 5L) {
  token_lists <- strsplit(lines, " ", fixed = TRUE)
  flat <- unlist(token_lists, use.names = FALSE)
  vocabulary <- sort(unique(flat))
  # 0 is a line-start symbol, never a predicted word.
  word_lengths <- lengths(token_lists)
  lens <- word_lengths + maximum_order - 1L
  starts <- c(1L, head(cumsum(lens), -1L))
  ids <- integer(sum(lens))
  ids[sequence(word_lengths, from = starts + maximum_order - 1L)] <- match(flat, vocabulary)
  doc <- rep.int(seq_along(lens), lens)
  rm(token_lists, flat, starts)
  gc(FALSE)
  tables <- lapply(seq_len(maximum_order), function(n) {
    starts <- seq_len(length(ids) - n + 1L)
    ends <- starts + n - 1L
    starts <- starts[doc[starts] == doc[ends] & ids[ends] != 0L]
    cols <- setNames(lapply(0:(n - 1L), function(k) ids[starts + k]), paste0("w", seq_len(n)))
    tab <- data.table::as.data.table(cols)
    counted <- tab[, .(count = .N), by = names(tab)]
    message(n, "-grams: ", format(nrow(counted), big.mark = ","))
    counted
  })
  list(tables = tables, vocabulary = vocabulary, training_lines = length(lines),
       training_tokens = sum(lens - maximum_order + 1L))
}

prepare_expanded_model <- function(counts, maximum_order = 5L, method = "kneser_ney",
                                   minimum_count = 2L, discount = .75) {
  stopifnot(method %in% c("kneser_ney", "absolute"), maximum_order <= length(counts$tables))
  vocab <- c(counts$vocabulary, "<unk>")
  effective <- function(n) {
    if (method == "kneser_ney" && n < maximum_order) {
      suffix <- paste0("w", 2:(n + 1L))
      tab <- counts$tables[[n + 1L]][, .(count = .N), by = suffix]
      data.table::setnames(tab, suffix, paste0("w", seq_len(n)))
      tab
    } else data.table::copy(counts$tables[[n]])
  }
  unigram <- effective(1L)
  base_counts <- numeric(length(vocab))
  base_counts[unigram$w1] <- unigram$count
  base <- (base_counts + .1) / sum(base_counts + .1)
  discounts <- numeric(maximum_order)
  tables <- lapply(2:maximum_order, function(n) {
    tab <- effective(n)
    d <- if (identical(discount, "estimated")) {
      n1 <- sum(tab$count == 1L); n2 <- sum(tab$count == 2L)
      if (n1 + 2 * n2 > 0) max(.1, min(.95, n1 / (n1 + 2 * n2))) else .75
    } else discount
    discounts[n] <<- d
    contexts <- paste0("w", seq_len(n - 1L))
    totals <- tab[, .(total = sum(count)), by = contexts]
    kept <- tab[count >= minimum_count]
    # Reallocate both discounted and pruned probability mass to shorter contexts.
    kept[, mass := count - d]
    stats <- kept[, .(kept_mass = sum(mass)), by = contexts]
    stats <- merge(stats, totals, by = contexts, sort = FALSE)
    stats[, backoff := pmax(0, 1 - kept_mass / total)]
    kept <- merge(kept, stats[, c(contexts, "total", "backoff"), with = FALSE],
                  by = contexts, sort = FALSE)
    kept[, weight := mass / total]
    kept[, mass := NULL]
    data.table::setnames(kept, paste0("w", n), "word_id")
    data.table::setkeyv(kept, contexts)
    kept
  })
  lookup <- data.table::data.table(word = vocab, id = seq_along(vocab))
  data.table::setkey(lookup, word)
  list(version = 3L, vocabulary = vocab, lookup = lookup, tables = c(list(NULL), tables),
       base_probability = base, maximum_order = maximum_order, method = method,
       minimum_count = minimum_count, discounts = discounts,
       training_lines = counts$training_lines, training_tokens = counts$training_tokens)
}

expanded_distribution <- function(model, text) {
  clean <- normalize_prediction_text(text)
  words <- if (nzchar(clean)) strsplit(clean, " ", fixed = TRUE)[[1L]] else character()
  ids <- model$lookup[list(words), id]
  ids[is.na(ids)] <- length(model$vocabulary)
  ids <- c(rep.int(0L, model$maximum_order - 1L), ids)
  probability <- model$base_probability
  evidence <- list(order = 1L, context_count = 0L, context = "")
  for (n in 2:model$maximum_order) {
    context <- tail(ids, n - 1L)
    tab <- model$tables[[n]][as.list(context), nomatch = 0L]
    if (!nrow(tab)) next
    probability <- probability * tab$backoff[1L]
    probability[tab$word_id] <- probability[tab$word_id] + tab$weight
    context_words <- ifelse(context == 0L, "<start>", model$vocabulary[pmax(context, 1L)])
    evidence <- list(order = n, context_count = tab$total[1L], context = paste(context_words, collapse = " "))
  }
  list(probability = probability, evidence = evidence)
}

predict_next_v3 <- function(model, text, top_n = 3L, choices = NULL) {
  dist <- expanded_distribution(model, text)
  out <- rank_expanded(model, dist, top_n, choices)
  attr(out, "evidence") <- dist$evidence
  out
}

rank_expanded <- function(model, distribution, top_n = 3L, choices = NULL) {
  if (is.null(choices)) return(rank_prediction(model, distribution, top_n))
  choices <- unique(normalize_prediction_text(choices))
  if (any(!nzchar(choices) | grepl(" ", choices))) stop("Each option must be one word.")
  ids <- model$lookup[list(choices), id]
  out <- data.table::data.table(word = choices, model_probability = distribution$probability[ids])
  data.table::setorder(out, -model_probability, word, na.last = TRUE)
  out[, rank := seq_len(.N)]
  out
}

make_choice_cases <- function(cases, frequency, seed) {
  # Diagnostic four-choice task: three distractors with similar training frequency.
  # These are synthetic alternatives, not real quiz questions or a semantic test.
  vocab <- frequency$word[order(-frequency$count, frequency$word)]
  set.seed(seed)
  lapply(cases$actual, function(target) {
    at <- match(target, vocab)
    if (is.na(at)) at <- length(vocab)
    pool <- vocab[seq.int(max(1L, at - 250L), min(length(vocab), at + 250L))]
    pool <- setdiff(pool, target)
    sample(c(target, sample(pool, 3L)), 4L)
  })
}

evaluate_expanded <- function(model, cases, options, version = 3L) {
  invisible(if (version == 3L) expanded_distribution(model, "a simple example") else
    prediction_distribution(model, "a simple example"))
  started <- proc.time()[[3L]]
  details <- data.table::rbindlist(lapply(seq_len(nrow(cases)), function(i) {
    dist <- if (version == 3L) expanded_distribution(model, cases$prefix[i]) else
      prediction_distribution(model, cases$prefix[i])
    ranked <- rank_prediction(model, dist)
    target <- if (version == 3L) model$lookup[list(cases$actual[i]), id] else
      match(cases$actual[i], model$vocabulary)
    oov <- is.na(target)
    if (oov) target <- length(model$vocabulary)
    choice_rank <- if (version == 3L) rank_expanded(model, dist, choices = options[[i]]) else
      rank_prediction(model, dist, choices = options[[i]])
    data.table::data.table(source = cases$source[i], rank = match(cases$actual[i], ranked$word, nomatch = 0L),
      choice_correct = choice_rank$word[1L] == cases$actual[i] && is.finite(choice_rank$model_probability[1L]),
      oov = oov, probability = dist$probability[target])
  }))
  latency <- (proc.time()[[3L]] - started) * 1000 / nrow(cases)
  summarize <- function(x) data.table::data.table(cases = nrow(x), top1 = mean(x$rank == 1L),
    top3 = mean(x$rank > 0L), choice_accuracy = mean(x$choice_correct),
    mrr_at_3 = mean(ifelse(x$rank > 0L, 1 / pmax(x$rank, 1L), 0)),
    perplexity = exp(-mean(log(x$probability))), oov = mean(x$oov))
  overall <- summarize(details)
  overall[, `:=`(milliseconds = latency, memory_mib = as.numeric(object.size(model)) / 1024^2)]
  by_source <- data.table::rbindlist(lapply(unique(details$source), function(s) {
    out <- summarize(details[source == s]); out[, source := s]; out
  }))
  list(overall = overall, by_source = by_source, details = details)
}
