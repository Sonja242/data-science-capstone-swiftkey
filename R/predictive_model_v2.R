# Smoothed next-word prediction, author: Sonja Sahebzad
# A single normalized distribution is used for ranking and perplexity.
stopifnot(requireNamespace("data.table", quietly = TRUE),
          requireNamespace("stringi", quietly = TRUE))

normalize_prediction_text <- function(text) {
  text <- stringi::stri_enc_toutf8(text, is_unknown_8bit = FALSE, validate = TRUE)
  text[is.na(text)] <- ""
  text <- stringi::stri_trans_tolower(text)
  text <- stringi::stri_replace_all_fixed(text, c("\u2018", "\u2019"),
                                         c("'", "'"), vectorize_all = FALSE)
  text <- stringi::stri_replace_all_regex(text, "https?://\\S+|www\\.\\S+|@[\\p{L}\\p{N}_]+|&[a-z]+;", " ")
  tokens <- stringi::stri_extract_all_regex(text, "[a-z]+(?:'[a-z]+)*", omit_no_match = TRUE)
  vapply(tokens, paste, character(1), collapse = " ")
}

sample_prediction_data <- function(paths, probability = .02, seed = 20260924L) {
  rows <- lapply(seq_along(paths), function(i) {
    set.seed(seed + i)
    con <- file(paths[[i]], open = "rb")
    on.exit(close(con))
    pieces <- list()
    scanned <- 0L
    repeat {
      chunk <- readLines(con, n = 20000L, warn = FALSE, skipNul = TRUE, encoding = "UTF-8")
      if (!length(chunk)) break
      scanned <- scanned + length(chunk)
      pieces[[length(pieces) + 1L]] <- chunk[runif(length(chunk)) < probability]
    }
    selected <- unlist(pieces, use.names = FALSE)
    message(names(paths)[i], ": scanned ", scanned, "; sampled ", length(selected))
    data.table::data.table(source = names(paths)[i], text = normalize_prediction_text(selected))
  })
  result <- data.table::rbindlist(rows)
  # Remove duplicate normalized lines globally before partitioning.
  result <- unique(result[nzchar(text)], by = "text")
  set.seed(seed + 100L)
  result[, split := sample(c("train", "validation", "test"), .N, replace = TRUE,
                          prob = c(.8, .1, .1)), by = source]
  result
}

build_prediction_counts <- function(lines, maximum_order = 4L) {
  tokens <- strsplit(lines, " ", fixed = TRUE)
  tokens <- tokens[lengths(tokens) > 0L]
  tables <- lapply(seq_len(maximum_order), function(n) {
    eligible <- tokens[lengths(tokens) >= n]
    parts <- lapply(eligible, function(words) {
      starts <- seq_len(length(words) - n + 1L)
      context <- if (n == 1L) rep("", length(starts)) else {
        do.call(paste, lapply(0:(n - 2L), function(k) words[starts + k]))
      }
      list(context = context, word = words[starts + n - 1L])
    })
    rows <- data.table::rbindlist(parts)
    result <- rows[, .(count = .N), by = .(context, word)]
    data.table::setkey(result, context)
    message(n, "-grams: ", nrow(result), " distinct sequences")
    result
  })
  list(tables = tables, maximum_order = maximum_order, training_lines = length(lines))
}

prepare_prediction_model <- function(counts, method = "interpolated", minimum_count = 1L,
                                     discount = .75) {
  stopifnot(method %in% c("interpolated", "witten_bell", "unigram", "legacy"),
            minimum_count >= 1L, discount > 0, discount < 1)
  vocabulary <- sort(counts$tables[[1L]]$word)
  vocabulary <- c(vocabulary, "<unk>")
  unigrams <- counts$tables[[1L]]
  base_counts <- c(unigrams$count[match(head(vocabulary, -1L), unigrams$word)], 0)
  base_probability <- (base_counts + .1) / sum(base_counts + .1)
  tables <- lapply(seq_along(counts$tables), function(n) {
    if (n == 1L || method == "unigram") return(NULL)
    result <- data.table::copy(counts$tables[[n]][count >= minimum_count])
    result[, word_id := match(word, vocabulary)]
    result[, word := NULL]
    data.table::setkey(result, context)
    result
  })
  structure(list(version = 2L, method = method, minimum_count = minimum_count,
                 discount = discount, vocabulary = vocabulary,
                 base_probability = base_probability, tables = tables,
                 maximum_order = if (method == "unigram") 1L else counts$maximum_order,
                 training_lines = counts$training_lines), class = "next_word_model_v2")
}

prediction_distribution <- function(model, text) {
  stopifnot(length(text) == 1L)
  clean <- normalize_prediction_text(text)
  words <- if (nzchar(clean)) strsplit(clean, " ", fixed = TRUE)[[1L]] else character()
  probability <- model$base_probability
  evidence <- list(order = 1L, context = "", context_count = 0L, continuations = 0L)
  if (model$maximum_order >= 2L) for (n in 2:model$maximum_order) {
    if (length(words) < n - 1L) next
    context_value <- paste(tail(words, n - 1L), collapse = " ")
    tab <- model$tables[[n]][list(context_value), nomatch = 0L]
    if (!nrow(tab)) next
    total <- sum(tab$count)
    types <- nrow(tab)
    if (model$method == "interpolated") {
      remaining_mass <- model$discount * types / total
      probability <- remaining_mass * probability
      probability[tab$word_id] <- probability[tab$word_id] +
        (tab$count - model$discount) / total
    } else {
      remaining_mass <- types / (total + types)
      probability <- remaining_mass * probability
      probability[tab$word_id] <- probability[tab$word_id] + tab$count / (total + types)
    }
    evidence <- list(order = n, context = context_value,
                     context_count = total, continuations = types)
  }
  list(probability = probability, evidence = evidence)
}

rank_prediction <- function(model, distribution, top_n = 3L, choices = NULL) {
  if (is.null(choices)) {
    candidates <- seq_len(length(model$vocabulary) - 1L)
    # Vocabulary order provides a stable tie-break without repeatedly sorting strings.
    candidates <- candidates[order(-distribution$probability[candidates], method = "radix")]
    candidates <- head(candidates, top_n)
    result <- data.table::data.table(rank = seq_along(candidates),
      word = model$vocabulary[candidates], model_probability = distribution$probability[candidates])
  } else {
    choices <- unique(normalize_prediction_text(choices))
    if (any(!nzchar(choices) | grepl(" ", choices))) stop("Each option must be one word.")
    ids <- match(choices, model$vocabulary)
    # Unknown options receive no individual probability or invented lexical evidence.
    result <- data.table::data.table(word = choices,
      model_probability = ifelse(is.na(ids), NA_real_, distribution$probability[ids]),
      in_vocabulary = !is.na(ids))
    data.table::setorder(result, -model_probability, word, na.last = TRUE)
    result[, rank := seq_len(.N)]
  }
  result
}

predict_next_v2 <- function(model, text, top_n = 3L, choices = NULL) {
  distribution <- prediction_distribution(model, text)
  result <- rank_prediction(model, distribution, top_n, choices)
  attr(result, "evidence") <- distribution$evidence
  result
}

legacy_prediction_ids <- function(model, text, top_n = 3L) {
  clean <- normalize_prediction_text(text)
  words <- if (nzchar(clean)) strsplit(clean, " ", fixed = TRUE)[[1L]] else character()
  result <- integer()
  for (n in rev(seq_len(model$maximum_order))) {
    if (n == 1L) {
      candidates <- order(-model$base_probability, method = "radix")
      candidates <- candidates[candidates != length(model$vocabulary)]
    } else {
      if (length(words) < n - 1L) next
      context_value <- paste(tail(words, n - 1L), collapse = " ")
      tab <- model$tables[[n]][list(context_value), nomatch = 0L]
      if (!nrow(tab)) next
      candidates <- tab$word_id[order(-tab$count, tab$word_id)]
    }
    result <- unique(c(result, candidates))
    if (length(result) >= top_n) break
  }
  head(result, top_n)
}

prediction_cases <- function(rows, per_source = 400L, seed = 20261001L) {
  result <- data.table::copy(rows)
  result[, words := strsplit(text, " ", fixed = TRUE)]
  result <- result[lengths(words) >= 3L]
  set.seed(seed)
  result <- result[, .SD[sample.int(.N, min(.N, per_source))], by = source]
  # Random positions represent intermediate typing, rather than sentence endings only.
  positions <- vapply(result$words, function(w) sample(2:length(w), 1L), integer(1))
  result[, prefix := mapply(function(w, k) paste(head(w, k - 1L), collapse = " "), words, positions)]
  result[, actual := mapply(function(w, k) w[k], words, positions)]
  result[, .(source, prefix, actual)]
}

evaluate_predictions <- function(model, cases, repeats = 2L) {
  predict_row <- function(text) {
    if (model$method == "legacy") return(list(ids = legacy_prediction_ids(model, text), p = NULL))
    distribution <- prediction_distribution(model, text)
    ranks <- rank_prediction(model, distribution)
    list(ids = match(ranks$word, model$vocabulary), p = distribution$probability)
  }
  invisible(predict_row(cases$prefix[[1L]]))
  elapsed <- numeric(repeats)
  for (r in seq_len(repeats)) {
    start <- proc.time()[[3L]]
    # Store only the target probability, ranks and compact details, not a dense matrix.
    details <- data.table::rbindlist(lapply(seq_len(nrow(cases)), function(i) {
      output <- predict_row(cases$prefix[[i]])
      target_id <- match(cases$actual[[i]], model$vocabulary)
      oov <- is.na(target_id)
      if (oov) target_id <- length(model$vocabulary)
      rank <- if (oov) 0L else match(target_id, output$ids, nomatch = 0L)
      data.table::data.table(source = cases$source[[i]], rank = rank, oov = oov,
        target_probability = if (is.null(output$p)) NA_real_ else output$p[target_id])
    }))
    elapsed[r] <- proc.time()[[3L]] - start
  }
  summarize <- function(d) data.table::data.table(
    cases = nrow(d), top1 = mean(d$rank == 1L),
    top2 = mean(d$rank > 0L & d$rank <= 2L), top3 = mean(d$rank > 0L),
    mrr = mean(ifelse(d$rank > 0L, 1 / pmax(d$rank, 1L), 0)),
    perplexity = if (all(is.na(d$target_probability))) NA_real_ else exp(-mean(log(d$target_probability))),
    oov_percent = 100 * mean(d$oov))
  overall <- summarize(details)
  latency_ms <- median(elapsed) * 1000 / nrow(cases)
  overall[, `:=`(method = model$method, threshold = model$minimum_count,
                 milliseconds = latency_ms,
                 memory_mib = as.numeric(object.size(model)) / 1024^2)]
  by_source <- data.table::rbindlist(lapply(unique(details$source), function(s) {
    value <- summarize(details[source == s]); value[, source := s]; value
  }))
  list(overall = overall, by_source = by_source, details = details)
}
