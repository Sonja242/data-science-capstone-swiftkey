# Predictive text model utilities
# Author: Sonja Sahebzad

if (!requireNamespace("data.table", quietly = TRUE)) {
  stop("Package 'data.table' is required.", call. = FALSE)
}
if (!requireNamespace("stringi", quietly = TRUE)) {
  stop("Package 'stringi' is required.", call. = FALSE)
}

normalize_model_text <- function(text) {
  text <- tryCatch(
    iconv(text, from = "", to = "UTF-8", sub = ""),
    error = function(e) {
      vapply(text, function(value) {
        tryCatch(
          iconv(value, from = "", to = "UTF-8", sub = ""),
          error = function(e) ""
        )
      }, character(1), USE.NAMES = FALSE)
    }
  )
  text[is.na(text)] <- ""
  text <- stringi::stri_trans_tolower(text)
  text <- gsub("https?://\\S+|www\\.\\S+", " ", text, perl = TRUE)
  text <- gsub("@[[:alnum:]_]+", " ", text, perl = TRUE)
  text <- gsub("&[[:alpha:]]+;", " ", text, perl = TRUE)
  text <- gsub("[^[:alpha:]']+", " ", text, perl = TRUE)
  text <- gsub("(^|[[:space:]])'+|'+($|[[:space:]])", " ", text, perl = TRUE)
  trimws(gsub("[[:space:]]+", " ", text, perl = TRUE))
}

sample_corpus_lines <- function(paths,
                                probability = 0.02,
                                maximum_per_source = 60000L,
                                chunk_size = 20000L,
                                seed = 20260919L) {
  stopifnot(length(paths) > 0L, !is.null(names(paths)))

  sampled <- Map(function(path, source, source_seed) {
    if (!file.exists(path)) stop("Missing corpus file: ", path, call. = FALSE)
    set.seed(source_seed)
    con <- file(path, open = "r")
    on.exit(close(con), add = TRUE)
    result <- character()

    repeat {
      chunk <- readLines(con, n = chunk_size, warn = FALSE, skipNul = TRUE)
      if (!length(chunk)) break
      keep <- chunk[runif(length(chunk)) < probability]
      if (length(keep)) result <- c(result, keep)
      if (length(result) >= maximum_per_source) {
        result <- result[seq_len(maximum_per_source)]
        break
      }
    }

    data.frame(source = source, text = result, stringsAsFactors = FALSE)
  }, unname(paths), names(paths), seed + seq_along(paths))

  do.call(rbind, sampled)
}

split_training_validation <- function(sampled,
                                      validation_fraction = 0.15,
                                      seed = 20260920L) {
  set.seed(seed)
  is_validation <- ave(
    seq_len(nrow(sampled)), sampled$source,
    FUN = function(index) sample(c(FALSE, TRUE), length(index), replace = TRUE,
      prob = c(1 - validation_fraction, validation_fraction))
  )
  is_validation <- as.logical(is_validation)
  list(
    training = sampled[!is_validation, , drop = FALSE],
    validation = sampled[is_validation, , drop = FALSE]
  )
}

make_ngram_rows <- function(lines, order) {
  clean <- normalize_model_text(lines)
  words_by_line <- strsplit(clean[nzchar(clean)], " ", fixed = TRUE)
  rows <- lapply(words_by_line, function(words) {
    if (length(words) < order) return(NULL)
    starts <- seq_len(length(words) - order + 1L)
    target <- words[starts + order - 1L]
    context <- if (order == 1L) {
      rep("", length(starts))
    } else {
      vapply(starts, function(i) {
        paste(words[i:(i + order - 2L)], collapse = " ")
      }, character(1))
    }
    data.table::data.table(context = context, word = target)
  })
  data.table::rbindlist(rows, use.names = TRUE)
}

build_ngram_model <- function(training_lines, maximum_order = 4L) {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required.", call. = FALSE)
  }
  if (!requireNamespace("stringi", quietly = TRUE)) {
    stop("Package 'stringi' is required.", call. = FALSE)
  }

  tables <- lapply(seq_len(maximum_order), function(order) {
    rows <- make_ngram_rows(training_lines, order)
    counts <- rows[, .(count = .N), by = .(context, word)]
    data.table::setorder(counts, context, -count, word)
    data.table::setkey(counts, context)
    counts
  })
  names(tables) <- paste0(seq_len(maximum_order), "gram")

  structure(
    list(
      tables = tables,
      maximum_order = maximum_order,
      vocabulary_size = data.table::uniqueN(tables[[1L]]$word),
      training_lines = length(training_lines)
    ),
    class = "backoff_ngram_model"
  )
}

prune_ngram_model <- function(model, minimum_count = 2L) {
  pruned <- model
  pruned$tables <- Map(function(tab, order) {
    if (order == 1L) tab else tab[count >= minimum_count]
  }, model$tables, seq_along(model$tables))
  pruned$minimum_count <- minimum_count
  pruned
}

context_for_order <- function(words, order) {
  required <- order - 1L
  if (required == 0L) return("")
  if (length(words) < required) return(NA_character_)
  paste(tail(words, required), collapse = " ")
}

predict_next <- function(model, text, top_n = 3L) {
  clean <- normalize_model_text(text)
  words <- if (nzchar(clean)) strsplit(clean, " ", fixed = TRUE)[[1L]] else character()
  predictions <- data.table::data.table(
    word = character(), score = numeric(), order = integer()
  )

  for (order in rev(seq_len(model$maximum_order))) {
    context_value <- context_for_order(words, order)
    if (is.na(context_value)) next
    candidates <- model$tables[[order]][list(context_value), nomatch = 0L]
    if (!nrow(candidates)) next
    candidates <- candidates[order(-count, word)]
    candidates[, score := count / sum(count)]
    candidates[, order := order]
    predictions <- data.table::rbindlist(list(
      predictions,
      candidates[, .(word, score, order)]
    ))
    predictions <- predictions[!duplicated(word)]
    if (nrow(predictions) >= top_n) break
  }

  predictions <- head(predictions, top_n)
  predictions[, rank := seq_len(.N)]
  predictions[, .(rank, word, probability = score, context_order = order)]
}

actual_word_probability <- function(model, prefix, actual, alpha = 0.5) {
  clean <- normalize_model_text(prefix)
  words <- if (nzchar(clean)) strsplit(clean, " ", fixed = TRUE)[[1L]] else character()
  actual <- normalize_model_text(actual)
  vocabulary <- max(model$vocabulary_size, 1L)

  for (order in rev(seq_len(model$maximum_order))) {
    context_value <- context_for_order(words, order)
    if (is.na(context_value)) next
    candidates <- model$tables[[order]][list(context_value), nomatch = 0L]
    if (!nrow(candidates)) next
    observed <- candidates[word == actual, sum(count)]
    if (!length(observed)) observed <- 0
    return((observed + alpha) / (sum(candidates$count) + alpha * vocabulary))
  }
  1 / vocabulary
}

make_validation_cases <- function(lines, maximum_cases = 1000L, seed = 20260921L) {
  clean <- normalize_model_text(lines)
  words <- strsplit(clean[nzchar(clean)], " ", fixed = TRUE)
  words <- words[lengths(words) >= 2L]
  cases <- data.frame(
    prefix = vapply(words, function(x) paste(head(x, -1L), collapse = " "), character(1)),
    actual = vapply(words, tail, character(1), n = 1L),
    stringsAsFactors = FALSE
  )
  if (nrow(cases) > maximum_cases) {
    set.seed(seed)
    cases <- cases[sample.int(nrow(cases), maximum_cases), , drop = FALSE]
  }
  cases
}

evaluate_ngram_model <- function(model, cases) {
  started <- proc.time()[[3L]]
  predictions <- lapply(cases$prefix, predict_next, model = model, top_n = 3L)
  elapsed <- proc.time()[[3L]] - started

  ranks <- mapply(function(pred, actual) {
    match(actual, pred$word, nomatch = 0L)
  }, predictions, cases$actual)
  probabilities <- mapply(actual_word_probability,
    prefix = cases$prefix, actual = cases$actual,
    MoreArgs = list(model = model)
  )

  data.frame(
    minimum_count = model$minimum_count %||% 1L,
    validation_cases = nrow(cases),
    top1_accuracy = mean(ranks == 1L),
    top2_accuracy = mean(ranks > 0L & ranks <= 2L),
    top3_accuracy = mean(ranks > 0L & ranks <= 3L),
    mean_reciprocal_rank = mean(ifelse(ranks > 0L, 1 / ranks, 0)),
    perplexity = exp(-mean(log(pmax(probabilities, .Machine$double.xmin)))),
    milliseconds_per_prediction = 1000 * elapsed / nrow(cases),
    model_size_mib = as.numeric(object.size(model)) / 1024^2,
    stringsAsFactors = FALSE
  )
}

`%||%` <- function(x, y) if (is.null(x)) y else x

format_predictions <- function(predictions) {
  if (!nrow(predictions)) return("No prediction available.")
  paste0(
    predictions$rank, ". ", predictions$word,
    " (", round(100 * predictions$probability, 1),
    "%, ", predictions$context_order, "-gram)"
  )
}
