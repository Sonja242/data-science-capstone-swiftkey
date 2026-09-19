# Generic chunked text-file query utilities
# Author: Sonja Sahebzad

with_text_connection <- function(path, callback, chunk_size = 10000L) {
  if (!file.exists(path)) stop("Text file does not exist: ", path, call. = FALSE)
  if (!is.function(callback)) stop("`callback` must be a function.", call. = FALSE)
  if (!is.numeric(chunk_size) || length(chunk_size) != 1L || chunk_size < 1L) {
    stop("`chunk_size` must be one positive number.", call. = FALSE)
  }

  previous_locale <- Sys.getlocale("LC_CTYPE")
  on.exit(suppressWarnings(Sys.setlocale("LC_CTYPE", previous_locale)), add = TRUE)

  requested_locale <- if (.Platform$OS.type == "windows") {
    "English_United States.1252"
  } else {
    "en_US.ISO8859-1"
  }
  active_locale <- suppressWarnings(Sys.setlocale("LC_CTYPE", requested_locale))
  if (!nzchar(active_locale)) stop("Required text locale is unavailable.", call. = FALSE)

  connection <- file(path, open = "r")
  on.exit(close(connection), add = TRUE)

  state <- NULL
  repeat {
    chunk <- readLines(
      connection,
      n = as.integer(chunk_size),
      warn = FALSE,
      skipNul = TRUE
    )
    if (length(chunk) == 0L) break
    state <- callback(chunk, state)
  }
  state
}

maximum_line_length <- function(path, chunk_size = 10000L) {
  reducer <- function(chunk, current_maximum) {
    if (is.null(current_maximum)) current_maximum <- 0L
    lengths <- nchar(chunk, type = "chars", allowNA = TRUE)
    max(current_maximum, lengths, na.rm = TRUE)
  }
  with_text_connection(path, reducer, chunk_size)
}

escape_regular_expression <- function(text) {
  if (!is.character(text) || length(text) != 1L || !nzchar(text)) {
    stop("Search text must be one non-empty character value.", call. = FALSE)
  }
  gsub("([][{}()+*^$|\\\\?.])", "\\\\\\1", text)
}

count_lines_with_word <- function(path,
                                  word,
                                  case_sensitive = TRUE,
                                  chunk_size = 10000L) {
  escaped_word <- escape_regular_expression(word)
  pattern <- paste0("(?<![[:alnum:]_])", escaped_word, "(?![[:alnum:]_])")

  reducer <- function(chunk, running_total) {
    if (is.null(running_total)) running_total <- 0L
    running_total + sum(grepl(
      pattern,
      chunk,
      perl = TRUE,
      ignore.case = !case_sensitive
    ))
  }
  with_text_connection(path, reducer, chunk_size)
}

safe_count_ratio <- function(numerator_count, denominator_count) {
  if (denominator_count == 0L) return(NA_real_)
  numerator_count / denominator_count
}

find_lines_with_word <- function(path,
                                 word,
                                 case_sensitive = TRUE,
                                 maximum_matches = 20L,
                                 chunk_size = 10000L) {
  escaped_word <- escape_regular_expression(word)
  pattern <- paste0("(?<![[:alnum:]_])", escaped_word, "(?![[:alnum:]_])")

  reducer <- function(chunk, matches) {
    if (is.null(matches)) matches <- character()
    found <- chunk[grepl(
      pattern,
      chunk,
      perl = TRUE,
      ignore.case = !case_sensitive
    )]
    head(c(matches, found), maximum_matches)
  }
  with_text_connection(path, reducer, chunk_size)
}

count_exact_lines <- function(path, target, chunk_size = 10000L) {
  if (!is.character(target) || length(target) != 1L) {
    stop("`target` must be one character value.", call. = FALSE)
  }

  reducer <- function(chunk, running_total) {
    if (is.null(running_total)) running_total <- 0L
    running_total + sum(chunk == target, na.rm = TRUE)
  }
  with_text_connection(path, reducer, chunk_size)
}
