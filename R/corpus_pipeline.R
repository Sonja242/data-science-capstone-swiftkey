# SwiftKey corpus processing utilities
# Author: Sonja Sahebzad

assert_scalar_number <- function(value, name, lower = -Inf, upper = Inf) {
  valid <- is.numeric(value) && length(value) == 1L && !is.na(value) &&
    value >= lower && value <= upper
  if (!valid) {
    stop(sprintf("`%s` must be one number between %s and %s.", name, lower, upper),
         call. = FALSE)
  }
  invisible(value)
}

locate_corpus_files <- function(project_root = ".") {
  data_dir <- file.path(project_root, "data", "final", "en_US")
  paths <- c(
    blogs = file.path(data_dir, "en_US.blogs.txt"),
    news = file.path(data_dir, "en_US.news.txt"),
    twitter = file.path(data_dir, "en_US.twitter.txt")
  )

  missing <- paths[!file.exists(paths)]
  if (length(missing) > 0L) {
    stop(
      paste("Missing official corpus file(s):", paste(missing, collapse = ", ")),
      call. = FALSE
    )
  }
  paths
}

build_file_inventory <- function(paths) {
  information <- file.info(paths)
  data.frame(
    source = names(paths),
    file = basename(paths),
    size_mib = round(information$size / 1024^2, 1),
    modified = as.Date(information$mtime),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

set_corpus_locale <- function() {
  requested <- if (.Platform$OS.type == "windows") {
    "English_United States.1252"
  } else {
    "en_US.ISO8859-1"
  }

  active <- suppressWarnings(Sys.setlocale("LC_CTYPE", requested))
  if (!nzchar(active)) {
    stop(
      paste("Could not activate the corpus-reading locale:", requested),
      call. = FALSE
    )
  }
  invisible(active)
}

sample_file_chunked <- function(path,
                                probability = 0.0005,
                                chunk_size = 10000L,
                                maximum = 2500L,
                                seed = 20260919L) {
  if (!file.exists(path)) stop("Corpus file does not exist: ", path, call. = FALSE)
  assert_scalar_number(probability, "probability", 0, 1)
  assert_scalar_number(chunk_size, "chunk_size", 1, Inf)
  assert_scalar_number(maximum, "maximum", 1, Inf)
  assert_scalar_number(seed, "seed", 0, Inf)

  previous_locale <- Sys.getlocale("LC_CTYPE")
  on.exit(suppressWarnings(Sys.setlocale("LC_CTYPE", previous_locale)), add = TRUE)
  set_corpus_locale()

  connection <- file(path, open = "r")
  on.exit(close(connection), add = TRUE)

  set.seed(as.integer(seed))
  selected <- character()
  lines_scanned <- 0L

  repeat {
    chunk <- readLines(
      connection,
      n = as.integer(chunk_size),
      warn = FALSE,
      skipNul = TRUE
    )
    if (length(chunk) == 0L) break

    lines_scanned <- lines_scanned + length(chunk)
    keep <- runif(length(chunk)) < probability
    if (any(keep)) selected <- c(selected, chunk[keep])

    if (length(selected) >= maximum) {
      selected <- selected[seq_len(maximum)]
      break
    }
  }

  selected <- iconv(
    selected,
    from = "windows-1252",
    to = "UTF-8",
    sub = ""
  )

  list(
    lines = selected[!is.na(selected)],
    diagnostics = data.frame(
      lines_scanned = lines_scanned,
      lines_sampled = sum(!is.na(selected)),
      sampling_rate = if (lines_scanned > 0L) sum(!is.na(selected)) / lines_scanned else 0,
      stringsAsFactors = FALSE
    )
  )
}

sample_corpus <- function(paths, config) {
  results <- Map(
    f = function(path, seed) {
      sample_file_chunked(
        path = path,
        probability = config$sampling_probability,
        chunk_size = config$chunk_size,
        maximum = config$maximum_per_source,
        seed = seed
      )
    },
    path = unname(paths),
    seed = config$seed + seq_along(paths)
  )
  names(results) <- names(paths)
  results
}

summarise_samples <- function(samples) {
  rows <- lapply(names(samples), function(source_name) {
    cbind(source = source_name, samples[[source_name]]$diagnostics)
  })
  result <- do.call(rbind, rows)
  row.names(result) <- NULL
  result$sampling_rate <- round(as.numeric(result$sampling_rate) * 100, 4)
  names(result)[names(result) == "sampling_rate"] <- "sampling_rate_percent"
  result
}

combine_sample_text <- function(samples) {
  unlist(lapply(samples, `[[`, "lines"), use.names = FALSE)
}

calculate_quality_metrics <- function(lines) {
  lengths <- nchar(lines, type = "chars", allowNA = TRUE)
  data.frame(
    sampled_lines = length(lines),
    empty_lines = sum(!nzchar(trimws(lines))),
    lines_with_url = sum(grepl("https?://|www\\.", lines, ignore.case = TRUE)),
    lines_with_mention = sum(grepl("@[[:alnum:]_]", lines)),
    median_characters = unname(median(lengths, na.rm = TRUE)),
    maximum_characters_in_sample = max(lengths, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}

normalize_english_text <- function(lines) {
  lines <- tolower(lines)
  lines <- gsub("https?://[^[:space:]]+|www\\.[^[:space:]]+", " ", lines)
  lines <- gsub("@[[:alnum:]_]+", " ", lines)
  lines <- gsub("&[[:alpha:]]+;", " ", lines)
  lines <- gsub("[^a-z']+", " ", lines)
  lines <- gsub("(^|[[:space:]])'+|'+([[:space:]]|$)", " ", lines)
  trimws(gsub("[[:space:]]+", " ", lines))
}

tokenize_english <- function(lines) {
  normalized <- normalize_english_text(lines)
  normalized <- normalized[nzchar(normalized)]
  tokens <- unlist(strsplit(normalized, " ", fixed = TRUE), use.names = FALSE)
  tokens[nzchar(tokens)]
}

filter_blocked_terms <- function(tokens, blocked_terms) {
  blocked_terms <- unique(tolower(blocked_terms))
  tokens[!(tokens %in% blocked_terms)]
}

count_top_terms <- function(tokens, stopwords = character(), n = 15L) {
  assert_scalar_number(n, "n", 1, Inf)
  retained <- tokens[!(tokens %in% unique(tolower(stopwords)))]
  counts <- sort(table(retained), decreasing = TRUE)
  counts <- head(counts, as.integer(n))
  data.frame(
    token = names(counts),
    frequency = as.integer(counts),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

validate_pipeline_output <- function(sample_text, raw_tokens, clean_tokens, top_terms) {
  checks <- c(
    sample_is_not_empty = length(sample_text) > 0L,
    tokens_were_created = length(raw_tokens) > 0L,
    filtering_did_not_add_tokens = length(clean_tokens) <= length(raw_tokens),
    top_terms_are_available = nrow(top_terms) > 0L,
    tokens_are_ascii_english = all(grepl("^[a-z]+(?:'[a-z]+)*$", clean_tokens))
  )

  if (!all(checks)) {
    stop(
      paste("Pipeline validation failed:", paste(names(checks)[!checks], collapse = ", ")),
      call. = FALSE
    )
  }
  data.frame(check = names(checks), passed = unname(checks), row.names = NULL)
}
