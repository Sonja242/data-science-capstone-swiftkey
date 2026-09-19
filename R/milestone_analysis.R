# Milestone-report analysis utilities
# Author: Sonja Sahebzad

scan_milestone_file <- function(path,
                                source_name,
                                sample_probability = 0.0015,
                                maximum_sample_lines = 4500L,
                                chunk_size = 20000L,
                                seed = 20260919L) {
  if (!file.exists(path)) stop("Missing corpus file: ", path, call. = FALSE)
  if (!requireNamespace("stringi", quietly = TRUE)) {
    stop("Package 'stringi' is required for Unicode-aware word counts.", call. = FALSE)
  }

  previous_locale <- Sys.getlocale("LC_CTYPE")
  on.exit(suppressWarnings(Sys.setlocale("LC_CTYPE", previous_locale)), add = TRUE)
  activate_corpus_locale()

  connection <- file(path, open = "r")
  on.exit(close(connection), add = TRUE)
  set.seed(as.integer(seed))

  line_count <- 0
  word_count <- 0
  sampled_lines <- character()

  repeat {
    chunk <- readLines(connection, n = chunk_size, warn = FALSE, skipNul = TRUE)
    if (length(chunk) == 0L) break

    line_count <- line_count + length(chunk)
    word_count <- word_count + sum(stringi::stri_count_words(chunk), na.rm = TRUE)

    remaining <- maximum_sample_lines - length(sampled_lines)
    if (remaining > 0L) {
      selected <- chunk[runif(length(chunk)) < sample_probability]
      if (length(selected) > remaining) selected <- selected[seq_len(remaining)]
      sampled_lines <- c(sampled_lines, selected)
    }
  }

  sampled_lines <- iconv(sampled_lines, from = "windows-1252", to = "UTF-8", sub = "")
  sampled_lines <- sampled_lines[!is.na(sampled_lines)]

  list(
    summary = data.frame(
      source = source_name,
      file = basename(path),
      size_mib = unname(file.info(path)$size / 1024^2),
      lines = line_count,
      words = word_count,
      words_per_line = word_count / line_count,
      sampled_lines = length(sampled_lines),
      stringsAsFactors = FALSE
    ),
    sample = data.frame(
      source = source_name,
      text = sampled_lines,
      characters = nchar(sampled_lines, type = "chars"),
      words = stringi::stri_count_words(sampled_lines),
      stringsAsFactors = FALSE
    )
  )
}

build_milestone_artifacts <- function(paths,
                                      sample_probability = 0.0015,
                                      maximum_sample_lines = 4500L,
                                      chunk_size = 20000L,
                                      seed = 20260919L) {
  scans <- Map(
    f = function(path, source_name, source_seed) {
      scan_milestone_file(
        path = path,
        source_name = source_name,
        sample_probability = sample_probability,
        maximum_sample_lines = maximum_sample_lines,
        chunk_size = chunk_size,
        seed = source_seed
      )
    },
    path = unname(paths),
    source_name = names(paths),
    source_seed = seed + seq_along(paths)
  )

  summaries <- do.call(rbind, lapply(scans, `[[`, "summary"))
  samples <- do.call(rbind, lapply(scans, `[[`, "sample"))
  row.names(summaries) <- NULL
  row.names(samples) <- NULL

  list(summary = summaries, sample = samples)
}

english_stopwords <- function() {
  c(
    "a", "about", "after", "all", "also", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "but", "by", "can", "could", "did",
    "do", "does", "for", "from", "get", "had", "has", "have", "he", "her",
    "here", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "just", "me", "more", "my", "no", "not", "of", "on", "one", "or", "our",
    "out", "said", "she", "so", "some", "than", "that", "the", "their", "them",
    "then", "there", "they", "this", "to", "up", "us", "was", "we", "were",
    "what", "when", "which", "who", "will", "with", "would", "you", "your"
  )
}

sample_top_terms <- function(sample_lines, n = 15L) {
  tokens <- tokenize_english(sample_lines)
  tokens <- tokens[nchar(tokens) > 1L]
  tokens <- tokens[!(tokens %in% english_stopwords())]
  counts <- sort(table(tokens), decreasing = TRUE)
  counts <- head(counts, n)
  data.frame(
    term = factor(names(counts), levels = rev(names(counts))),
    frequency = as.integer(counts),
    stringsAsFactors = FALSE
  )
}

ngram_profile <- function(lines, maximum_n = 3L) {
  normalized <- normalize_english_text(lines)
  normalized <- normalized[nzchar(normalized)]

  rows <- lapply(seq_len(maximum_n), function(n) {
    grams <- unlist(lapply(strsplit(normalized, " ", fixed = TRUE), function(words) {
      if (length(words) < n) return(character())
      starts <- seq_len(length(words) - n + 1L)
      vapply(starts, function(i) paste(words[i:(i + n - 1L)], collapse = " "), character(1))
    }), use.names = FALSE)

    counts <- table(grams)
    data.frame(
      model = paste0(n, "-gram"),
      observed_tokens = length(grams),
      unique_sequences = length(counts),
      singleton_share_percent = if (length(counts)) 100 * mean(counts == 1L) else 0,
      stringsAsFactors = FALSE
    )
  })

  do.call(rbind, rows)
}
