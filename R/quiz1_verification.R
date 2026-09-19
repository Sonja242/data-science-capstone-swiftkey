# Reproducible verification utilities for completed Capstone Quiz 1
# Author: Sonja Sahebzad

official_corpus_paths <- function(project_root = ".") {
  paths <- c(
    blogs = file.path(project_root, "data", "final", "en_US", "en_US.blogs.txt"),
    news = file.path(project_root, "data", "final", "en_US", "en_US.news.txt"),
    twitter = file.path(project_root, "data", "final", "en_US", "en_US.twitter.txt")
  )

  missing <- paths[!file.exists(paths)]
  if (length(missing) > 0L) {
    stop("Missing official corpus file(s): ", paste(missing, collapse = ", "), call. = FALSE)
  }
  paths
}

activate_corpus_locale <- function() {
  requested <- if (.Platform$OS.type == "windows") {
    "English_United States.1252"
  } else {
    "en_US.ISO8859-1"
  }

  active <- suppressWarnings(Sys.setlocale("LC_CTYPE", requested))
  if (!nzchar(active)) {
    stop("Could not activate the locale required to read the archived corpus.", call. = FALSE)
  }
  invisible(active)
}

scan_corpus_file <- function(path,
                             source_name,
                             inspect_twitter = FALSE,
                             chunk_size = 10000L) {
  previous_locale <- Sys.getlocale("LC_CTYPE")
  on.exit(suppressWarnings(Sys.setlocale("LC_CTYPE", previous_locale)), add = TRUE)
  activate_corpus_locale()

  connection <- file(path, open = "r")
  on.exit(close(connection), add = TRUE)

  target_sentence <- paste(
    "A computer once beat me at chess, but it was no match for me at kickboxing"
  )

  line_count <- 0L
  longest_line <- 0L
  love_count <- 0L
  hate_count <- 0L
  biostat_matches <- character()
  exact_sentence_count <- 0L

  repeat {
    chunk <- readLines(
      connection,
      n = as.integer(chunk_size),
      warn = FALSE,
      skipNul = TRUE
    )
    if (length(chunk) == 0L) break

    line_count <- line_count + length(chunk)
    character_lengths <- nchar(chunk, type = "chars", allowNA = TRUE)
    longest_line <- max(longest_line, character_lengths, na.rm = TRUE)

    if (inspect_twitter) {
      # Match the course's grep method: lowercase, case-sensitive substrings.
      love_count <- love_count + sum(grepl("love", chunk, fixed = TRUE))
      hate_count <- hate_count + sum(grepl("hate", chunk, fixed = TRUE))

      found <- grep("biostat", chunk, ignore.case = TRUE, value = TRUE)
      if (length(found) > 0L) biostat_matches <- c(biostat_matches, found)

      exact_sentence_count <- exact_sentence_count +
        sum(chunk == target_sentence, na.rm = TRUE)
    }
  }

  list(
    source = source_name,
    line_count = line_count,
    longest_line = longest_line,
    love_count = love_count,
    hate_count = hate_count,
    biostat_matches = unique(biostat_matches),
    exact_sentence_count = exact_sentence_count
  )
}

scan_official_corpus <- function(paths, chunk_size = 10000L) {
  results <- Map(
    f = function(path, source_name) {
      scan_corpus_file(
        path = path,
        source_name = source_name,
        inspect_twitter = identical(source_name, "twitter"),
        chunk_size = chunk_size
      )
    },
    path = unname(paths),
    source_name = names(paths)
  )
  names(results) <- names(paths)
  results
}

corpus_scan_table <- function(scan_results) {
  rows <- lapply(scan_results, function(result) {
    data.frame(
      source = result$source,
      line_count = result$line_count,
      longest_line = result$longest_line,
      stringsAsFactors = FALSE
    )
  })
  result <- do.call(rbind, rows)
  row.names(result) <- NULL
  result
}
