# Manual ratio of complete-word line counts in the SwiftKey Twitter corpus
# Author: Sonja Sahebzad
#
# Run this complete script interactively in RStudio with Source.
# Enter both requested lowercase words yourself when prompted.

source(file.path("R", "text_query_utils.R"))

twitter_file <- file.path(
  "data",
  "final",
  "en_US",
  "en_US.twitter.txt"
)

if (!file.exists(twitter_file)) {
  stop(
    "The official Twitter corpus was not found. Open Data Science Capstone.Rproj first.",
    call. = FALSE
  )
}

if (!interactive()) {
  stop("Run this script interactively in RStudio.", call. = FALSE)
}

numerator_word <- readline(
  prompt = "Type the numerator word exactly, then press Enter: "
)

denominator_word <- readline(
  prompt = "Type the denominator word exactly, then press Enter: "
)

if (!nzchar(numerator_word) || !nzchar(denominator_word)) {
  stop("Both words must be entered.", call. = FALSE)
}

numerator_count <- count_lines_with_word(
  path = twitter_file,
  word = numerator_word,
  case_sensitive = TRUE
)

denominator_count <- count_lines_with_word(
  path = twitter_file,
  word = denominator_word,
  case_sensitive = TRUE
)

ratio <- safe_count_ratio(numerator_count, denominator_count)

results <- data.frame(
  numerator_word = numerator_word,
  numerator_count = numerator_count,
  denominator_word = denominator_word,
  denominator_count = denominator_count,
  ratio = ratio,
  stringsAsFactors = FALSE
)

print(results)
