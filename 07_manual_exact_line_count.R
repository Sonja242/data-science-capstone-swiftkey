# Manual exact-line count in the official SwiftKey Twitter corpus
# Author: Sonja Sahebzad
#
# Run this complete script interactively in RStudio with Source.
# Paste the requested complete line yourself when prompted.

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

target_line <- readline(
  prompt = "Paste the complete target line exactly, then press Enter: "
)

if (!nzchar(target_line)) {
  stop("No target line was entered.", call. = FALSE)
}

exact_count <- count_exact_lines(
  path = twitter_file,
  target = target_line
)

cat("\nNumber of exact complete-line matches:", exact_count, "\n")
