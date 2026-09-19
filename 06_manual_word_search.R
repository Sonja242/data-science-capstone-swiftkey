# Manual word search in the official SwiftKey Twitter corpus
# Author: Sonja Sahebzad
#
# Run this complete script interactively in RStudio with Source.
# Type the requested word yourself when the Console asks for it.

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

search_word <- readline(
  prompt = "Type the requested word exactly, then press Enter: "
)

if (!nzchar(search_word)) {
  stop("No search word was entered.", call. = FALSE)
}

matches <- find_lines_with_word(
  path = twitter_file,
  word = search_word,
  case_sensitive = TRUE
)

cat("\nNumber of matching lines returned:", length(matches), "\n\n")
print(matches)
