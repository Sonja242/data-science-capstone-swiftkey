# Manual maximum-line-length check for one SwiftKey source
# Author: Sonja Sahebzad
#
# Run this script once for each source and record the printed result.

source(file.path("R", "text_query_utils.R"))

corpus_files <- c(
  blogs = file.path("data", "final", "en_US", "en_US.blogs.txt"),
  news = file.path("data", "final", "en_US", "en_US.news.txt"),
  twitter = file.path("data", "final", "en_US", "en_US.twitter.txt")
)

missing_files <- corpus_files[!file.exists(corpus_files)]
if (length(missing_files) > 0L) {
  stop(
    "One or more official corpus files are missing. Open Data Science Capstone.Rproj first.",
    call. = FALSE
  )
}

if (!interactive()) {
  stop("Run this script interactively in RStudio.", call. = FALSE)
}

source_name <- tolower(trimws(readline(
  prompt = "Type one source name (blogs, news, or twitter), then press Enter: "
)))

if (!(source_name %in% names(corpus_files))) {
  stop("Choose exactly one of: blogs, news, twitter.", call. = FALSE)
}

maximum_length <- maximum_line_length(corpus_files[[source_name]])

result <- data.frame(
  source = source_name,
  maximum_line_length = maximum_length,
  stringsAsFactors = FALSE
)

print(result)
