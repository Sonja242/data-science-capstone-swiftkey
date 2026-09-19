# Data Science Capstone: SwiftKey Corpus Analysis

**Author:** Sonja Sahebzad  
**Tools:** R, R Markdown, base R text processing

This repository contains a reproducible analysis workflow for the Johns Hopkins University Data Science Capstone. It explores the official English SwiftKey corpus and demonstrates memory-conscious methods for working with large text files.


## View the finished reports

Open the published [Data Science Capstone report website](https://sonja242.github.io/data-science-capstone-swiftkey/). The site renders the knitted HTML reports as complete web pages; the `.Rmd` files in this repository remain available as reproducible source code.
## What the project demonstrates

- locating and validating the official corpus files;
- inspecting file sizes and line counts;
- processing large text files in chunks;
- measuring maximum line lengths;
- searching for words with explicit case and boundary rules;
- counting exact complete-line matches;
- tokenization and text cleaning;
- producing reproducible results from R Markdown.

## Main reports

| File | Purpose |
|---|---|
| `01_getting_started.Rmd` | Initial corpus inspection and project setup |
| `03_tokenization_cleaning_case_v4.Rmd` | Tokenization and cleaning case study |
| `04_professional_corpus_pipeline.Rmd` | Reusable, memory-safe corpus pipeline |
| `10_quiz1_reproducible_verification.Rmd` | Reproducible verification of the introductory corpus questions |

The knitted `.html` files preserve the corresponding code and printed results.

## Reproduce the analysis

1. Download the official [Coursera SwiftKey dataset](https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip).
2. Extract it so the English files are available under `data/final/en_US/`.
3. Open `Data Science Capstone.Rproj` in RStudio.
4. Open an `.Rmd` report and choose **Knit**.

The corpus and ZIP archive are intentionally excluded from Git because they are large and should be obtained from the official course source.

## Reusable R code

The `R/` directory contains helper functions for chunked corpus processing and exact text verification. These functions avoid loading the complete corpus into memory when that is unnecessary.

## Notes

Results depend on the official Coursera corpus version. The reports state matching assumptions such as case sensitivity, word boundaries, and exact equality so the analysis can be checked and reproduced.

