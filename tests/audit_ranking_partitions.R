# Independent hash-only checks of calibration/final split membership.
# Author: Sonja Sahebzad. Does not inspect prediction outcomes.
library(data.table)
setDTthreads(4L)
rows <- readRDS("models/expanded_partitions_v3.rds")
stopifnot(uniqueN(rows$text)==nrow(rows))
rows[,row_id:=.I]
eligible <- rows[stringi::stri_count_fixed(text," ")>=2L]
pick <- function(x,n,seed) {set.seed(seed);x[,.SD[sample.int(.N,min(.N,n))],by=source]}
hashes <- function(x) vapply(x,digest::digest,character(1),algo="sha256",serialize=FALSE,USE.NAMES=FALSE)
oldval <- pick(eligible[split=="validation"],600L,20261102L)
oldtest <- pick(eligible[split=="test"],1000L,20261103L)
prior_files <- c("data/neural_evaluation/development.csv","data/neural_evaluation/test.csv",
  "data/adaptation/development.csv","data/adaptation/test.csv",
  "models/learning_curve_reserved_case_hashes.csv","models/word_generation_reserved_case_hashes.csv")
prior <- unique(c(hashes(oldval$text),hashes(oldtest$text),unlist(lapply(prior_files,function(p) fread(p,select="line_hash")$line_hash))))
reserved <- fread("models/ranking_study_reserved_hashes.csv")
stopifnot(nrow(reserved)==1800L,uniqueN(reserved$line_hash)==1800L,
  all(reserved[,.N,by=.(role,source)]$N==300L),!any(reserved$line_hash %chin% prior))
pool <- rows[split %in% c("validation","test")]
matched <- list()
for (start in seq.int(1L,nrow(pool),10000L)) {
  stop <- min(start+9999L,nrow(pool)); h <- hashes(pool$text[start:stop])
  at <- which(h %chin% reserved$line_hash)
  if (length(at)) matched[[length(matched)+1L]] <- data.table(line_hash=h[at],
    source=pool$source[start+at-1L],split=pool$split[start+at-1L])
}
matched <- rbindlist(matched)
stopifnot(nrow(matched)==1800L,uniqueN(matched$line_hash)==1800L)
lookup <- matched[match(reserved$line_hash,line_hash)]
stopifnot(identical(lookup$source,reserved$source),
  all(lookup$split==ifelse(reserved$role=="calibration","validation","test")))
external <- readLines("data/adaptation/taskmaster_training.txt",encoding="UTF-8",warn=FALSE)
local <- readLines("data/adaptation/local_training.txt",encoding="UTF-8",warn=FALSE)
stopifnot(!any(reserved$line_hash %chin% hashes(c(local,external))))
result <- list(author="Sonja Sahebzad",passed=TRUE,reserved_cases=1800L,
  calibration_validation_membership=900L,final_test_membership=900L,
  source_labels_confirmed=TRUE,global_normalized_line_uniqueness=TRUE,
  prior_evaluation_cases=length(prior),prior_overlap=0L,training_overlap=0L,
  original_partition_sha256=digest::digest(file="models/expanded_partitions_v3.rds",algo="sha256"),
  reserved_hashes_sha256=digest::digest(file="models/ranking_study_reserved_hashes.csv",algo="sha256"),
  method="Independent membership reconstruction from partition hashes, without prediction outcomes.",
  limitation="Exact-line checks cannot exclude near duplicates or unknown pretrained-model overlap.")
destination <- "models/ranking_study_partition_audit.json"
if (file.exists(destination)) stopifnot(identical(jsonlite::read_json(destination,simplifyVector=TRUE),result)) else
  jsonlite::write_json(result,destination,auto_unbox=TRUE,pretty=TRUE)
print(result)
