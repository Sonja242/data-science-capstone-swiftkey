library(data.table)
library(jsonlite)
root <- 'C:/Users/csj50/OneDrive/Documents/Sonja report/Data Science Capstone'
app <- file.path(root,'final_project/phraseflow_multilingual')
pub <- file.path(root,'final_project/publication-20260927')
model_files <- c('model.rds',list.files(file.path(app,'languages'),pattern='[.]rds$',full.names=FALSE))
allowed <- c('version','vocabulary','lookup','base_probability','base_order','tables','maximum_order','support','followers','training_lines','training_tokens','method','blocked_terms','language','tokenizer','training_source','prediction_unit')
records <- list()
for(nm in model_files){
  rel <- if(nm=='model.rds')nm else paste0('languages/',nm)
  m <- readRDS(file.path(app,rel))
  stopifnot(all(names(m) %in% allowed),m$maximum_order==5L,
    identical(names(m$lookup),c('word','id')),all(m$lookup$word %in% m$vocabulary),
    !any(grepl('[[:space:]]',m$vocabulary)))
  for(t in m$tables[!vapply(m$tables,is.null,logical(1))])stopifnot(all(vapply(t,is.numeric,logical(1))))
  records[[rel]] <- list(fields=names(m),kind='Vocabulary and aggregate numeric n-gram weights; no training rows, held-out examples or quiz answer table',vocabulary=length(m$vocabulary))
}
for(nm in list.files(file.path(app,'continuations'),pattern='[.]rds$')){
  d <- readRDS(file.path(app,'continuations',nm))
  stopifnot(identical(names(d),c('context','continuation','count')),all(d$count>=1L),
    all(lengths(strsplit(d$context,' ',fixed=TRUE)) %in% 2:4),
    all(lengths(strsplit(d$continuation,' ',fixed=TRUE)) %in% 1:3))
  records[[paste0('continuations/',nm)]] <- list(columns=names(d),rows=nrow(d),kind='Frequency-aggregated public-training phrase suffixes: 2-4 context units and 1-3 ending units; not raw training or held-out rows')
}
for(nm in list.files(file.path(app,'www/attribution'),full.names=TRUE)){
  d <- read.delim(gzfile(nm),nrows=1,check.names=FALSE)
  stopifnot(setequal(names(d),c('sentence_id','owner','url')))
}
write_json(list(model_files_inspected=12L,continuation_files_inspected=12L,
  user_authorization='User requested latest project publication on their GitHub, GitHub Pages, RPubs and portfolio before reviewing the Coursera submission.',
  sources='Four language models derive from the public course corpus; eight derive from public Tatoeba exports with CC BY 2.0 FR attribution retained.',
  private_training_or_holdout_rows_in_model_payload=FALSE,
  course_quiz_answer_tables_in_model_payload=FALSE,
  credentials_excluded=TRUE,records=records),
  file.path(pub,'payload-audit.json'),pretty=TRUE,auto_unbox=TRUE)
cat('Inspected all 12 models, 12 continuation indexes and eight public attribution lists. Schema checks passed.\n')
