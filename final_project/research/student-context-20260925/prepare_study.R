# Sonja Projects | Author: Sonja Sahebzad
# Run from the corpus project root. Raw text stays in ignored data/.
library(data.table)
library(jsonlite)
setDTthreads(4L)
stopifnot(file.exists("final_project/app/predictor.R"),file.exists("models/expanded_partitions_v3.rds"))
out <- "final_project/research/student-context-20260925"
private <- "data/student_context"
dir.create(out,recursive=TRUE,showWarnings=FALSE)
if(file.exists(file.path(out,"protocol.json"))) stop("Preserve the registered study.")
dir.create(private,recursive=TRUE,showWarnings=FALSE)
p <- list(author="Sonja Sahebzad",registered_utc=format(Sys.time(),tz="UTC",usetz=TRUE),
 objective="Bounded pilot of a compact longer-context word model and complete-word teacher distillation, compared with deployed CPU n-grams.",
 training="90,000 normalized deduplicated training lines, 30,000/source. At most six randomly sampled next-word positions/line. Last 48 words of each prefix. No validation or test lines in gradient updates. One fixed training seed: exploratory pilot, not seed robustness evidence.",
 teacher="Existing Qwen3-1.7B-Base with frozen local adapter; 128 model-token prefix; 64 leading-space single-token proposals plus compact n-gram top32, restricted to the existing 50,000-word vocabulary. Score complete canonical word token sequences plus following-word boundary probability. Never pass actual word or answer options to generation. 3,000 training examples, 1,000/source. Same teacher rule on 600 development cases.",
 student="Trainable 64-dimensional word embeddings, single-layer GRU with 96 hidden units, tanh 96-to-64 projection, tied 50,000-word output embedding and output bias. Padding and unknown input IDs separate. Two supervised pretraining epochs, batch256, AdamW lr0.002, clip1. Then three equal-budget five-epoch continuations on 3,000 teacher examples: CE-only control, 25% distillation, 50% distillation. Fine-tuning lr0.0005. Distillation temperature2, complete-word shortlist distribution padded with zero probability for other output words; T-squared correction. CE ignores unknown targets; soft supervision still applies.",
 comparisons="Current CPU model, teacher reference (GPU, not hosting candidate), three students alone and with current probability interpolation at student weights0.10,0.25,0.50. Same 600 development cases for every comparison. Fixed final epochs only; no adaptive hyperparameter search.",
 selection="Development: at least six more top3 successes than Current, no lower top1, prepared combined R objects <=256MiB, warm local CPU median <20ms and p95<50ms. Select top3 then top1 then median latency then registered order. GPU teacher is ineligible. R export must match Python probabilities within1e-6 and exact top3 on every development case before promotion.",
 final="Only a qualifying frozen winner may access the still-unused 900 cases in data/final_product_quality/reserved_test.csv. Require source-stratified paired bootstrap95% lower bound of top3 difference >0 and no lower top1. Otherwise retain current app. Do not retune on final data. If qualified, fit top1 confidence on a separately reserved validation sample and assess it on final data before displaying it.",
 interactions="GRU learns dependencies between earlier words and recent context. Fixed probability mixtures explicitly test interaction with local n-gram evidence. Supervised control separates longer-context learning from the added distillation loss.",
 limitations="Single seed, bounded training subset and teacher cases, shortlists truncate teacher distribution, unknown pretrained-data overlap, full-line deduplication does not exclude near duplicates. Reused development results are exploratory; no calibrated confidence or 100% accuracy is presumed.",
 timing="Three interleaved warm CPU rounds, same 600 cases, normalization and complete scoring included. GPU teacher timed separately. Loading, typing delay, networking and display excluded. Resource measurements describe R objects, not process RSS.",
 seeds=list(lines=20261005,positions=20261006,training=20261007,timing=20261008,bootstrap=20261009))
write_json(p,file.path(out,"protocol.json"),pretty=TRUE,auto_unbox=TRUE)
file.copy("final_project/app/predictor.R",file.path(out,"predictor_snapshot.R"),overwrite=FALSE)
source(file.path(out,"predictor_snapshot.R"))
model<-prepare_predictor(readRDS("final_project/app/model.rds"))
fwrite(data.table(word_id=seq_along(model$vocabulary)-1L,word=model$vocabulary,
 base_probability=model$base_probability),file.path(private,"vocabulary.csv"))
rows<-readRDS("models/expanded_partitions_v3.rds")
setDT(rows)
eligible<-rows[split=="train" & stringi::stri_count_fixed(text," ")>=2L]
set.seed(p$seeds$lines)
train<-eligible[,.SD[sample.int(.N,30000L)],by=source]
hash<-function(x)vapply(x,digest::digest,character(1),algo="sha256",serialize=FALSE,USE.NAMES=FALSE)
train[,line_hash:=hash(text)]
dev<-fread("data/adaptation/development.csv")[,.(source,line_hash,prefix,actual)]
reserved<-fread("final_project/research/speed-quality-20260925/reserved_hashes.csv")
stopifnot(nrow(train)==90000L,all(train$split=="train"),uniqueN(train$text)==nrow(train),
 !any(train$line_hash %in% c(dev$line_hash,reserved$line_hash)))
fwrite(train[,.(source,line_hash,text)],file.path(private,"training_lines.csv"))
fwrite(train[,.(source,line_hash)],file.path(out,"training_hashes.csv"))
fwrite(dev,file.path(private,"development.csv"))
set.seed(p$seeds$positions)
teacher<-train[,.SD[sample.int(.N,1000L)],by=source]
words<-strsplit(teacher$text," ",fixed=TRUE)
pos<-vapply(words,function(w)sample(2:length(w),1L),integer(1))
teacher[,prefix:=mapply(function(w,k)paste(head(w,k-1L),collapse=" "),words,pos)]
teacher[,actual:=mapply(function(w,k)w[k],words,pos)]
for(kind in c("teacher_training","development")) {
 x<-if(kind=="teacher_training")teacher[,.(source,line_hash,prefix,actual)]else dev
 x[,prefix:=normalize_phrase(prefix)]
 x[,ngram_candidates_json:=vapply(prefix,function(s)toJSON(predict_word(model,s,32L)$words,auto_unbox=FALSE),character(1))]
 fwrite(x,file.path(private,paste0(kind,".csv")))
 if(kind=="development") {
   probs<-vapply(x$prefix,function(s)predict_word(model,s,dense=TRUE)$distribution,numeric(50000L))
   con<-file(file.path(private,"ngram_dev.f32"),"wb");writeBin(as.numeric(probs),con,size=4L);close(con)
 }
}
write_json(list(training_lines=nrow(train),training_by_source=as.data.frame(train[,.N,by=source]),
 development=nrow(dev),teacher_training=nrow(teacher),test_accessed=FALSE,
 no_training_hash_overlap_with_development_or_reserved=TRUE,
 model_md5=unname(tools::md5sum("final_project/app/model.rds")),
 private_file_md5=as.list(tools::md5sum(list.files(private,full.names=TRUE)))),
 file.path(out,"data_manifest.json"),pretty=TRUE,auto_unbox=TRUE)
writeLines(capture.output(sessionInfo()),file.path(out,"R-session-info.txt"))
cat("Prepared 90,000 training lines, 3,000 teacher prefixes and 600 development cases. Reserved test unread.\n")
