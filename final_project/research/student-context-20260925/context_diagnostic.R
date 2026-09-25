library(data.table)
library(jsonlite)
source("final_project/research/student-context-20260925/predictor_snapshot.R")
source("final_project/research/student-context-20260925/student_predictor.R")
out<-"final_project/research/student-context-20260925"
stopifnot(file.exists(file.path(out,"context_diagnostic_plan.json")))
model<-prepare_predictor(readRDS("final_project/app/model.rds"))
dev<-fread("data/student_context/development.csv")
result<-list()
for(f in c("CE_control","Distill25","Distill50")) {
 student<-load_student(file.path("data/student_context",f))
 details<-rbindlist(lapply(seq_len(nrow(dev)),function(i) {
   phrase<-dev$prefix[i]
   short<-paste(tail(strsplit(phrase," ",fixed=TRUE)[[1L]],4L),collapse=" ")
   a<-predict_student(student,model,phrase)$words
   b<-predict_student(student,model,short)$words
   data.table(family=f,source=dev$source[i],line_hash=dev$line_hash[i],
    full_rank=match(dev$actual[i],a,nomatch=0L),short_rank=match(dev$actual[i],b,nomatch=0L),
    changed=!identical(a,b),longer_input=length(strsplit(phrase," ",fixed=TRUE)[[1L]])>4L)
 }))
 result[[f]]<-details
}
d<-rbindlist(result);fwrite(d,file.path(out,"context_diagnostic_cases.csv"))
summary<-d[,.(cases=.N,longer_inputs=sum(longer_input),changed_lists=sum(changed),
  full_top1=sum(full_rank==1),full_top3=sum(full_rank>0),
  last4_top1=sum(short_rank==1),last4_top3=sum(short_rank>0),
  gained=sum(full_rank>0 & short_rank==0),lost=sum(full_rank==0 & short_rank>0)),by=family]
fwrite(summary,file.path(out,"context_diagnostic.csv"));print(summary)
