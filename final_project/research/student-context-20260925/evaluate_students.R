# Sonja Projects | Author: Sonja Sahebzad
library(data.table)
library(jsonlite)
setDTthreads(1L)
out<-"final_project/research/student-context-20260925"
private<-"data/student_context"
stopifnot(file.exists(file.path(out,"protocol.json")))
if(file.exists(file.path(out,"decision.json")))stop("Preserve the completed comparison.")
source(file.path(out,"predictor_snapshot.R"))
source(file.path(out,"student_predictor.R"))
model<-prepare_predictor(readRDS("final_project/app/model.rds"))
stopifnot(unname(tools::md5sum("final_project/app/model.rds"))==fromJSON(file.path(out,"data_manifest.json"))$model_md5)
dev<-fread(file.path(private,"development.csv"))
families<-c("CE_control","Distill25","Distill50")
students<-setNames(lapply(families,function(x)load_student(file.path(private,x))),families)
config<-rbindlist(c(list(data.table(name="Current",family="Current",weight=0)),
 lapply(families,function(f)data.table(name=paste0(f,"_",c("0.1","0.25","0.5","1.0")),family=f,weight=c(.1,.25,.5,1)))))
public<-list();parity<-list();mem<-sapply(students,function(x)as.numeric(object.size(x))/1024^2)
get_prediction<-function(cfg,phrase) {
 if(cfg$family=="Current")predict_word(model,phrase)else predict_student(students[[cfg$family]],model,phrase,cfg$weight)
}
for(f in families) {
 con<-file(file.path(private,paste0(f,"_dev.f32")),"rb")
 py<-matrix(readBin(con,"numeric",n=600*50000,size=4L,endian="little"),nrow=50000L);close(con)
 error<-0;identical_top3<-TRUE
 for(i in seq_len(nrow(dev))) {
   p<-student_distribution(students[[f]],model,dev$prefix[i])
   error<-max(error,max(abs(p-py[,i])))
   identical_top3<-identical_top3 && identical(head(order(-p,model$vocabulary),3L),head(order(-py[,i],model$vocabulary),3L))
 }
 parity[[f]]<-list(max_probability_error=error,exact_top3=identical_top3,cases=600L,
   passed=isTRUE(identical_top3) && error<1e-6)
 cat(f,"R export max error",error,"same top3",identical_top3,"\n")
}
write_json(parity,file.path(out,"export_checks.json"),pretty=TRUE,auto_unbox=TRUE,digits=12)
for(j in seq_len(nrow(config))) {
 cfg<-config[j]
 result<-rbindlist(lapply(seq_len(nrow(dev)),function(i) {
  pred<-get_prediction(cfg,dev$prefix[i])
  data.table(name=cfg$name,source=dev$source[i],line_hash=dev$line_hash[i],
   rank=match(dev$actual[i],pred$words,nomatch=0L))
 }))
 public[[j]]<-result
 cat(cfg$name,":",sum(result$rank==1L),"first,",sum(result$rank>0L),"top3\n")
}
detail<-rbindlist(public);fwrite(detail,file.path(out,"development_cases.csv"))
# Comparison is exact to the unchanged sparse production implementation.
old<-fread("final_project/research/dictionary-check-20260925/development_predictions.csv")
base<-detail[name=="Current"]
stopifnot(sum(base$rank==1L)==106L,sum(base$rank>0L)==164L)
set.seed(20261008)
schedule<-CJ(round=1:3,case=seq_len(nrow(dev)))
schedule<-schedule[sample.int(.N)]
timings<-vector("list",nrow(schedule))
cat("Benchmark: three randomized interleaved rounds, 1800 calls per rule\n")
for(k in seq_len(nrow(schedule))) {
 row<-schedule[k]
 order<-sample.int(nrow(config))
 timings[[k]]<-rbindlist(lapply(order,function(j) {
   cfg<-config[j];t0<-as.numeric(Sys.time())
   pred<-get_prediction(cfg,dev$prefix[row$case])
   elapsed<-(as.numeric(Sys.time())-t0)*1000
   data.table(name=cfg$name,round=row$round,case=row$case,milliseconds=elapsed)
 }))
 if(k%%300L==0L)cat("Benchmark",k,"/",nrow(schedule),"\n")
}
timing<-rbindlist(timings);fwrite(timing,file.path(out,"timing.csv"))
summary<-detail[,.(cases=.N,top1=sum(rank==1L),top3=sum(rank>0L)),by=name]
times<-timing[,.(median_ms=median(milliseconds),p95_ms=unname(quantile(milliseconds,.95))),by=name]
summary<-merge(config,merge(summary,times,by="name"),by="name",sort=FALSE)
summary[,prepared_mib:=as.numeric(object.size(model))/1024^2]
summary[family!="Current",prepared_mib:=prepared_mib+mem[family]]
summary[,export_passed:=family=="Current" | vapply(family,function(f)if(f=="Current")TRUE else parity[[f]]$passed,logical(1))]
summary[,qualified:=name!="Current" & top3>=170L & top1>=106L & median_ms<20 & p95_ms<50 & prepared_mib<=256 & export_passed]
fwrite(summary,file.path(out,"development_comparison.csv"))
eligible<-summary[qualified==TRUE]
if(nrow(eligible)){setorder(eligible,-top3,-top1,median_ms);chosen<-eligible$name[1L]}else chosen<-"Current"
write_json(list(author="Sonja Sahebzad",selected=chosen,development_only=TRUE,final_test_used=FALSE,
 production_changed=FALSE,confidence_calibrated=FALSE,
 qualifying_candidates=nrow(eligible),model_md5=unname(tools::md5sum("final_project/app/model.rds")),
 reason=if(chosen=="Current")"No CPU candidate passed the predeclared accuracy, timing, memory and export gates. Keep production; reserved test unread."else"Freeze this development winner before the one permitted independent final comparison."),
 file.path(out,"decision.json"),pretty=TRUE,auto_unbox=TRUE)
print(summary)
