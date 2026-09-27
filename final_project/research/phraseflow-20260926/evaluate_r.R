# Native R CPU parity, complete inference timing and memory. No test/quiz access.
library(data.table);library(jsonlite);setDTthreads(1L)
out<-'final_project/research/phraseflow-20260926';private<-'data/phraseflow_20260926'
if(file.exists(file.path(out,'cpu_checks.json')))stop('Preserve completed CPU evaluation.')
source('final_project/app/predictor.R')
source('final_project/research/student-context-20260925/student_predictor.R')
source(file.path(out,'reranker.R'))
model<-prepare_predictor(readRDS('final_project/app/model.rds'))
student<-load_student(file.path(private,'export/encoder'));head<-load_head(file.path(private,'export/head'))
dev<-fread('data/student_context/development.csv')[,.(source,line_hash,prefix,actual)]
con<-file(file.path(private,'selected_development_scores.f32'),'rb');expected<-matrix(readBin(con,'numeric',600L*128L,size=4L,endian='little'),nrow=600,byrow=TRUE);close(con)
con<-file(file.path(private,'selected_development_ids.i32'),'rb');expected_ids<-matrix(readBin(con,'integer',600L*3L,size=4L,endian='little'),nrow=600,byrow=TRUE);close(con)
observed<-vector('list',600);errors<-numeric(600);same<-logical(600)
invisible(predict_phraseflow(model,student,head,'a neutral example before timing'))
for(i in seq_len(nrow(dev))){
 start<-as.numeric(Sys.time());p<-predict_phraseflow(model,student,head,dev$prefix[i]);ms<-(as.numeric(Sys.time())-start)*1000
 errors[i]<-max(abs(p$all_scores-expected[i,]));same[i]<-identical(unname(match(p$words,model$vocabulary)-1L),as.integer(expected_ids[i,]))
 rank<-match(dev$actual[i],p$words,nomatch=0L)
 observed[[i]]<-data.table(case_id=i,source=dev$source[i],line_hash=dev$line_hash[i],rank=rank,top1=rank==1L,top3=rank>0L,
  milliseconds=ms,max_score_difference=errors[i],same_top3=same[i])
}
stopifnot(all(same),max(errors)<1e-3)
cases<-rbindlist(observed);fwrite(cases,file.path(out,'cpu_development_cases.csv'))
set.seed(20260926L);timing<-vector('list',1800L);j<-0L
for(round in 1:3)for(i in sample(seq_len(600L))){
 t<-as.numeric(Sys.time());p<-predict_phraseflow(model,student,head,dev$prefix[i]);elapsed<-(as.numeric(Sys.time())-t)*1000;j<-j+1L
 timing[[j]]<-data.table(round=round,case_id=i,milliseconds=elapsed)
}
timing<-rbindlist(timing);fwrite(timing,file.path(out,'cpu_timing.csv'))
size<-as.numeric(object.size(model)+object.size(student)+object.size(head))/1024^2
med<-median(timing$milliseconds);p95<-unname(quantile(timing$milliseconds,.95))
result<-list(all600_top3_match=all(same),maximum_score_difference=max(errors),cases=600L,
 top1=sum(cases$top1),top3=sum(cases$top3),combined_objects_mib=size,
 median_ms=med,p95_ms=p95,timed_calls=1800L,device='native R CPU',
 timing_scope='CPU128 generation, normalized context encoding and head scoring. Three shuffled warm rounds; no network or UI time.',
 cpu_gate_passed=med<20 && p95<50 && size<=256,reserved_test_opened=FALSE,production_changed=FALSE)
write_json(result,file.path(out,'cpu_checks.json'),auto_unbox=TRUE,pretty=TRUE,digits=16)
selection<-fromJSON(file.path(out,'selection.json'))
selection$cpu_gate_passed<-result$cpu_gate_passed
selection$development_gates_passed<-selection$quality_gate_passed && result$cpu_gate_passed
selection$production_decision<-'Retain current production model. This study has no independent test of the new candidate.'
write_json(selection,file.path(out,'decision.json'),auto_unbox=TRUE,pretty=TRUE,digits=16)
writeLines(capture.output(sessionInfo()),file.path(out,'R-session-info.txt'))
print(result)
