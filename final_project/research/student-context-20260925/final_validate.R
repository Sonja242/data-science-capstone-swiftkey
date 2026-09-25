# Sonja Projects | Author: Sonja Sahebzad
# Conditional, single final comparison. Refuses to run without a qualifying model.
library(data.table)
library(jsonlite)
out<-"final_project/research/student-context-20260925"
private<-"data/student_context"
decision<-fromJSON(file.path(out,"decision.json"))
stopifnot(decision$selected!="Current",!decision$final_test_used,
 file.exists(file.path(out,"final_validation_plan.json")))
if(file.exists(file.path(out,"final_summary.json")))stop("Preserve the completed final test.")
if(!file.exists(file.path(out,"development_decision.json")))file.copy(file.path(out,"decision.json"),file.path(out,"development_decision.json"))
source(file.path(out,"predictor_snapshot.R"))
source(file.path(out,"student_predictor.R"))
cfg<-fread(file.path(out,"development_comparison.csv"))[name==decision$selected]
stopifnot(nrow(cfg)==1L,cfg$qualified)
model<-prepare_predictor(readRDS("final_project/app/model.rds"))
student<-load_student(file.path(private,cfg$family))
hash<-function(x)vapply(x,digest::digest,character(1),algo="sha256",serialize=FALSE,USE.NAMES=FALSE)
pool<-readRDS("models/expanded_partitions_v3.rds");setDT(pool);pool[,row_id:=.I]
pool<-pool[split=="validation" & stringi::stri_count_fixed(text," ")>=2L]
set.seed(20261102L);old<-pool[,.SD[sample.int(.N,600L)],by=source]
pool<-pool[!row_id %in% old$row_id]
excluded<-unique(unlist(lapply(c("data/neural_evaluation/development.csv","data/adaptation/development.csv","data/ranking_study/calibration.csv"),function(p)fread(p)$line_hash)))
set.seed(20261010L);candidates<-pool[,.SD[sample.int(.N,1500L)],by=source]
candidates[,line_hash:=hash(text)]
cal<-candidates[!line_hash %chin% excluded][,head(.SD,300L),by=source]
stopifnot(nrow(cal)==900L,!any(cal$line_hash %chin% fread(file.path(out,"training_hashes.csv"))$line_hash))
set.seed(20261011L);ww<-strsplit(cal$text," ",fixed=TRUE)
pos<-vapply(ww,function(w)sample(2:length(w),1L),integer(1))
cal[,prefix:=mapply(function(w,k)paste(head(w,k-1L),collapse=" "),ww,pos)]
cal[,actual:=mapply(function(w,k)w[k],ww,pos)]
fwrite(cal[,.(source,line_hash,prefix,actual)],file.path(private,"calibration.csv"))
fwrite(cal[,.(source,line_hash)],file.path(out,"calibration_hashes.csv"))
rm(pool,candidates,old,ww);gc(FALSE)
run<-function(x)rbindlist(lapply(seq_len(nrow(x)),function(i) {
 a<-predict_word(model,x$prefix[i]);b<-predict_student(student,model,x$prefix[i],cfg$weight)
 rbindlist(lapply(c("Current","Challenger"),function(which) {
  p<-if(which=="Current")a else b
  data.table(method=which,source=x$source[i],line_hash=x$line_hash[i],rank=match(x$actual[i],p$words,nomatch=0L),score=p$scores[1L])
 }))
}))
cal_detail<-run(cal)
fit<-function(x) {
 z<-qlogis(pmin(pmax(x$score,1e-6),1-1e-6));y<-as.numeric(x$rank==1L)
 loss<-function(par) {
  prob<-pmin(pmax(plogis(par[1]+par[2]*z),1e-12),1-1e-12)
  -mean(y*log(prob)+(1-y)*log1p(-prob))+.0005*par[2]^2
 }
 opt<-optim(c(qlogis(mean(y)),0),loss,method="L-BFGS-B",lower=c(-Inf,0))
 stopifnot(opt$convergence==0L);as.list(setNames(opt$par,c("intercept","slope")))
}
maps<-lapply(split(cal_detail,cal_detail$method),fit)
write_json(list(selected=decision$selected,maps=maps,
 checkpoint_md5=unname(tools::md5sum(file.path(private,paste0(cfg$family,".pt"))))),file.path(out,"frozen_calibration.json"),pretty=TRUE,auto_unbox=TRUE,digits=12)
fwrite(cal_detail,file.path(out,"calibration_cases.csv"))
# The final file is first opened only after the model and calibration are frozen.
test<-fread("data/final_product_quality/reserved_test.csv")
stopifnot(nrow(test)==900L,!any(test$line_hash %chin% cal$line_hash))
pred<-run(test)
pred[,confidence:=vapply(seq_len(.N),function(i){m<-maps[[method[i]]];plogis(m$intercept+m$slope*qlogis(pmin(pmax(score[i],1e-6),1-1e-6)))},numeric(1))]
fwrite(pred,file.path(out,"final_cases.csv"))
wilson<-function(k,n) {
 z<-qnorm(.975);p<-k/n;den<-1+z*z/n
 mid<-(p+z*z/(2*n))/den;half<-z*sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 c(mid-half,mid+half)
}
metrics<-pred[,.(cases=.N,top1=sum(rank==1),top3=sum(rank>0),brier=mean((confidence-as.numeric(rank==1))^2)),by=method]
metrics[,c("top1_low","top1_high"):=as.list(wilson(top1,cases)),by=method]
metrics[,c("top3_low","top3_high"):=as.list(wilson(top3,cases)),by=method]
a<-pred[method=="Current"];b<-pred[method=="Challenger"]
pair<-merge(a,b,by=c("source","line_hash"),suffixes=c("_old","_new"))
diff<-as.integer(pair$rank_new>0)-as.integer(pair$rank_old>0);groups<-split(seq_len(nrow(pair)),pair$source)
set.seed(20261009);boot<-replicate(10000L,mean(diff[unlist(lapply(groups,function(i)sample(i,length(i),TRUE)))]))
interval<-quantile(boot,c(.025,.975));gained<-sum(diff==1);lost<-sum(diff== -1)
pvalue<-if(gained+lost)min(1,2*pbinom(min(gained,lost),gained+lost,.5))else 1
promote<-interval[1]>0 && metrics[method=="Challenger",top1]>=metrics[method=="Current",top1]
pred[,bin:=pmin(9L,floor(confidence*10))]
reliability<-pred[,.(n=.N,mean_confidence=mean(confidence),observed_accuracy=mean(rank==1)),by=.(method,bin)]
ece<-reliability[,.(ece=sum(n*abs(mean_confidence-observed_accuracy))/sum(n)),by=method]
metrics<-merge(metrics,ece,by="method")
fwrite(reliability,file.path(out,"reliability.csv"));fwrite(metrics,file.path(out,"final_metrics.csv"))
thresholds<-rbindlist(lapply(c(0,.5,.7,.85,.95),function(t)pred[,{
 x<-.SD[confidence>=t];ci<-if(nrow(x))wilson(sum(x$rank==1),nrow(x))else c(NA_real_,NA_real_)
 list(threshold=t,answered=nrow(x),coverage=nrow(x)/.N,top1=if(nrow(x))mean(x$rank==1)else NA_real_,top3=if(nrow(x))mean(x$rank>0)else NA_real_,top1_low=ci[1],top1_high=ci[2])
},by=method]))
fwrite(thresholds,file.path(out,"confidence_thresholds.csv"))
write_json(list(selected=decision$selected,top3_difference=mean(diff),difference_interval=unname(interval),gained=gained,lost=lost,secondary_p=pvalue,promote=unname(promote)),file.path(out,"final_summary.json"),pretty=TRUE,auto_unbox=TRUE,digits=10)
decision$development_only<-FALSE;decision$final_test_used<-TRUE;decision$confidence_calibrated<-TRUE
decision$final_qualified<-unname(promote)
decision$reason<-if(promote)"Frozen candidate passed the independent final gate. Deployment integration remains to be verified."else"Independent final gate failed. Keep the deployed model and do not retune on these test cases."
write_json(decision,file.path(out,"decision.json"),pretty=TRUE,auto_unbox=TRUE)
print(metrics);cat("Independent promotion gate:",promote,"\n")
