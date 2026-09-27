# Sonja PhraseFlow | Sonja Projects | Author: Sonja Sahebzad
# Experimental native R inference. The production app does not load this model.
load_head <- function(directory) {
 spec<-jsonlite::fromJSON(file.path(directory,'weights.json'),simplifyVector=FALSE)
 lapply(spec,function(s){
  shape<-as.integer(unlist(s$shape));con<-file(file.path(directory,s$file),'rb');on.exit(close(con))
  x<-readBin(con,'numeric',n=prod(shape),size=4L,endian='little')
  stopifnot(length(x)==prod(shape),all(is.finite(x)))
  if(length(shape)==2L)matrix(x,nrow=shape[1],ncol=shape[2],byrow=TRUE)else x
 })
}
phraseflow_features <- function(student,model,phrase,prediction) {
 words<-if(nzchar(prediction$normalized))strsplit(prediction$normalized,' ',fixed=TRUE)[[1]]else '<unknown>'
 ids<-model$lookup[list(tail(words,48L)),id]+2L;ids[is.na(ids)]<-2L
 h<-numeric(96L)
 for(id in ids){
  gi<-as.vector(student$gru.weight_ih_l0 %*% student$embedding.weight[id,])+student$gru.bias_ih_l0
  gh<-as.vector(student$gru.weight_hh_l0 %*% h)+student$gru.bias_hh_l0
  r<-plogis(gi[1:96]+gh[1:96]);z<-plogis(gi[97:192]+gh[97:192])
  h<-(1-z)*tanh(gi[193:288]+r*gh[193:288])+z*h
 }
 context<-tanh(as.vector(student$projection.weight %*% h)+student$projection.bias)
 candidate_ids<-model$lookup[list(prediction$words),id]
 e<-student$embedding.weight[candidate_ids+2L,,drop=FALSE]
 contexts<-matrix(context,nrow=nrow(e),ncol=64L,byrow=TRUE)
 scalar<-cbind(log(prediction$scores),log(model$base_probability[candidate_ids]),
  rowSums(e*contexts)+student$bias[candidate_ids],log1p(nchar(prediction$words)),
  as.numeric((candidate_ids+2L) %in% ids),rep(log1p(length(words)),nrow(e)))
 cbind(contexts,e,contexts*e,scalar)
}
predict_phraseflow <- function(model,student,head,phrase,top_n=3L) {
 p<-predict_word(model,phrase,128L)
 f<-phraseflow_features(student,model,phrase,p)
 f<-sweep(sweep(f,2L,head$feature_mean,'-'),2L,head$feature_scale,'/')
 hidden<-tanh(sweep(f %*% t(head$hidden.weight),2L,head$hidden.bias,'+'))
 score<-log(p$scores)+as.vector(hidden %*% t(head$output.weight))+head$output.bias
 idx<-head(order(-score,p$words),top_n)
 list(words=p$words[idx],ranking_scores=score[idx],all_scores=score,candidates=p$words,
      input_words=p$input_words,context_limit=48L,normalized=p$normalized)
}
