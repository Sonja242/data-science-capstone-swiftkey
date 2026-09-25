# Sonja Projects | Author: Sonja Sahebzad
# Native R evaluation of the exported PyTorch GRU. No Python at prediction time.
load_student <- function(directory) {
  spec<-jsonlite::fromJSON(file.path(directory,"weights.json"),simplifyVector=FALSE)
  weights<-lapply(spec,function(s) {
    shape<-as.integer(unlist(s$shape));con<-file(file.path(directory,s$file),"rb")
    on.exit(close(con))
    x<-readBin(con,"numeric",n=prod(shape),size=4L,endian="little")
    stopifnot(length(x)==prod(shape),all(is.finite(x)))
    if(length(shape)==2L)matrix(x,nrow=shape[1],ncol=shape[2],byrow=TRUE)else x
  })
  stopifnot(identical(dim(weights$embedding.weight),c(50002L,64L)))
  weights
}
student_distribution <- function(student,model,phrase) {
  clean<-normalize_phrase(phrase)
  words<-if(nzchar(clean))tail(strsplit(clean," ",fixed=TRUE)[[1L]],48L)else "<unknown>"
  ids<-model$lookup[list(words),id]+2L
  ids[is.na(ids)]<-2L # row1 padding, row2 unknown, row3 first output word
  h<-numeric(96L)
  for(id in ids) {
    gi<-as.vector(student$gru.weight_ih_l0 %*% student$embedding.weight[id,])+student$gru.bias_ih_l0
    gh<-as.vector(student$gru.weight_hh_l0 %*% h)+student$gru.bias_hh_l0
    r<-plogis(gi[1:96]+gh[1:96]);z<-plogis(gi[97:192]+gh[97:192])
    proposed<-tanh(gi[193:288]+r*gh[193:288])
    h<-(1-z)*proposed+z*h
  }
  projected<-tanh(as.vector(student$projection.weight %*% h)+student$projection.bias)
  logits<-as.vector(student$embedding.weight %*% projected)[-(1:2)]+student$bias
  p<-exp(logits-max(logits));p/sum(p)
}
predict_student <- function(student,model,phrase,student_weight=1) {
  p<-student_distribution(student,model,phrase)
  if(student_weight<1) p<-student_weight*p+(1-student_weight)*predict_word(model,phrase,dense=TRUE)$distribution
  take<-head(order(-p,model$vocabulary,method="radix"),3L)
  list(words=model$vocabulary[take],scores=p[take],distribution=p)
}
