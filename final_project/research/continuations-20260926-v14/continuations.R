# Sonja Projects | Short endings observed in existing training sentences.
# This helper does not change the next-word model or its probabilities.
predict_continuations<-function(index,phrase,tokenizer,exclude=character(),limit=3L){
 empty<-list(text=character(),context='',context_words=0L)
 if(is.null(index)||!nrow(index))return(empty)
 clean<-normalize_phrase(phrase,tokenizer)
 words<-if(nzchar(clean))strsplit(clean,' ',fixed=TRUE)[[1L]] else character()
 if(length(words)<2L)return(empty)
 for(k in seq.int(min(4L,length(words)),2L)){
  matched_context<-paste(tail(words,k),collapse=' ')
  hits<-data.table::copy(index[list(matched_context),nomatch=0L])
  if(!nrow(hits))next
  # Two-word suffixes give weak context: show one word, not an assumed full ending.
  if(k==2L){hits[,continuation:=sub(' .*','',continuation)];hits<-hits[,.(count=sum(count)),by=continuation]}
  hits<-hits[!continuation %in% exclude]
  if(!nrow(hits))return(empty)
  data.table::setorderv(hits,c('count','continuation'),c(-1L,1L))
  take<-head(hits$continuation,limit)
  if(tokenizer=='chinese')take<-gsub(' ','',take,fixed=TRUE)
  return(list(text=take,context=matched_context,context_words=k))
 }
 empty
}
