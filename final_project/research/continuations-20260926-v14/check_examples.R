library(data.table);library(jsonlite)
app<-'final_project/phraseflow_multilingual';out<-'final_project/research/continuations-20260926-v14'
source(file.path(app,'predictor.R'));source(file.path(app,'continuations.R'))
index<-readRDS(file.path(app,'continuations/nl_NL.rds'))
m<-prepare_predictor(readRDS(file.path(app,'languages/nl_NL.rds')))
examples<-list()
for(s in c('we houden van','we leven lang en','ik wens je een','dank je wel voor','ik kijk uit naar','mag ik een kopje','we zijn blij dat')){
 p<-predict_word(m,s,20L);c<-predict_continuations(index,s,'unicode',exclude=head(p$words,10L),limit=3L)
 examples[[s]]<-list(first=p$words[1],alternatives=p$words[2:10],continuations=c)
 print(list(phrase=s,first=p$words[1],continuations=c))
}
write_json(examples,file.path(out,'demonstration_examples.json'),pretty=TRUE,auto_unbox=TRUE)
