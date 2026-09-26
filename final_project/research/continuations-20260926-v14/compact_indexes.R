library(data.table);library(jsonlite)
out<-'final_project/research/continuations-20260926-v14';app<-'final_project/phraseflow_multilingual'
cfg<-fromJSON(file.path(app,'language-config.json'),simplifyVector=FALSE);sizes<-list()
# Discard contexts with only one stored occurrence, before any behavioural check.
for(code in names(cfg)){
 tab<-fread(file.path(out,paste0(code,'_index.tsv')),encoding='UTF-8')
 tab<-tab[context %in% tab[,.(support=sum(count)),by=context][support>=2L,context]]
 setkey(tab,context);dest<-file.path(app,'continuations',paste0(code,'.rds'))
 saveRDS(tab,dest,compress='gzip')
 sizes[[code]]<-list(rows=nrow(tab),memory_mib=as.numeric(object.size(tab))/1024^2,file_mib=file.info(dest)$size/1024^2)
 print(c(code=code,rows=nrow(tab),memory_mib=sizes[[code]]$memory_mib))
}
write_json(sizes,file.path(out,'index_sizes.json'),pretty=TRUE,auto_unbox=TRUE)
cfg<-fromJSON(file.path(out,'protocol.json'))
cfg$pruning<-'Retain up to eight endings per context by count, lexical tie-break. Discard contexts with less than two retained training occurrences; gzip storage for quick loading.'
write_json(cfg,file.path(out,'protocol.json'),pretty=TRUE,auto_unbox=TRUE)
