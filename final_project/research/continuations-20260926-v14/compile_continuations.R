library(data.table);library(jsonlite)
app<-'final_project/phraseflow_multilingual';out<-'final_project/research/continuations-20260926-v14'
cfg<-fromJSON(file.path(app,'language-config.json'),simplifyVector=FALSE)
sizes<-list()
for(code in names(cfg)){
 tab<-fread(file.path(out,paste0(code,'_index.tsv')),encoding='UTF-8');setkey(tab,context)
 dest<-file.path(app,'continuations',paste0(code,'.rds'))
 stopifnot(!file.exists(dest));saveRDS(tab,dest,compress='xz')
 sizes[[code]]<-list(rows=nrow(tab),memory_mib=as.numeric(object.size(tab))/1024^2,file_mib=file.info(dest)$size/1024^2)
}
write_json(sizes,file.path(out,'index_sizes.json'),pretty=TRUE,auto_unbox=TRUE)
