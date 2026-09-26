args <- commandArgs(trailingOnly=TRUE)
root <- 'C:/Users/csj50/OneDrive/Documents/Sonja report/Data Science Capstone'
pub <- file.path(root,'final_project/publication-20260927')
app <- file.path(root,'final_project/phraseflow_multilingual')
Sys.setenv(RSCONNECT_FORCE_GET=TRUE)
if(identical(args[1],'app')) {
  rsconnect::deployApp(appDir=app,appFiles=readLines(file.path(pub,'app-files.txt')),
    appName='sonja-next-word',appId='18010841',appTitle='Sonja PhraseFlow',
    account='sonjasahebzad',server='shinyapps.io',appMode='shiny',
    launch.browser=FALSE,forceUpdate=TRUE)
} else if(identical(args[1],'portfolio')) {
  rsconnect::deployApp(appDir=file.path(dirname(root),'sonja-data-science-portfolio'),
    appFiles=readLines(file.path(pub,'portfolio-files.txt')),
    appName='data-science-portfolio',appId='17941009',
    account='sonjasahebzad',server='shinyapps.io',appMode='shiny',
    launch.browser=FALSE,forceUpdate=TRUE)
} else if(identical(args[1],'rpubs')) {
  previous <- jsonlite::fromJSON(file.path(root,'final_project/rsconnect/pitch_update_receipt.json'))
  receipt <- rsconnect::rpubsUpload(title='Sonja PhraseFlow',
    contentFile=file.path(app,'PhraseFlow_Pitch.html'),
    originalDoc=file.path(app,'PhraseFlow_Pitch.Rpres'),id=previous$id)
  if(!is.null(receipt$error))stop('RPubs update failed; inspect the response locally.')
  # The update identifier is a credential; preserve it only in ignored local records.
  jsonlite::write_json(receipt,file.path(root,'final_project/rsconnect/phraseflow_v14_receipt.json'),auto_unbox=TRUE,pretty=TRUE)
  cat('RPubs update accepted; public display still requires verification.\n')
} else stop('Expected app, portfolio or rpubs.')
