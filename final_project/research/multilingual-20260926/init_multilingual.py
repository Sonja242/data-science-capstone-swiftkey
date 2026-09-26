from pathlib import Path
import shutil
ROOT=Path.cwd();OLD=ROOT/'final_project/phraseflow_preview';APP=ROOT/'final_project/phraseflow_multilingual'
if APP.exists():raise FileExistsError('Preserve the existing multilingual preview')
shutil.copytree(OLD,APP,ignore=shutil.ignore_patterns('.Rproj.user','.Rhistory','.RData','*.html','*.md','*.Rpres','pitch.css'))
source=(APP/'predictor.R').read_text(encoding='utf-8')
source=source.replace('normalize_phrase <- function(text) {','normalize_phrase <- function(text, tokenizer = "english") {\n  if (!tokenizer %in% c("english", "unicode")) stop("Unsupported tokenizer")\n  if (tokenizer == "unicode") text <- stringi::stri_trans_nfc(text)')
needle='  words <- stringi::stri_extract_all_regex(text, "[a-z]+(?:\'[a-z]+)*", omit_no_match = TRUE)'
replacement='  pattern <- if (tokenizer == "english") "[a-z]+(?:\'[a-z]+)*" else "[\\\\p{L}][\\\\p{L}\\\\p{M}]*(?:\'[\\\\p{L}][\\\\p{L}\\\\p{M}]*)*"\n  words <- stringi::stri_extract_all_regex(text, pattern, omit_no_match = TRUE)'
assert needle in source;source=source.replace(needle,replacement)
source=source.replace('  clean <- normalize_phrase(phrase)','  clean <- normalize_phrase(phrase, if (is.null(model$tokenizer)) "english" else model$tokenizer)')
(APP/'predictor.R').write_text(source,encoding='utf-8')
shutil.copy2(OLD/'pitch.css',APP/'pitch.css')
(APP/'README.md').write_text('# Sonja PhraseFlow multilingual preview\n\nAuthor: Sonja Sahebzad. Sonja Projects.\n\nLocal experimental extension. English model unchanged. Other models require separate validation.\n',encoding='utf-8')
print('Separate app created. English model preserved. Unicode tokenizer added.')
