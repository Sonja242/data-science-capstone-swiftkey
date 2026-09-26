from pathlib import Path
import json
root=Path.cwd();app=root/'final_project/phraseflow_multilingual';out=root/'final_project/research/languages-20260926-v13'
p=app/'predictor.R';s=p.read_text(encoding='utf-8')
s=s.replace('c("english", "unicode")','c("english", "unicode", "chinese")')
s=s.replace('if (tokenizer == "unicode") text <-','if (tokenizer != "english") text <-')
needle='  pattern <- if (tokenizer == "english")'
pos=s.index(needle)
s=s[:pos]+'''  if (tokenizer == "chinese") {
    units <- stringi::stri_split_boundaries(text, type = "word", locale = "zh",
       skip_word_none = TRUE, skip_word_number = TRUE)
    return(vapply(units, paste, character(1), collapse = " "))
  }
'''+s[pos:]
p.write_text(s,encoding='utf-8')
cfg=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
cfg['chinese']='ICU dictionary word segmentation with locale zh. Evaluate raw prefixes without inserted spaces; prediction segments the prefix alone. The target is one unit of full-sentence ICU segmentation, not a human-annotated word.'
(out/'protocol.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
