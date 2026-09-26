from pathlib import Path
p=Path.cwd()/'final_project/phraseflow_multilingual/www/input.js'
s=p.read_text(encoding='utf-8')
s=s.replace("document.addEventListener('change', function(event) {\n    if (event.target.id !== 'language') return;", "document.addEventListener('change', function(event) {\n    if (event.target.id === 'phrase') { guardSuggestions(); sendCurrentText(); return; }\n    if (event.target.id !== 'language') return;")
p.write_text(s,encoding='utf-8')
