"""Record verified local preview artifacts; does not read held-out English data."""
from pathlib import Path
import hashlib,json,re

root=Path.cwd()
app=root/'final_project/phraseflow_multilingual'
out=root/'final_project/research/multilingual-20260926'
checks=json.loads((out/'checks.json').read_text())
html=(app/'PhraseFlow_Pitch.html').read_text(encoding='utf-8')
assert len(re.findall(r'<section\b',html))==5
assert 'SONJA PHRASEFLOW   /   SONJA PROJECTS   /   SONJA SAHEBZAD' in html
assert checks['english_exact_600'] and not checks['english_reserved_test_opened']
assert checks['english_model_md5']==checks['production_model_md5']
assert (app/'Multilingual_Guide.html').read_bytes()==(app/'www/Multilingual_Guide.html').read_bytes()
assert (app/'Sonja PhraseFlow.Rproj').exists()
record=dict(native_presenter_slides=5,all_five_slides_visually_checked=True,
    language_choices=['en_US','de_DE','fi_FI','ru_RU'],all_languages_supported=False,
    browser_language_reset_checked=True,browser_cyrillic_append_refresh_checked=True,
    browser_finnish_metrics_match_saved_results=True,language_selector_text_visible=True,
    html_guide_served_by_shiny=True,production_deployment_changed=False,
    course_submission_made=False)
(out/'release_check.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
paths=[app/'app.R',app/'predictor.R',app/'www/styles.css',app/'www/input.js',
    app/'PhraseFlow_Pitch.Rpres',app/'PhraseFlow_Pitch.html',app/'Multilingual_Guide.Rmd',
    app/'Multilingual_Guide.html',app/'pitch.css',app/'www/multilingual-demo.png',
    app/'www/multilingual-preview.png']
(out/'artifact_sha256.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
