from pathlib import Path
import json,shutil,hashlib
root=Path.cwd();app=root/'final_project/phraseflow_multilingual';out=root/'final_project/research/continuations-20260926-v14';stage=Path(__file__).parent
p=app/'Multilingual_Guide.Rmd';s=p.read_text(encoding='utf-8')
timing=json.loads((out/'continuation_checks.json').read_text(encoding='utf-8'))
needle='Cold requests can take longer.'
s=s.replace(needle,needle+f" A new local timing check on 600 reused English development prefixes measured a median of **{timing['median_ms']:.2f} ms** and a 95th percentile of **{timing['p95_ms']:.2f} ms** for twenty word candidates plus the extra lookup. This is a warm CPU measurement, not browser response time or an accuracy test.")
p.write_text(s,encoding='utf-8')
for name in ['multilingual-demo.png']:
 shutil.copy2(stage/name,app/'www'/name)
manifest=json.loads((app/'preview_manifest.json').read_text(encoding='utf-8'))
manifest.update(version='1.4 preview',guide='Multilingual_Guide.html',previous_preview=str(root/'final_project/phraseflow_multilingual_v13_20260926'),
 features=['5, 10 or 20 visible word choices','Supplementary training-text endings in twelve languages','Stable insertion and language change handling'],
 continuations_independently_evaluated=False)
(app/'preview_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
readme=app/'README.md';s=readme.read_text(encoding='utf-8').replace('preview 1.3','preview 1.4')
s+='\nVersion 1.4 adds selectable 5/10/20 visible word suggestions and separate short continuations derived only from the existing training files. These ending suggestions are experimental and are not included in the reported word accuracy. Version 1.3 is preserved in `../phraseflow_multilingual_v13_20260926`. Code, provenance, timing and checks: `../research/continuations-20260926-v14`.\n'
readme.write_text(s,encoding='utf-8')
for f in stage.iterdir():
 if f.is_file():shutil.copy2(f,out/f.name)
for name in ['app.R','predictor.R','continuations.R','Multilingual_Guide.Rmd','PhraseFlow_Pitch.Rpres']:
 shutil.copy2(app/name,out/name)
for name in ['input.js','styles.css']:shutil.copy2(app/'www'/name,out/name)
