from pathlib import Path
import json,hashlib,shutil
ROOT=Path.cwd();OUT=ROOT/'final_project/research/multilingual-20260926';APP=ROOT/'final_project/phraseflow_multilingual'
m=json.loads((OUT/'metrics.json').read_text());labels={'de_DE':'German','fi_FI':'Finnish','ru_RU':'Russian'}
score=', '.join(f'{labels[k]} {v["overall"]["top3"]*100:.1f}%' for k,v in m.items())
p=APP/'PhraseFlow_Pitch.Rpres';s=p.read_text(encoding='utf-8');s=s.replace('LANGUAGE_SCORES',f'Correct within three: {score}. Each uses 600 held-out cases. These early results call for further improvement.')
p.write_text(s,encoding='utf-8')
assert sum(x.startswith('==========') for x in s.splitlines())==5
(APP/'README.md').write_text('''# Sonja PhraseFlow multilingual preview

Author: Sonja Sahebzad | Sonja Projects | Preview 1.2

Open `Sonja PhraseFlow.Rproj` in RStudio, then `app.R` and Run App. Four real model choices are available: English, German, Finnish and Russian. A language switch clears the input and previous suggestions. Models load once per server process as needed. The interface remains English. This is a local prototype, not support for every language.

`Multilingual_Guide.Rmd` and its knitted HTML document usage, fixed training settings, held-out scores and limitations. The original English PDF in documentation is retained as a version 1.1 artifact; use the multilingual guide for this version.

For the native five-slide deck, run `.rs.showPresentation("PhraseFlow_Pitch.Rpres")` in RStudio from this folder. In the Presentation pane use More > Save as Web Page. This preserves the required RStudio Presenter format. The identity bar appears at the top of every slide.

Research scripts and results are in `../research/multilingual-20260926`. Raw training text and held-out answers are private under `data/multilingual_20260926` in the corpus project root. Do not publish that data directory. No publication, deployment or course submission is part of this preview.
''',encoding='utf-8')
(OUT/'README.md').write_text('''# Multilingual extension study

Author: Sonja Sahebzad. Sonja Projects. 26 September 2026.

Run from the corpus project root. Preserve the completed run; preparation and training refuse overwrites. For a new experiment use a separate copy and a new output location.

Order: init_multilingual.py, prepare_multilingual.py, build_multilingual.R, install_multilingual_ui.py, portable_r_sources.py, test_multilingual.R, finish_multilingual.py. Use the project Python environment for Python scripts and R 4.6.1 for R scripts. App documents are rendered with rmarkdown and the five-slide pitch with native RStudio Presenter.

`protocol.json` fixes settings before held-out evaluation. `source_manifest.json` records the local official archive checksum and members. `metrics.json` contains separate language metrics, Wilson intervals, unigram references, CPU timings and size. `*_case_metrics.csv` preserves per-case outcomes without phrase or answer text. `checks.json` records unchanged English predictions and Unicode/server checks.

The three new fixed language models use 75,000 training lines each and 600 held-out examples each. English remains unchanged. The reserved English 900 cases are never read. These are small experimental language models; no all-language or calibrated-confidence claim is supported. The three language-specific hold-outs must not be reused as independent final tests after tuning based on these results.
''',encoding='utf-8')
files=[*OUT.glob('*.py'),*OUT.glob('*.R'),APP/'app.R',APP/'predictor.R',APP/'www/input.js']
(OUT/'code_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
print(score)
