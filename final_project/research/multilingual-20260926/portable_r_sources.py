"""Use R Unicode escapes so source works even under the Windows C locale."""
from pathlib import Path
ROOT=Path.cwd();OUT=ROOT/'final_project/research/multilingual-20260926';APP=ROOT/'final_project/phraseflow_multilingual'
for p in [OUT/'multilingual_app.R',OUT/'test_multilingual.R']:
 s=p.read_text(encoding='utf-8')
 s=''.join(c if ord(c)<128 else '\\u%04x'%ord(c) for c in s)
 p.write_text(s,encoding='ascii')
(APP/'app.R').write_text((OUT/'multilingual_app.R').read_text(encoding='ascii'),encoding='ascii')
