"""Sample original non-English course text. Never reads English test cases."""
from pathlib import Path
import zipfile,random,csv,json,hashlib,time
ROOT=Path.cwd();OUT=ROOT/'final_project/research/multilingual-20260926';DATA=ROOT/'data/multilingual_20260926'
OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
protocol=OUT/'protocol.json'
if protocol.exists():raise FileExistsError('Preserve completed experiment')
cfg=dict(author='Sonja Sahebzad',brand='Sonja Projects',date='2026-09-26',
 languages=['de_DE','fi_FI','ru_RU'],samples_per_source=32000,training_cap_per_source=25000,
 heldout_cases_per_source=200,maximum_words_per_line=80,seed=20260927,
 maximum_order=5,vocabulary_cap=40000,minimum_count=2,context_support=3,followers=8,
 method='Pruned interpolated Kneser-Ney, fixed discount 0.75',
 selection='No tuning or model selection on held-out cases. Fixed compact settings for first language prototypes.',
 baselines='Same-language unigram frequency, evaluated on the exact same held-out cases.',
 split='Global normalized-line deduplication within language before source-stratified split; one random target per held-out line.',
 tokenizer='NFC Unicode letters with internal apostrophes, lowercase, URL/mention removal. English tokenizer unchanged.',
 english_model_changed=False,english_reserved_test_opened=False,
 deployment='Local prototype only. No claim of all-language support or semantic understanding.')
protocol.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
archive=ROOT/'data/Coursera-SwiftKey.zip'
sha=hashlib.sha256()
with archive.open('rb') as f:
 for chunk in iter(lambda:f.read(1<<20),b''):sha.update(chunk)
provenance=[]
with zipfile.ZipFile(archive) as z:
 for i,lang in enumerate(cfg['languages']):
  selected=[]
  for j,source in enumerate(['blogs','news','twitter']):
   name=f'final/{lang}/{lang}.{source}.txt';rng=random.Random(cfg['seed']+100*i+j)
   reservoir=[];n=0
   with z.open(name) as f:
    for n,line in enumerate(f,1):
     text=line.decode('utf-8',errors='replace').replace('\x00','').strip()
     if n<=cfg['samples_per_source']:reservoir.append(text)
     else:
      k=rng.randrange(n)
      if k<len(reservoir):reservoir[k]=text
   selected.extend(dict(source=source,text=t) for t in reservoir)
   provenance.append(dict(language=lang,source=source,member=name,archive_lines=n,sampled=len(reservoir),crc=z.getinfo(name).CRC))
  with (DATA/f'{lang}_sample.csv').open('w',encoding='utf-8',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=['source','text']);writer.writeheader();writer.writerows(selected)
  print(lang,'sampled',len(selected),flush=True)
(OUT/'source_manifest.json').write_text(json.dumps(dict(source_url='https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip',archive_sha256=sha.hexdigest(),members=provenance),indent=2),encoding='utf-8')
