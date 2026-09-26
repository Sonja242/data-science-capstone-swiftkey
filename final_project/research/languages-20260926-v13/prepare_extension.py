"""Freeze public Tatoeba exports and provenance for eight compact prototypes."""
from pathlib import Path
import bz2,csv,hashlib,json,random,shutil,urllib.request,concurrent.futures,datetime
ROOT=Path.cwd(); APP=ROOT/'final_project/phraseflow_multilingual'
OUT=ROOT/'final_project/research/languages-20260926-v13'
PRIVATE=ROOT/'data/languages_20260926_v13'
LANGS={
 'nl_NL':dict(iso='nld',name='Dutch',label='Nederlands',tokenizer='unicode',examples=['Ik kijk uit naar','Het weer is vandaag','Dank je wel voor']),
 'fr_FR':dict(iso='fra',name='French',label='Français',tokenizer='unicode',examples=['Je voudrais aller au','Merci beaucoup pour','Il fait beau']),
 'ko_KR':dict(iso='kor',name='Korean',label='한국어',tokenizer='unicode',examples=['저는 한국어를','오늘 날씨가','저는 학교에']),
 'zh_CN':dict(iso='cmn',name='Mandarin Chinese',label='中文（普通话）',tokenizer='chinese',examples=['我想去','今天的天气','非常感谢你的']),
 'pt_PT':dict(iso='por',name='Portuguese',label='Português',tokenizer='unicode',examples=['Eu gostaria de','Muito obrigado pela','Hoje o tempo está']),
 'it_IT':dict(iso='ita',name='Italian',label='Italiano',tokenizer='unicode',examples=['Vorrei andare al','Grazie mille per','Il tempo oggi è']),
 'hi_IN':dict(iso='hin',name='Hindi',label='हिन्दी',tokenizer='unicode',examples=['मुझे घर जाना','आज मौसम बहुत','आपका बहुत बहुत']),
 'es_ES':dict(iso='spa',name='Spanish',label='Español',tokenizer='unicode',examples=['Me gustaría ir a','Muchas gracias por','Hoy hace buen'])}

def main():
 if OUT.exists(): raise SystemExit('Preserve the completed or partial experiment; use its saved downloads for recovery.')
 backup=ROOT/'final_project/phraseflow_multilingual_v12_20260926'
 if backup.exists(): raise SystemExit('Existing backup must be preserved')
 shutil.copytree(APP,backup)
 OUT.mkdir(parents=True);PRIVATE.mkdir(parents=True)
 protocol=dict(author='Sonja Sahebzad',location='Utrecht, the Netherlands',brand='Sonja Projects',
  version='1.3',languages=LANGS,seed=20260928,sample_cap=100000,training_cap=75000,heldout_cases=600,
  maximum_words_per_line=80,maximum_order=5,vocabulary_cap=40000,minimum_count=2,context_support=3,followers=8,
  method='Fixed pruned interpolated Kneser-Ney, discount 0.75',
  split='Normalize, exact-deduplicate, randomly hold out 600 sentences per language; no tuning on these cases.',
  limitation='Tatoeba is a volunteer example-sentence corpus, not a representative news/chat benchmark. Near-duplicate templates may remain.',
  chinese='ICU dictionary word segmentation with locale zh; prefixes retained with known token boundaries during evaluation; one segmented unit as next word.',
  korean='Whitespace-delimited eojeol units retained, including inflections; no morpheme decomposition.',
  confidence='Uncalibrated; report Wilson sampling intervals separately from individual confidence.',
  english_reserved_test_opened=False,production_changed=False)
 (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
 def download(item):
  code,d=item;iso=d['iso'];url=f'https://downloads.tatoeba.org/exports/per_language/{iso}/{iso}_sentences_detailed.tsv.bz2'
  dest=PRIVATE/f'{iso}_sentences_detailed.tsv.bz2'
  req=urllib.request.Request(url,headers={'User-Agent':'SonjaPhraseFlow-research/1.3'})
  with urllib.request.urlopen(req,timeout=90) as response, dest.open('wb') as f:
   last=response.headers.get('Last-Modified');size=int(response.headers.get('Content-Length','0'))
   if size>80_000_000: raise ValueError('Export exceeds bounded prototype download limit')
   shutil.copyfileobj(response,f)
  rng=random.Random(protocol['seed']+list(LANGS).index(code));rows=[];count=0
  with bz2.open(dest,'rt',encoding='utf-8') as f:
   for line in f:
    parts=line.rstrip('\n').split('\t')
    if len(parts)<6 or parts[1]!=iso: continue
    count+=1;record=(parts[0],parts[2],parts[3])
    if len(rows)<protocol['sample_cap']:rows.append(record)
    else:
     j=rng.randrange(count)
     if j<len(rows):rows[j]=record
  with (PRIVATE/f'{code}_sample.csv').open('w',encoding='utf-8',newline='') as f:
   writer=csv.writer(f);writer.writerow(['sentence_id','text','owner']);writer.writerows(rows)
  result=dict(language=code,iso=iso,url=url,last_modified=last,downloaded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),source_sentences=count,sampled_sentences=len(rows),
   license='CC BY 2.0 FR',license_url='https://creativecommons.org/licenses/by/2.0/fr/',source='https://tatoeba.org/en/downloads')
  print(json.dumps(result),flush=True);return code,result
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  results=dict(pool.map(download,LANGS.items()))
 (OUT/'source_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
 (APP/'extension-languages.json').write_text(json.dumps(LANGS,ensure_ascii=False,indent=2),encoding='utf-8')
 for p in Path(__file__).parent.glob('*'):
  if p.suffix in {'.py','.R','.Rmd','.Rpres'} and p.resolve()!=(OUT/p.name).resolve():shutil.copy2(p,OUT/p.name)

if __name__=='__main__': main()
