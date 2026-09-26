"""Derive short endings from existing training text only; never reads evaluation text."""
from pathlib import Path
from collections import Counter, defaultdict
import csv, hashlib, json, shutil, time

root=Path.cwd(); app=root/'final_project/phraseflow_multilingual'
out=root/'final_project/research/continuations-20260926-v14'
if out.exists(): raise SystemExit('Preserve existing experiment')
backup=root/'final_project/phraseflow_multilingual_v13_20260926'
if backup.exists(): raise SystemExit('Preserve existing backup')
shutil.copytree(app,backup);out.mkdir(parents=True)
(app/'continuations').mkdir()
cfg=json.loads((app/'language-config.json').read_text(encoding='utf-8'))
manifest={}
for code in cfg:
    path=(root/'data/student_context/training_lines.csv' if code=='en_US' else
          root/'data'/('multilingual_20260926' if code in ['de_DE','fi_FI','ru_RU'] else 'languages_20260926_v13')/f'{code}_train.csv')
    start=time.perf_counter();counts=Counter();lines=0
    with path.open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            w=row['text'].split()
            lines+=1
            # Only the final one to three normalized units; no invented sequences.
            for size in range(1,4):
                cut=len(w)-size
                if cut<2: continue
                continuation=' '.join(w[cut:])
                for context_size in range(2,min(4,cut)+1):
                    counts[(' '.join(w[cut-context_size:cut]),continuation)]+=1
    groups=defaultdict(list)
    for (context,continuation),count in counts.items(): groups[context].append((continuation,count))
    dest=out/f'{code}_index.tsv'
    with dest.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['context','continuation','count'])
        for context,items in sorted(groups.items()):
            for continuation,count in sorted(items,key=lambda x:(-x[1],x[0]))[:8]:
                writer.writerow([context,continuation,count])
    manifest[code]={'training_file':str(path.relative_to(root)),
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'training_lines':lines,
        'contexts':len(groups),'build_seconds':round(time.perf_counter()-start,3)}
    print(code,manifest[code],flush=True)
    del counts,groups
(out/'training_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(out/'protocol.json').write_text(json.dumps({'version':'1.4','author':'Sonja Sahebzad',
    'method':'Training-sentence endings; exact suffix match of 2 to 4 normalized units; one to three unit continuations.',
    'pruning':'Retain up to eight endings per context by count, lexical tie-break.',
    'runtime':'Prefer longest matching suffix. If only two units match, offer the first next unit only. For longer matches, retain complete saved endings. Rank by training count.',
    'limits':'Local context lookup; not full-sentence semantic understanding. Examples are demonstrations, not an accuracy evaluation.',
    'reserved_test_opened':False,'new_training':False,'hardcoded_user_answers':False},indent=2),encoding='utf-8')
