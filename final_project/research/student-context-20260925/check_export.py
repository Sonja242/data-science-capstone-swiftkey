"""Neutral R/Python parity check before examining student outcomes."""
from pathlib import Path
import importlib.util, json
import numpy as np
import torch
p=Path('final_project/research/student-context-20260925/student_study.py')
spec=importlib.util.spec_from_file_location('student_pilot',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
model=m.Student().eval()
out=m.DATA/'neutral_export';out.mkdir(exist_ok=True)
manifest={}
for key,t in model.state_dict().items():
    arr=t.detach().numpy();fname=key.replace('.','_')+'.f32'
    arr.astype('<f4').tofile(out/fname)
    manifest[key]={'file':fname,'shape':list(arr.shape),'sha256':m.sha(out/fname)}
m.save(out/'weights.json',manifest)
phrases=['','we are talking about','a clear scientific explanation','i would like a','top of the',
         "i'm looking forward to",'unknownqqzz strangeqqzz',' '.join(['the']*70)]
x,l=m.inputs(phrases)
with torch.inference_mode():prob=torch.softmax(model(torch.tensor(x),torch.tensor(l)),dim=1).numpy()
prob.astype('<f4').tofile(out/'probabilities.f32')
m.save(out/'phrases.json',phrases)
print('Neutral Python export prepared, no development outcomes examined.')
