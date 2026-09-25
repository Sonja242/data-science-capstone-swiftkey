"""Isolate GPU reduced-precision inference from native R double precision."""
from pathlib import Path
import importlib.util
import shutil
import numpy as np
import torch
spec=importlib.util.spec_from_file_location('student_pilot',Path('final_project/research/student-context-20260925/student_study.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if (m.OUT/'precision_correction.json').exists():raise FileExistsError('Preserve the completed precision check.')
archive=m.DATA/'precision-audit-before';archive.mkdir(exist_ok=True)
for name in ['CE_control_dev.f32','Distill25_dev.f32','Distill50_dev.f32']:
 if not (archive/name).exists():shutil.copy2(m.DATA/name,archive/name)
for name in ['python_development_comparison.csv','python_development_cases.csv','export_checks.json']:
 if (m.OUT/name).exists() and not (archive/name).exists():shutil.copy2(m.OUT/name,archive/name)
cases=m.rows(m.DATA/'development.csv');x,l=m.inputs([r['prefix']for r in cases])
records=[]
old_flags={'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
for name in ['CE_control','Distill25','Distill50']:
 model=m.Student();model.load_state_dict(torch.load(m.DATA/(name+'.pt'),map_location='cpu',weights_only=True));model.eval()
 with torch.inference_mode():
  cpu=torch.softmax(model(torch.tensor(x[:32]),torch.tensor(l[:32])),dim=1).numpy()
  model=model.cuda()
  gpu=torch.softmax(model(torch.tensor(x[:32],device='cuda'),torch.tensor(l[:32],device='cuda')),dim=1).cpu().numpy()
  model=model.cpu().double()
  double=torch.softmax(model(torch.tensor(x[:32]),torch.tensor(l[:32])),dim=1).numpy()
 print(name,'CPU32 vs CPU64',abs(cpu-double).max(),'strict GPU32 vs CPU64',abs(gpu-double).max(),flush=True)
 # Publish CPU reference matching the R hosting path. No weights are changed.
 all_probs=[]
 with torch.inference_mode():
  for at in range(0,600,32):
   p=torch.softmax(model(torch.tensor(x[at:at+32]),torch.tensor(l[at:at+32])),dim=1).numpy()
   all_probs.append(p)
 result=np.concatenate(all_probs).astype('<f4')
 before=np.fromfile(m.DATA/'precision-audit-before'/(name+'_dev.f32'),dtype='<f4').reshape(600,50000)
 before_top=np.argsort(-before,axis=1,kind='stable')[:,:3]
 after_top=np.argsort(-result,axis=1,kind='stable')[:,:3]
 result.tofile(m.DATA/(name+'_dev.f32'))
 records.append({'model':name,'cpu32_cpu64_max_error_32cases':float(abs(cpu-double).max()),
 'strict_gpu32_cpu64_max_error_32cases':float(abs(gpu-double).max()),
 'previous_gpu_reference_max_error_600cases':float(abs(result-before).max()),
 'changed_top3_lists_600cases':int(np.any(before_top!=after_top,axis=1).sum()),
 'checkpoint_sha256':m.sha(m.DATA/(name+'.pt'))})
m.save(m.OUT/'precision_correction.json',{'original_gpu_flags':old_flags,'records':records,
 'correction':'Use double-precision CPU inference for the Python reference exported to native R. Weights, data, candidate grid, training budget and selection rules unchanged. Earlier GPU-reference outputs preserved privately.',
 'final_test_accessed':False})
m.compare()
