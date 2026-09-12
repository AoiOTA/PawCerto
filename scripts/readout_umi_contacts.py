"""Read existing full sampled contacts only; retain original1N counts and signed world force conventions."""
import argparse
import json,hashlib
from pathlib import Path
import numpy as np
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--policy', type=Path, required=True, help='Directory containing case_00.json/npz through case_15')
parser.add_argument('--reference', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
B = args.output
B.mkdir(parents=True, exist_ok=False)
GROUPS = {'official': args.reference, 'candidate': args.policy}
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(rows):
 active=np.array([r['nonfoot_gt1'] for r in rows]);f=np.array([r['nonfoot_up_Fz_N'] for r in rows]);allf=np.array([r['all_ground_up_Fz_N'] for r in rows]);shares=np.divide(f,allf,out=np.zeros_like(f),where=allf>0)
 return dict(saved_endpoints=len(rows),nonfoot_external_gt1_endpoints=int(active.sum()),nonfoot_up_Fz_sum_N_samples=float(f.sum()),all_ground_up_Fz_sum_N_samples=float(allf.sum()),nonfoot_fraction_of_summed_ground_up_Fz=float(f.sum()/allf.sum()) if allf.sum()>0 else None,endpoint_fraction_mean=float(shares.mean()),endpoint_fraction_max=float(shares.max()),active_endpoint_fraction_mean=float(shares[active].mean()) if active.any() else None,nonfoot_up_Fz_max_N=float(f.max()),zero_ground_foot_support_endpoints=sum(r['ground_feet']==0 for r in rows),external_active_with_zero_ground_feet=sum(r['nonfoot_gt1'] and r['ground_feet']==0 for r in rows))
reports={};paths=[]
for group,directory in GROUPS.items():
 cases=[];allrows=[];body={}
 for i in range(16):
  jp=directory/f'case_{i:02d}.json';npz=directory/f'case_{i:02d}.npz';paths += [jp,npz];meta=load(jp)
  if meta.get('completion_status') != 'complete' or meta.get('failure') is not None:
   raise ValueError(f'{group} case {i} is not complete; this contact report requires all 16 complete cases')
  with np.load(npz) as raw:clock=raw['metrics'][:,0].copy();ground=raw['ground_supported_feet'].copy();savedfoot=raw['foot_ground_force_z_N'].copy()
  index={round(float(t),9):j for j,t in enumerate(clock)}
  rows=[dict(case_index=i,time_s=float(t),nonfoot_gt1=False,nonfoot_up_Fz_N=0.,all_ground_up_Fz_N=0.,ground_feet=int(ground[j]),bodies_gt1=[]) for j,t in enumerate(clock)]
  signedfeet=np.zeros_like(savedfoot);footnames=['FR_foot','FL_foot','RR_foot','RL_foot'];bodycase={}
  allcontacts=meta['nonfoot_contact_records']+meta['foot_or_finger_only_contact_records']
  for c in allcontacts:
   j=index[round(c['time_s'],9)]
   if c['kind']!='external':continue
   assert 'world' in c['body_names'] and 0 in c['geom_ids']
   world_side=c['body_names'].index('world');side=1-world_side;name=c['body_names'][side]
   force=np.array(c['force_world_on_geom2_N'])*(1 if side==1 else -1)
   if name in footnames:signedfeet[j,footnames.index(name)]+=force[2]
   up=max(float(force[2]),0.)
   rows[j]['all_ground_up_Fz_N']+=up
   if c['nonfoot_nongripper']:
    rows[j]['nonfoot_up_Fz_N']+=up
    if name not in bodycase:bodycase[name]=dict(gt1_times=set(),up_Fz_sum_N_samples=0.,max_force_norm_N=0.)
    bodycase[name]['up_Fz_sum_N_samples']+=up;bodycase[name]['max_force_norm_N']=max(bodycase[name]['max_force_norm_N'],c['force_norm_N'])
    if c['force_norm_gt_1N']:
     rows[j]['nonfoot_gt1']=True;rows[j]['bodies_gt1'].append(name);bodycase[name]['gt1_times'].add(round(c['time_s'],9))
  np.testing.assert_allclose(signedfeet,savedfoot,rtol=0,atol=1e-10)
  assert sum(r['nonfoot_gt1'] for r in rows)==meta['nonfoot_external_contact_samples_gt_1N']
  for name,v in bodycase.items():
   if name not in body:body[name]=dict(body_endpoint_count_gt1=0,up_Fz_sum_N_samples=0.,max_force_norm_N=0.,cases_with_gt1=[])
   body[name]['body_endpoint_count_gt1']+=len(v['gt1_times']);body[name]['up_Fz_sum_N_samples']+=v['up_Fz_sum_N_samples'];body[name]['max_force_norm_N']=max(body[name]['max_force_norm_N'],v['max_force_norm_N'])
   if v['gt1_times']:body[name]['cases_with_gt1'].append(i)
  windows={name:stats(subset) for name,subset in [('all',rows),('startup_before_1s',[r for r in rows if r['time_s']<1]),('at_or_after_1s',[r for r in rows if r['time_s']>=1])]}
  cases.append(dict(case_index=i,windows=windows,endpoint_values=rows));allrows+=rows
 windows={name:stats(subset) for name,subset in [('all',allrows),('startup_before_1s',[r for r in allrows if r['time_s']<1]),('at_or_after_1s',[r for r in allrows if r['time_s']>=1])]}
 total=windows['all']['nonfoot_up_Fz_sum_N_samples']
 for v in body.values():v['fraction_of_nonfoot_up_Fz_sum']=v['up_Fz_sum_N_samples']/total if total else None
 reports[group]=dict(windows=windows,bodies=body,cases=cases)
tracked={str(p):sha(p) for p in paths};tracked[str(Path(__file__).resolve())]=sha(Path(__file__))
result=dict(method='Existing20ms policy-endpoint contact JSON and raw support arrays only; no simulation. Reported counts are sums over cases of distinct sampled times with any nonfoot external contact world-force norm>1N, not contact-point counts.',warmup_boundary='Actual evaluator20ms zero-action warmup is not saved. Earliest retained state/contact time is0.04s. Existing coordination descriptive1s split reused: saved0.04..0.98s startup versus>=1s; this is not a newly defined warmup or success threshold.',force_fraction_definition='For each saved external ground contact, apply existing geom2 world-force sign to robot, then sum positive vertical components. Nonfoot fraction=sum(nonfoot upward Fz)/sum(all ground upward Fz), including foot/finger/nonfoot; force sums include all recorded values regardless of1N count threshold. Report both ratio of sums and endpoint fraction mean/max. N-samples are not continuous impulses or bodyweight fractions.',inference_boundary='Counts/force fractions are sampled load descriptors, not continuous support or necessary coordination. No claim that the policy mainly relies on nonfoot support or that residual contact is harmless. Self contacts are excluded from external ground-load denominator. Original1N threshold and behavior acceptance unchanged.',groups=reports,input_sha256=tracked)
(B/'contact-quality-readout.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
lines=['# Saved-contact quality comparison','',result['method'],'',result['warmup_boundary'],'',result['force_fraction_definition'],'',result['inference_boundary']]
for name,g in reports.items():
 lines += ['',f'## {name}: sampled windows','','| Group/window | Endpoints | Nonfoot active | Nonfoot fraction of summed upward ground Fz | Endpoint mean / max fraction | Zero foot support / also nonfoot active |','|---|---:|---:|---:|---:|---:|']
 for win,w in g['windows'].items():lines.append(f"| {name}/{win} | {w['saved_endpoints']} | {w['nonfoot_external_gt1_endpoints']} | {w['nonfoot_fraction_of_summed_ground_up_Fz']:.8%} | {w['endpoint_fraction_mean']:.8%} / {w['endpoint_fraction_max']:.8%} | {w['zero_ground_foot_support_endpoints']} / {w['external_active_with_zero_ground_feet']} |")
 lines+=['',f'## {name}: body contributions','','| Body | Active body-endpoints | Cases | Upward Fz share within nonfoot | Max contact force norm N |','|---|---:|---|---:|---:|']
 for body,v in sorted(g['bodies'].items(),key=lambda kv:-kv[1]['up_Fz_sum_N_samples']):lines.append(f"| {body} | {v['body_endpoint_count_gt1']} | {v['cases_with_gt1']} | {v['fraction_of_nonfoot_up_Fz_sum']:.6%} | {v['max_force_norm_N']:.6g} |")
 lines+=['','| Case | Nonfoot active startup / later | Nonfoot ratio of summed upward ground Fz | Max endpoint fraction |','|---:|---:|---:|---:|']
 for c in g['cases']:
  w=c['windows'];lines.append(f"| {c['case_index']} | {w['startup_before_1s']['nonfoot_external_gt1_endpoints']} / {w['at_or_after_1s']['nonfoot_external_gt1_endpoints']} | {w['all']['nonfoot_fraction_of_summed_ground_up_Fz']:.7%} | {w['all']['endpoint_fraction_max']:.7%} |")
(B/'contact-quality-readout.md').write_text('\n'.join(lines)+'\n')
assert all(sha(Path(p))==s for p,s in tracked.items())
print(json.dumps({k:dict(windows=v['windows'],bodies=v['bodies']) for k,v in reports.items()},indent=2))
