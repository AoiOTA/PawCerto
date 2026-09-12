"""Append actual final native nominal16 and author latest500 evidence to all-seed summary."""
import os
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--summary-dir', type=Path, required=True, help='Output from summarize_umi_three_seeds.py; amended in place')
args = parser.parse_args()
ROOT = args.root.resolve()
BASE = args.summary_dir.resolve()
ISAAC=ROOT/'outputs/isaac'
PATHS={str(s):dict(nominal=ISAAC/f'umi-relaxed-body-speed-scratch{f"-seed{s}" if s else ""}-nominal16-seed2027',author=ISAAC/f'umi-relaxed-body-speed-scratch{f"-seed{s}" if s else ""}-author-model4000-seed2027') for s in range(3)}
PATHS['official']=dict(nominal=ISAAC/'umi-relaxed-body-speed-scratch-nominal16-official-seed2027',author=ISAAC/'umi-relaxed-body-speed-adapt-author-official-seed2027')
INPUTS={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    INPUTS[str(p)]=sha(p)
    return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
    result={}
    for seed,paths in PATHS.items():
        for directory in paths.values():assert load(directory/'execution.json')['exit_code']==0
        r=load(paths['nominal']/'rollout.json')
        assert r['num_envs']==16 and r['steps']==850 and r['seed']==2027 and len(r['rows'])==849
        assert [row['step'] for row in r['rows']]==list(range(849))
        clock=np.asarray([row['episode_time_s'] for row in r['rows']])
        assert clock.shape==(849,16) and np.isfinite(clock).all()
        assert np.all(np.diff(clock,axis=0)>0) and np.all(clock==clock[:,:1])
        np.testing.assert_allclose(clock[:,0],np.arange(2,851)*.02,rtol=0,atol=.001)
        keys=['ee_error_m','ee_error_rad','up_dot','ground_supported_feet']
        a={k:np.asarray([x[k] for x in r['rows']]) for k in keys}
        assert all(v.shape==(849,16) and np.isfinite(v).all() for v in a.values())
        nominal=dict(position_error_mean_m=float(a['ee_error_m'].mean()),orientation_error_mean_rad=float(a['ee_error_rad'].mean()),minimum_root_up_dot=float(a['up_dot'].min()),inverted_indices=np.flatnonzero((a['up_dot']<0).any(axis=0)).tolist(),ground_supported_feet_mean=float(a['ground_supported_feet'].mean()),zero_ground_supported_feet_fraction=float((a['ground_supported_feet']==0).mean()),case_count=16,policy_endpoints_per_case=849,first_saved_episode_time_s=float(clock[0,0]),last_saved_episode_time_s=float(clock[-1,0]),source=str(paths['nominal']/'rollout.json'))
        s=load(paths['author']/'summary.json');all_episodes=load(paths['author']/'completed_episodes.json')
        assert len(all_episodes)==s['actual_completed_episodes'] and s['included_completed_episodes']==500
        episodes=all_episodes[-500:]
        assert s['evicted_completed_episodes']==len(all_episodes)-500
        author={}
        for name,subset,key in [('latest500',episodes,'metrics'),('timeout',[x for x in episodes if x['time_out']],'timeout_episode_metrics'),('early_termination',[x for x in episodes if not x['time_out']],'early_termination_metrics')]:
            observed=s[key]
            if not subset:
                author[name]=dict(completed_episodes=0,metrics=observed);continue
            assert observed['completed_episodes']==len(subset)
            for field,value in observed.items():
                if field in subset[0] and isinstance(value,(int,float)):
                    assert np.isclose(value,np.mean([x[field] for x in subset]),rtol=0,atol=1e-10),(seed,name,field)
            assert observed['inverted_completed_episodes']==sum(x['inverted'] for x in subset)
            assert observed['minimum_root_up_dot']==min(x['root_up_dot_min'] for x in subset)
            assert np.isclose(observed['timeout_survival_proxy'],np.mean([x['time_out'] for x in subset]),atol=1e-12)
            author[name]=observed
        author.update(actual_completed_episodes=len(all_episodes),included_completed_episodes=500,evicted_completed_episodes=s['evicted_completed_episodes'],unfinished_episode_count=s['unfinished_episode_count'],source=str(paths['author']/'summary.json'),boundary=s['aggregate_scope'])
        result[seed]=dict(nominal16=nominal,author_protocol=author)
    report=json.loads((BASE/'summary.json').read_text())
    assert report['status']=='all_three_fixed_4000_cpu_scopes_complete'
    assert all(sha(p)==h for p,h in INPUTS.items())
    report['native_final_evaluations']=result;report['native_input_sha256']=INPUTS
    report['native_boundary']='Nominal16 uses850 policy intervals including one unsaved warmup; averages all849 retained endpoints per case, approximately0.04..17s with native float32 clock drift recorded. Author protocol averages per-episode means over exact latest500 completed episodes; timeout and early termination remain separate. Native and MuJoCo both retain849 endpoints, but their engine behavior and episode protocols are not interchangeable, and completion counts do not establish formal WBC acceptance.'
    dump(BASE/'summary.json',report)
    lines=['','<!-- BEGIN ACTUAL NATIVE FINAL READOUT -->','',report['native_boundary'],'','| Policy | Lab nominal EE (mm) | Orientation (rad) | Inverted /16 | Ground feet mean | Zero ground-foot % |','|---|---:|---:|---:|---:|---:|']
    for seed,r in result.items():
        m=r['nominal16'];lines.append(f"| {seed} | {m['position_error_mean_m']*1000:.3f} | {m['orientation_error_mean_rad']:.5f} | {len(m['inverted_indices'])} | {m['ground_supported_feet_mean']:.4f} | {m['zero_ground_supported_feet_fraction']*100:.4f} |")
    lines+=['','| Policy | Author stratum | Episodes | EE mean (mm) | Orientation (rad) | Timeout % | Inverted episodes |','|---|---|---:|---:|---:|---:|---:|']
    for seed,r in result.items():
        for stratum in ('latest500','timeout','early_termination'):
            m=r['author_protocol'][stratum]
            if not m['completed_episodes']:
                lines.append(f'| {seed} | {stratum} | 0 | unavailable | unavailable | unavailable | 0 |');continue
            lines.append(f"| {seed} | {stratum} | {m['completed_episodes']} | {m['position_error/mean']*1000:.3f} | {m['orientation_error/mean']:.5f} | {m['timeout_survival_proxy']*100:.2f} | {m['inverted_completed_episodes']} |")
    lines+=['']
    for seed,paths in PATHS.items():lines.append(f"- {seed}: [nominal16]({os.path.relpath(paths['nominal'], BASE)}/rollout.json), [author summary]({os.path.relpath(paths['author'], BASE)}/summary.json), [all completed episodes]({os.path.relpath(paths['author'], BASE)}/completed_episodes.json).")
    lines+=['','<!-- END ACTUAL NATIVE FINAL READOUT -->','']
    path=BASE/'README.md';text=path.read_text();marker='<!-- BEGIN ACTUAL NATIVE FINAL READOUT -->'
    if marker in text:text=text[:text.index(marker)].rstrip()+'\n'
    path.write_text(text+'\n'.join(lines))
    print(json.dumps({s:{'nominal':r['nominal16'],'latest500_position_m':r['author_protocol']['latest500']['position_error/mean'],'latest500_timeout_fraction':r['author_protocol']['latest500']['timeout_survival_proxy']} for s,r in result.items()},indent=2))
if __name__=='__main__':main()
