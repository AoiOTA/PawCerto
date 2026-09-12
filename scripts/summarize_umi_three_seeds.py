"""Final fixed-seed comparison; execute only after all three CPU scopes complete."""
import hashlib
import argparse
import os
import umi_readout_metrics as node
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1], help='Root of the historical result/run tree')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--plot', action='store_true', help='Also render training plots (requires matplotlib)')
args = parser.parse_args()
ROOT = args.root.resolve()
BASE = args.output.resolve()
BASE.mkdir(parents=True, exist_ok=False)
DIRS={s:ROOT/f'outputs/mujoco/umi_reconstructed_solver_relaxed_body_speed_scratch_seed{s}_seed2027' for s in range(3)}
OFF=ROOT/'outputs/mujoco/version-3.13-regression/official_ours'
INPUTS={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    INPUTS[str(p)]=sha(p)
    return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
    summaries={}
    results={}
    training={}
    contacts={}
    coordination={}
    for seed,directory in DIRS.items():
        execution=load(directory/'verification-execution.json')
        assert execution['requested_scope_complete'], (seed,'CPU scope incomplete')
        summary=load(directory/'summary.json')
        assert summary['actual_evaluated_iterations']==[0,500,4000]
        summaries[str(seed)]=summary
        for iteration in (0,500,4000):
            name=f'seed{seed}_model{iteration}'
            results[name]=[load(directory/f'model_{iteration}/case_{i:02d}.json') for i in range(16)]
        run=ROOT/f'runs/umi_reconstructed_solver_relaxed_body_speed_scratch_seed{seed}_4096'
        path=run/'metrics.jsonl';INPUTS[str(path)]=sha(path)
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        assert len(rows)==4000 and [r['iteration'] for r in rows]==list(range(1,4001))
        assert all(r['total_transitions']==r['iteration']*98304 for r in rows)
        assert all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float)))
        fields=['position_error','orientation_error','reward','terminated','supported_feet','constraint/even_mass_distribution','learning_rate','final_policy_kl','action_std','pos_sigma','orn_sigma']
        phases={}
        for start,end in [(1,500),(501,1000),(1001,2000),(2001,3000),(3001,4000)]:
            phases[f'{start}-{end}']={key:float(np.mean([r[key] for r in rows[start-1:end]])) for key in fields}
        training[str(seed)]=dict(iterations=4000,transitions=rows[-1]['total_transitions'],phase_means=phases,last_iteration=rows[-1],metrics_jsonl=str(path),phase_boundary='Fixed descriptive iteration bands; no checkpoint selection or success threshold.')
        contacts[str(seed)]=load(directory/'contact-quality-readout.json')
        coordination[str(seed)]=load(directory/'coordination_readout.json')
    results['official']=[load(OFF/f'case_{i:02d}.json') for i in range(16)]
    final_names=[f'seed{s}_model4000' for s in range(3)]+['official']
    final_common=sorted(set.intersection(*(set(i for i,r in enumerate(results[n]) if r['completion_status']=='complete') for n in final_names)))
    final_compare={n:node.aggregate(results[n],final_common) for n in final_names}
    nodes={n:dict(complete_indices=[i for i,r in enumerate(rs) if r['completion_status']=='complete'],invalid_indices=[i for i,r in enumerate(rs) if r['completion_status']!='complete'],inverted_indices=[i for i,r in enumerate(rs) if r.get('inverted',False)],own_complete_subset=node.aggregate(rs,[i for i,r in enumerate(rs) if r['completion_status']=='complete'])) for n,rs in results.items()}
    final_prefixes=[]
    for i in range(16):
        available={}
        for name in final_names:
            if results[name][i]['completion_status']=='entry_failed':continue
            directory=OFF if name=='official' else DIRS[int(name[4])]/'model_4000'
            path=directory/f'case_{i:02d}.npz';INPUTS[str(path)]=sha(path)
            with np.load(path,allow_pickle=False) as z:available[name]={k:z[k].copy() for k in z.files}
        arrays=list(available.values())
        for a in arrays[1:]:
            assert np.array_equal(a['sampled_target_positions'],arrays[0]['sampled_target_positions'])
            assert np.array_equal(a['sampled_target_rotations'],arrays[0]['sampled_target_rotations'])
        count=min(len(a['metrics']) for a in arrays) if len(arrays)==len(final_names) else 0
        if count:assert all(np.array_equal(a['metrics'][:count,0],arrays[0]['metrics'][:count,0]) for a in arrays)
        final_prefixes.append(dict(case_index=i,shared_count=count,last_time_s=float(arrays[0]['metrics'][count-1,0]) if count else None,metrics={name:node.metrics({k:a[k][:count] for k in ('metrics','ground_supported_feet')}) for name,a in available.items()} if count else {},statuses={name:results[name][i]['completion_status'] for name in final_names}))
    report=dict(status='all_three_fixed_4000_cpu_scopes_complete',training_seeds=[0,1,2],evaluation_seed=2027,requested_cases=16,requested_seconds=17,integration_timestep_seconds=.005,nodes=nodes,final_common_complete_indices=final_common,final_common_complete_comparison=final_compare,final_same_clock_prefixes=final_prefixes,training=training,source_reports={str(s):str(d/'README.md') for s,d in DIRS.items()},contact_quality_reports={str(s):str(d/'contact-quality-readout.md') for s,d in DIRS.items()},coordination_reports={str(s):str(d/'coordination_readout.md') for s,d in DIRS.items()},boundary='Aggregation of saved evidence only; historical source-code hashes are retained in source reports, not revalidated against current code. Three fixed seeds on the same trajectory pool. Main final comparison uses only shared complete cases; each node retains own coverage and failures, and final prefixes use matched clocks. Training bands describe stochastic training measurements, not deterministic evaluation. No sensor equivalence, causal coordination, independent trajectory holdout, sufficient statistical power, or formal WBC acceptance claim.',input_sha256=INPUTS)
    assert all(sha(p)==h for p,h in INPUTS.items())
    dump(BASE/'summary.json',report)
    lines=['# Three fixed training seeds: MuJoCo result','',report['boundary'],'','All three runs use random initialization and the fixed 4000-iteration endpoint: 393,216,000 transitions each. No intermediate checkpoint was chosen for the final comparison.','',f'Final common-complete denominator: **{len(final_common)}/16**; indices: {final_common}.','', '| Policy | EE mean (mm) | Orientation (rad) | Min up | Ground feet mean | Zero ground-foot % | External / self nonfoot endpoints |','|---|---:|---:|---:|---:|---:|---:|']
    for name,m in final_compare.items():
        if m is None:lines.append(f'| {name} | unavailable | | | | | |');continue
        lines.append(f"| {name} | {m['position_error_mean_m']*1000:.3f} | {m['orientation_error_mean_rad']:.5f} | {m['minimum_root_up_dot']:.5f} | {m['ground_supported_feet_mean']:.4f} | {m['zero_ground_supported_feet_fraction']*100:.4f} | {m['nonfoot_external_contact_samples_gt_1N']} / {m['nonfoot_self_contact_samples_gt_1N']} |")
    lines+=['','Every node below retains its own full-horizon coverage. Its own-complete metric may have a different denominator from other rows; use each seed report for matched initial/final comparisons.','','| Node | Complete /16 | Invalid indices | Inverted indices | Own-complete EE (mm) | Own-complete orientation (rad) |','|---|---:|---|---|---:|---:|']
    for name,n in nodes.items():
        m=n['own_complete_subset'];ee=f"{m['position_error_mean_m']*1000:.3f}" if m else 'unavailable';ori=f"{m['orientation_error_mean_rad']:.5f}" if m else 'unavailable'
        lines.append(f"| {name} | {len(n['complete_indices'])} | {n['invalid_indices']} | {n['inverted_indices']} | {ee} | {ori} |")
    lines+=['','Nonfoot counts are distinct retained endpoint times per case with any external nonfoot force norm >1 N. Ground-load fraction sums upward nonfoot Fz over all upward external ground Fz, including feet/fingers. These are sampled load descriptors, not continuous support or impulses. The true initial 20 ms warmup is unsaved; the existing 1 s descriptive split is retained.','','| Seed | External endpoints: all / before 1s / later | Nonfoot fraction of all ground upward Fz (%) | Max endpoint fraction (%) |','|---|---:|---:|---:|']
    for seed,c in contacts.items():
        newkey=next(k for k in c['groups'] if k!='official');w=c['groups'][newkey]['windows']
        lines.append(f"| {seed} | {w['all']['nonfoot_external_gt1_endpoints']} / {w['startup_before_1s']['nonfoot_external_gt1_endpoints']} / {w['at_or_after_1s']['nonfoot_external_gt1_endpoints']} | {w['all']['nonfoot_fraction_of_summed_ground_up_Fz']*100:.5f} | {w['all']['endpoint_fraction_max']*100:.5f} |")
    lines+=['','A low pooled force fraction can hide strong local nonfoot contact. The table below retains each observed nonfoot ground body, including case indices and peak force norm, rather than interpreting no inversion as contact quality. Counts are sampled body-endpoints above the existing1N threshold.','','| Seed | Nonfoot body | Cases above1N | Active body-endpoints | Maximum contact force norm (N) |','|---|---|---|---:|---:|']
    peak_notes=[]
    for seed,c in contacts.items():
        newkey=next(k for k in c['groups'] if k!='official');g=c['groups'][newkey]
        for name,b in sorted(g['bodies'].items(),key=lambda item:-item[1]['max_force_norm_N']):
            lines.append(f"| {seed} | {name} | {b['cases_with_gt1']} | {b['body_endpoint_count_gt1']} | {b['max_force_norm_N']:.3f} |")
        peak=max(g['cases'],key=lambda case:case['windows']['all']['endpoint_fraction_max'])
        peak_notes.append(f"\nSeed {seed}: the largest sampled nonfoot fraction of upward ground Fz is {peak['windows']['all']['endpoint_fraction_max']*100:.4f}% in case {peak['case_index']}. This local load and the body contacts above remain distinct from the pooled fraction.\n")
    lines+=peak_notes
    lines+=['','Base movement and foot clearance document simultaneous physical behavior; they do not prove necessary coordination or gait quality. See the existing saved-state readouts together with tracking, support, contact and inversion evidence.','']
    for seed,d in DIRS.items():lines.append(f'- Seed {seed}: [full result]({os.path.relpath(d, BASE)}/README.md), [coordination]({os.path.relpath(d, BASE)}/coordination_readout.md), [contact quality]({os.path.relpath(d, BASE)}/contact-quality-readout.md).')
    lines += ['', '[Machine-readable summary](summary.json)', '']
    if args.plot:
        lines += ['[Training and evaluation chart](training-and-evaluation.png) · [PDF](training-and-evaluation.pdf)', '']
    (BASE/'README.md').write_text('\n'.join(lines))
    if not args.plot:
        print(json.dumps(final_compare, indent=2))
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13.6,9),constrained_layout=True)
    for seed,t in training.items():
        path=Path(t['metrics_jsonl']);rows=[json.loads(x) for x in path.read_text().splitlines()]
        for ax,key,label,factor in [(axes[0,0],'position_error','Training position error (mm)',1000),(axes[0,1],'orientation_error','Training orientation error (rad)',1)]:
            # Disjoint100-iteration means preserve all4000 rows; descriptive smoothing only.
            ax.plot(np.arange(50.5,4000,100),np.asarray([np.mean([r[key] for r in rows[i:i+100]]) for i in range(0,4000,100)])*factor,label=f'Seed {seed}')
            ax.set(xlabel='Training iteration',ylabel=label);ax.grid(alpha=.2)
    for ax,key,label,factor in [(axes[1,0],'position_error_mean_m','MuJoCo EE mean (mm)',1000),(axes[1,1],'orientation_error_mean_rad','MuJoCo orientation mean (rad)',1)]:
        for seed in range(3):
            paired=summaries[str(seed)]['common_complete_comparison']
            values=[paired[f'model_{it}'] for it in (0,500,4000)]
            coverage=len(summaries[str(seed)]['common_complete_indices'])
            curve,=ax.plot([0,500,4000],[v[key]*factor if v else np.nan for v in values],marker='o',label=f'Seed {seed} ({coverage}/16)')
            official=paired['official']
            if official is not None:ax.axhline(official[key]*factor,color=curve.get_color(),ls='--',alpha=.7,label=f'Official paired seed {seed} ({coverage}/16)')
        ax.set(xlabel='Fixed evaluated checkpoint',ylabel=label);ax.set_xticks([0,500,4000]);ax.grid(alpha=.2)
    for ax in axes.flat:ax.legend()
    fig.suptitle('Three fixed training seeds — same physical variant and trajectory pool\nTraining: all rows in100-iteration means; MuJoCo: per-seed all-node complete intersection; see table for coverage/failures',fontsize=13)
    fig.savefig(BASE/'training-and-evaluation.png',dpi=160);fig.savefig(BASE/'training-and-evaluation.pdf');plt.close(fig)
    print(json.dumps(dict(status=report['status'],final_common_complete_indices=final_common,final_comparison=final_compare),indent=2))

if __name__=='__main__':main()
