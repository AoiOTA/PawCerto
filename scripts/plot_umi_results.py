"""Static figures for the completed UMI transfer Pilot and its paired evaluations."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/figures'
STEPS = [0,25,50,75,100]
COLORS = {'Isaac Lab': '#2369a6', 'MuJoCo': '#d87825'}


def load_data():
    train = [json.loads(line) for line in (ROOT/'runs/umi_lab_transfer_4096/metrics.jsonl').read_text().splitlines()]
    lab = json.loads((ROOT/'outputs/isaac/paired-transfer-16.json').read_text())
    cases = {'Isaac Lab': {}, 'MuJoCo': {}}
    for run in lab:
        step = int(Path(run['checkpoint']).stem.split('_')[-1])
        cases['Isaac Lab'][step] = {
            'position': np.array([r['ee_error_m'] for r in run['rows']]).mean(0),
            'orientation': np.array([r['ee_error_rad'] for r in run['rows']]).mean(0),
        }
    for step in STEPS:
        report = json.loads((ROOT/f'outputs/mujoco/umi_lab_transfer/model_{step}/summary.json').read_text())
        cases['MuJoCo'][step] = {
            'position': np.array([r['position_error_mean_m'] for r in report['case_results']]),
            'orientation': np.array([r['orientation_error_mean_rad'] for r in report['case_results']]),
        }
    return train,cases


def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def training_figure(rows):
    x=np.array([r['iteration'] for r in rows])
    fig,axes=plt.subplots(2,2,figsize=(11.8,7.2),layout='constrained')
    for ax,key,scale,label,color in [
        (axes[0,0],'position_error',1000,'EE position error (mm)','#2369a6'),
        (axes[0,1],'orientation_error',1,'EE orientation error (rad)','#d87825')]:
        y=np.array([r[key] for r in rows])*scale
        ax.plot(x,y,color=color,alpha=.35,lw=1.1,label='Each rollout')
        ax.plot(x[4:],np.convolve(y,np.ones(5)/5,mode='valid'),color=color,lw=2,label='Trailing 5-rollout mean')
        ax.set(ylabel=label,xlabel='Training iteration',ylim=(0,None),xlim=(1,100))
        ax.legend(fontsize=8,loc='upper right')
    axes[0,0].set_title('A  Tracking during randomized training',loc='left',fontweight='bold')
    axes[0,1].set_title('B  Orientation remains variable',loc='left',fontweight='bold')
    ax=axes[1,0]
    ax.plot(x,[r['final_policy_kl'] for r in rows],color='#2369a6',lw=1.6,label='Final rollout KL')
    ax.axhline(.01,color='#2369a6',ls=':',lw=1,label='Configured desired KL = 0.01')
    ax.set(yscale='log',ylabel='Final rollout KL',xlabel='Training iteration',xlim=(1,100))
    ax.tick_params(axis='y',labelcolor='#2369a6')
    right=ax.twinx()
    right.plot(x,[r['learning_rate'] for r in rows],color='#a24c86',lw=1.2,label='Learning rate')
    right.set(yscale='log',ylabel='Learning rate',ylim=(8e-5,5e-4))
    right.tick_params(axis='y',labelcolor='#a24c86')
    right.grid(False)
    ax.set_title('C  Large first update; adaptive learning rate',loc='left',fontweight='bold')
    lines,labels=ax.get_legend_handles_labels();rl,rt=right.get_legend_handles_labels()
    ax.legend(lines+rl,labels+rt,fontsize=8,loc='upper right')
    ax=axes[1,1]
    ax.plot(x,[r['pos_sigma'] for r in rows],color='#2369a6',lw=1.8,label=r'Position $\sigma$ (m²)')
    ax.plot(x,[r['orn_sigma'] for r in rows],color='#d87825',lw=1.8,label=r'Orientation $\sigma$ (rad)')
    ax.set(yscale='log',ylabel='Reward scale (rollout average)',xlabel='Training iteration',xlim=(1,100))
    ax.set_title('D  Global error curriculum tightens the reward',loc='left',fontweight='bold')
    ax.legend(fontsize=8,loc='upper right')
    fig.suptitle('Official weights adapted in Isaac Lab: 4096 environments × 24 steps × 100 updates\nTraining curves include noise, resets, pushes and transports; they are not paired test results.',fontsize=12,fontweight='bold')
    save(fig,'umi_transfer_training')


def paired_figure(cases):
    fig,axes=plt.subplots(1,3,figsize=(13.4,4.3),layout='constrained')
    for engine,values in cases.items():
        color=COLORS[engine]
        axes[0].plot(STEPS,[values[k]['position'].mean()*1000 for k in STEPS],'-o',color=color,lw=2,label=engine,ms=5)
        axes[1].plot(STEPS,[values[k]['orientation'].mean() for k in STEPS],'-o',color=color,lw=2,label=engine,ms=5)
        dp=(values[100]['position']-values[0]['position'])*1000
        do=values[100]['orientation']-values[0]['orientation']
        axes[2].scatter(dp,do,color=color,label=engine,s=36,alpha=.8,edgecolor='white',linewidth=.5)
    axes[0].set(ylabel='Mean EE position error (mm)',xlabel='Checkpoint iteration',ylim=(0,15),xticks=STEPS)
    axes[0].set_title('A  Position does not improve overall',loc='left',fontweight='bold')
    axes[1].set(ylabel='Mean EE orientation error (rad)',xlabel='Checkpoint iteration',ylim=(0,.04),xticks=STEPS)
    axes[1].set_title('B  Orientation error increases',loc='left',fontweight='bold')
    axes[2].axvline(0,color='#777777',lw=.8,ls='--');axes[2].axhline(0,color='#777777',lw=.8,ls='--')
    axes[2].set(xlabel='Change in position error (mm)',ylabel='Change in orientation error (rad)')
    axes[2].set_title('C  Final − initial, each paired case',loc='left',fontweight='bold')
    axes[2].text(.03,.98,'Positive = larger error',transform=axes[2].transAxes,va='top',fontsize=8,color='#555555')
    for ax in axes[:2]:
        ax.legend(fontsize=8,loc='lower right')
    axes[2].legend(fontsize=8,loc='lower left')
    fig.suptitle('Nominal paired evaluation: same 16 sampled cases × 17 s in each engine\nAll five checkpoints: 0/16 inverted cases per engine; mean support ≈ 3.95 feet.',fontsize=12,fontweight='bold')
    save(fig,'umi_transfer_nominal_paired')


def scratch_chain(through):
    """Persisted branch: original through 2000, then the separately resumed run."""
    original_path=ROOT/'runs/umi_lab_scratch_4096/metrics.jsonl'
    resume_path=ROOT/'runs/umi_lab_scratch_4096_resume/metrics.jsonl'
    def read_rows(path):
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().rsplit('\n',1)[0].splitlines()]
    original=read_rows(original_path)
    resumed=read_rows(resume_path)
    rows=[r for r in original if r['iteration']<=min(through,2000)]
    rows.extend(r for r in resumed if 2000<r['iteration']<=through)
    if [r['iteration'] for r in rows]!=list(range(1,through+1)):
        raise ValueError(f'Training node {through} is not recorded yet')
    tail=[r for r in original if r['iteration']>2000]
    saved=next((r for r in original if r['iteration']==2000),None)
    tail_transitions=tail[-1]['total_transitions']-saved['total_transitions'] if tail and saved else 0
    tail_seconds=sum(r['iteration_seconds'] for r in tail)
    accounting=dict(
        through_iteration=through,
        effective_chain_sources=[str(original_path),str(resume_path)],
        effective_chain='Original iterations 1–2000 plus resumed iterations 2001 onward; physics state resets at resume',
        effective_chain_transitions=rows[-1]['total_transitions'],
        effective_chain_logged_training_seconds=sum(r['iteration_seconds'] for r in rows),
        original_exit_status=143,
        original_unpersisted_tail_iterations=[r['iteration'] for r in tail],
        original_unpersisted_tail_transitions=tail_transitions,
        original_unpersisted_tail_logged_training_seconds=tail_seconds,
        total_recorded_sampling_including_unpersisted_tail=rows[-1]['total_transitions']+tail_transitions,
        total_recorded_training_seconds_including_unpersisted_tail=sum(r['iteration_seconds'] for r in rows)+tail_seconds,
        accounting_scope='Effective chain through selected node plus observed original unpersisted tail; excludes initialization and evaluation time')
    return rows,accounting


def scratch_figures(through):
    rows,accounting=scratch_chain(through)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'umi_scratch_stage1_training_accounting.json').write_text(json.dumps(accounting,indent=2))
    x=np.array([r['iteration'] for r in rows])
    fig,axes=plt.subplots(2,3,figsize=(13.4,7.2),layout='constrained')
    specs=[('position_error',1000,'EE position error (mm)','#2369a6'),
           ('orientation_error',1,'EE orientation error (rad)','#d87825'),
           ('supported_feet',1,'Mean feet: normal-contact Fz >1 N','#2c8a68')]
    for ax,(key,scale,label,color) in zip(axes[0],specs):
        y=np.array([r[key] for r in rows])*scale
        ax.plot(x,y,color=color,lw=.7,alpha=.25,label='Each rollout')
        window=min(50,len(rows))
        ax.plot(x[window-1:],np.convolve(y,np.ones(window)/window,mode='valid'),color=color,lw=1.8,label=f'Trailing {window}-rollout mean')
        ax.set(xlabel='Training iteration',ylabel=label,xlim=(1,through),ylim=(0,4 if key=='supported_feet' else None))
        ax.legend(fontsize=7,loc='upper right')
    axes[0,0].set_title('A  Position tracking',loc='left',fontweight='bold')
    axes[0,1].set_title('B  Orientation tracking',loc='left',fontweight='bold')
    axes[0,2].set_title('C  Foot force during randomized training',loc='left',fontweight='bold')
    axes[1,0].plot(x,[r['final_policy_kl'] for r in rows],color='#2369a6',lw=.9)
    axes[1,0].axhline(.01,color='#777777',ls=':',label='Desired KL = 0.01')
    axes[1,0].set(xlabel='Training iteration',ylabel='Final rollout KL',xlim=(1,through),ylim=(0,None))
    axes[1,0].set_title('D  Policy update size',loc='left',fontweight='bold')
    axes[1,0].legend(fontsize=7)
    axes[1,1].plot(x,[r['learning_rate'] for r in rows],color='#a24c86',lw=1.)
    axes[1,1].set(xlabel='Training iteration',ylabel='Learning rate',yscale='log',xlim=(1,through))
    axes[1,1].set_title('E  Original adaptive learning rate',loc='left',fontweight='bold')
    axes[1,2].plot(x,[r['pos_sigma'] for r in rows],color='#2369a6',label=r'Position $\sigma$ (m²)')
    axes[1,2].plot(x,[r['orn_sigma'] for r in rows],color='#d87825',label=r'Orientation $\sigma$ (rad)')
    axes[1,2].set(xlabel='Training iteration',ylabel='Reward scale (rollout average)',yscale='log',xlim=(1,through))
    axes[1,2].set_title('F  Global error curriculum',loc='left',fontweight='bold')
    axes[1,2].legend(fontsize=7)
    if through>2000:
        for ax in axes.flat:
            ax.axvline(2000,color='#777777',ls='--',lw=.8)
    chain_note='Resumed from node 2000; physics reset. Original unsaved 2001–2114 tail excluded.' if through>2000 else '4096 environments × 24 steps; includes noise, resets, pushes and transports.'
    fig.suptitle(f'From random initialization: saved node {through} of the planned 4000-update stage\n{chain_note}',fontsize=12,fontweight='bold')
    save(fig,'umi_scratch_stage1_training')
    paired=json.loads((ROOT/'outputs/mujoco/umi_lab_scratch/paired_summary.json').read_text())
    # Later model-selection probes must not rewrite the preselected stage figure.
    stage_nodes={0,500,1000,2000,4000}
    selected=sorted((c for c in paired['comparisons'] if c['iteration']<=through and c['iteration'] in stage_nodes),key=lambda c:c['iteration'])
    incomplete=[c for c in selected if c.get('completion_status')=='incomplete']
    iterations=[c['iteration'] for c in selected if c.get('completion_status')!='incomplete']
    latest=next(c for c in selected if c['iteration']==iterations[-1])
    latest_report=json.loads((ROOT/f"outputs/mujoco/umi_lab_scratch/model_{iterations[-1]}/summary.json").read_text())
    inverted=[c['inverted'] for c in latest_report['case_results']]
    fig,axes=plt.subplots(1,2,figsize=(10.,4.1),layout='constrained')
    for ax,key,scale,label in [(axes[0],'position_error_mean_m',1000,'Episode mean EE position error (mm)'),
                              (axes[1],'orientation_error_mean_rad',1,'Episode mean orientation error (rad)')]:
        per_case=[]
        for iteration in iterations:
            report=json.loads((ROOT/f'outputs/mujoco/umi_lab_scratch/model_{iteration}/summary.json').read_text())
            per_case.append([case[key]*scale for case in report['case_results']])
        per_case=np.array(per_case)
        marked=False
        for case,was_inverted in zip(per_case.T,inverted):
            ax.plot(iterations,case,'o-',color='#b94343' if was_inverted else '#8daec8',
                    alpha=.85 if was_inverted else .45,lw=1.3 if was_inverted else .8,ms=3,
                    label='Inverted at latest complete node' if was_inverted and not marked else None)
            marked=marked or was_inverted
        ax.plot(iterations,per_case.mean(-1),'o-',color='#174c78',lw=2.5,ms=6,label='Mean of 16 paired cases')
        ax.set(xlabel='Checkpoint iteration',ylabel=label,ylim=(0,None),xticks=[c['iteration'] for c in selected])
        for result in incomplete:
            at=result['iteration']
            ax.axvline(at,color='#b94343',ls=':',lw=1)
            ax.text(at,.9,f"{result['complete_cases']}/{result['requested_cases']} complete\n{result['invalid_cases']} numerical failure\nNo full-set mean",transform=ax.get_xaxis_transform(),ha='right',va='top',fontsize=8,color='#963737')
            ax.set_xlim(right=at*1.1)
        ax.legend(fontsize=8,loc='center right' if incomplete else 'best')
    axes[0].set_title('A  Position tracking across saved weights',loc='left',fontweight='bold')
    axes[1].set_title('B  Orientation tracking across the same cases',loc='left',fontweight='bold')
    latest_note=(f"Node {incomplete[-1]['iteration']}: incomplete 16-case evaluation; invalid prefix reported separately."
                 if incomplete else f"Latest node {iterations[-1]}: {latest['inverted_cases']}/16 inverted; mean feet with net Fz >1 N: {latest['supported_feet_mean']:.3f}.")
    fig.suptitle('MuJoCo fixed-weight diagnostic: same 16 training-pool cases × 17 s\n'+latest_note,fontsize=12,fontweight='bold')
    save(fig,'umi_scratch_stage1_mujoco_paired')


def scratch_eval_figure():
    names=['umi-ours-author-eval','umi-scratch-500-author-eval','umi-scratch-4000-author-eval']
    reports=[json.loads((ROOT/f'outputs/isaac/{name}/summary.json').read_text()) for name in names]
    labels=['Official ours','Scratch 500','Scratch 4000']
    colors=['#70818d','#2369a6','#d87825']
    specs=[('position_error/mean',1000,'A  EE position error','mm',None),
           ('orientation_error/mean',1,'B  EE orientation error','rad',None),
           ('timeout_survival_proxy',100,'C  Timeout survival proxy','%',100),
           ('ground_supported_feet/mean',1,'D  Feet with ground Fz >1 N','feet',4),
           ('zero_ground_supported_fraction/mean',100,'E  Zero ground-foot support','% of observed time',None),
           ('electrical_power/mean',1,'F  Author electrical-power estimate','W (formula estimate)',None)]
    fig,axes=plt.subplots(2,3,figsize=(12.8,7.2),layout='constrained')
    for ax,(key,scale,title,unit,upper) in zip(axes.flat,specs):
        values=[r['metrics'][key]*scale for r in reports]
        bars=ax.bar(labels,values,color=colors,width=.65)
        ax.bar_label(bars,labels=[f'{v:.3f}' if max(values)<1 else f'{v:.2f}' for v in values],padding=3,fontsize=8)
        ax.set(ylabel=unit,ylim=(0,upper if upper is not None else max(values)*1.16))
        ax.set_title(title,loc='left',fontweight='bold')
        ax.tick_params(axis='x',labelsize=8)
        ax.grid(axis='x',visible=False)
    fig.suptitle('Same saved-input + author-eval protocol: seed 2026, 250 environments, latest 500 completed episodes\nTerminated episodes retain their executed prefixes. Same training pool; neither holdout nor paired episode targets.',fontsize=11,fontweight='bold')
    save(fig,'umi_scratch_stage1_author_eval')


def scratch_lab_figure():
    fig,axes=plt.subplots(2,3,figsize=(12.8,7.2),layout='constrained')
    for row,(name,label,color) in enumerate([('nominal','Nominal','#2369a6'),('randomized','Randomized + push/transport','#d87825')]):
        reports=json.loads((ROOT/f'outputs/isaac/paired-scratch-{name}-16-summary.json').read_text())['results']
        x=[int(Path(r['checkpoint']).stem.split('_')[-1]) for r in reports]
        for col,(key,scale,title,unit,upper) in enumerate([
            ('mean_position_error_m',1000,'Position tracking','EE position error (mm)',None),
            ('mean_orientation_error_rad',1,'Orientation tracking','EE orientation error (rad)',None),
            ('mean_ground_supported_feet',1,'Ground-foot support','Mean feet with ground Fz >1 N',4.25)]):
            ax=axes[row,col]
            values=[r[key]*scale for r in reports]
            ax.plot(x,values,'o-',color=color,lw=2,ms=5)
            ax.set(xlabel='Checkpoint iteration',ylabel=unit,xticks=x,ylim=(0,upper))
            ax.set_title(f'{label}\n{title}',loc='left',fontweight='bold',fontsize=10)
            if col==2:
                for xx,yy,r in zip(x,values,reports):
                    ax.annotate(f"{len(r['inverted_case_ids'])}/16",(xx,yy),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
                ax.text(.97,.06,'Labels = inverted cases',ha='right',transform=ax.transAxes,fontsize=8)
    fig.suptitle('Isaac Lab fixed-weight paired diagnostic: same sample(16,0), 17 s per case, all 16 retained\nNo autoreset. Nominal and randomized conditions are separate; this is not the author episode evaluation.',fontsize=11,fontweight='bold')
    save(fig,'umi_scratch_stage1_lab_paired')


def repeated_seed_training(through):
    """Compare the recorded learning curves without changing earlier stage figures."""
    seed0,seed0_accounting=scratch_chain(through)
    path=ROOT/'runs/umi_lab_scratch_seed1_4096/metrics.jsonl'
    # A concurrent writer may have an unfinished last line; only complete rows count.
    seed1=[json.loads(line) for line in path.read_text().splitlines(keepends=True)
           if line.endswith('\n') and line.strip()]
    seed1=[row for row in seed1 if row['iteration']<=through]
    if [r['iteration'] for r in seed1]!=list(range(1,through+1)):
        raise ValueError(f'Seed1 node {through} is not completely recorded')
    runs={'Training seed 0':seed0,'Training seed 1':seed1}
    colors={'Training seed 0':'#2369a6','Training seed 1':'#d87825'}
    summary={'through_iteration':through,'fixed_seed1_budget':3500,
             'seed0_accounting':seed0_accounting,
             'boundary':'Current normal-contact EMD proxy retained; independent training repeat, not sensor reconstruction or a formal three-seed study.',
             'training_runs':{}}
    for label,rows in runs.items():
        metric_keys=[key for key,value in rows[0].items() if isinstance(value,(int,float))]
        if not all(np.isfinite(row[key]) for row in rows for key in metric_keys):
            raise ValueError(f'{label} contains a nonfinite training metric')
        summary['training_runs'][label]={
            'logged_iterations':len(rows),'total_transitions':rows[-1]['total_transitions'],
            'logged_training_seconds':sum(r['iteration_seconds'] for r in rows),
            'median_iteration_seconds':float(np.median([r['iteration_seconds'] for r in rows])),
            'first_50_mean':{key:float(np.mean([r[key] for r in rows[:50]])) for key in metric_keys},
            'last_50_mean':{key:float(np.mean([r[key] for r in rows[-50:]])) for key in metric_keys},
            'final_row':rows[-1],
            'final_policy_kl_range':[min(r['final_policy_kl'] for r in rows),max(r['final_policy_kl'] for r in rows)]}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'umi_repeated_seeds_training_readout.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    fig,axes=plt.subplots(2,3,figsize=(13.4,7.2),layout='constrained')
    specs=[('position_error',1000,'EE position error (mm)'),
           ('orientation_error',1,'EE orientation error (rad)'),
           ('supported_feet',1,'Feet: normal-contact Fz >1 N'),
           ('final_policy_kl',1,'Final rollout KL'),('learning_rate',1,'Learning rate')]
    for ax,(key,scale,unit) in zip(axes.flat,specs):
        for label,rows in runs.items():
            x=np.array([r['iteration'] for r in rows]);y=np.array([r[key] for r in rows])*scale
            window=min(50,len(rows))
            ax.plot(x,y,color=colors[label],alpha=.2,lw=.6)
            ax.plot(x[window-1:],np.convolve(y,np.ones(window)/window,mode='valid'),color=colors[label],lw=1.6,label=label)
        ax.set(xlabel='Training iteration',ylabel=unit,xlim=(1,through))
        ax.set_title(unit,loc='left',fontweight='bold')
        if key=='learning_rate':ax.set_yscale('log')
        else:ax.set_ylim(0,4 if key=='supported_feet' else None)
        ax.legend(fontsize=7)
    ax=axes[1,2]
    for label,rows in runs.items():
        for key,style,metric in [('pos_sigma','-','position'),('orn_sigma','--','orientation')]:
            ax.plot([r['iteration'] for r in rows],[r[key] for r in rows],style,color=colors[label],lw=1.2,label=f'{label}: {metric}')
    ax.set(xlabel='Training iteration',ylabel='Reward scale (rollout mean)',yscale='log',xlim=(1,through))
    ax.set_title('Global error curriculum',loc='left',fontweight='bold')
    ax.legend(fontsize=7)
    if through>2000:
        for ax in axes.flat:ax.axvline(2000,color='#777777',ls=':',lw=.8)
    resume_note='Seed 0 resumed at 2000 with physics reset; its unsaved tail is excluded.' if through>2000 else 'Both runs use independent random initialization and the current normal-contact EMD proxy.'
    fig.suptitle(f'Independent training repeat: seed 1 through {through} / 3500 preselected updates\n50-rollout means with raw curves faint. {resume_note}',fontsize=11,fontweight='bold')
    save(fig,'umi_repeated_seeds_training')


def repeated_seed_mujoco(through):
    source=ROOT/'outputs/mujoco/umi_seed1_validation_seed2027/paired_summary.json'
    report=json.loads(source.read_text())
    entries=[(f"Seed 1\n{r['label'].replace('model_','')}",r) for r in report['new_seed1_results']
             if int(r['label'].split('_')[-1])<=through]
    reference_labels={'scratch_500':'Seed 0\n500',
                      'scratch_3500':'Seed 0\n3500\n(post hoc)',
                      'official_ours':'Official\nours'}
    references={r['label']:r for r in report['saved_seed0_and_official_references']}
    entries.extend((reference_labels[label],references[label])
                   for label in ('scratch_500','scratch_3500','official_ours'))
    (OUT/'umi_repeated_seeds_mujoco_readout.json').write_text(json.dumps({
        'source':str(source),'through_iteration':through,
        'sampling':report['sampling'],
        'reference_boundary':'Existing sample(16,2027) results reused; seed0_500 is a same-budget reference, seed0_3500 was selected post hoc.',
        'displayed_results':[dict(display_label=label,**r) for label,r in entries]},indent=2,allow_nan=False))
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.8),layout='constrained')
    for ax,(key,scale,unit) in zip(axes,[('position_error_mean_m',1000,'EE position error (mm)'),
                                     ('orientation_error_mean_rad',1,'EE orientation error (rad)'),
                                     ('ground_supported_feet_mean',1,'Mean feet with ground Fz >1 N')]):
        for i,(label,r) in enumerate(entries):
            if r['complete_cases']!=r['requested_cases'] or r['invalid_cases']:
                ax.text(i,.55,f"{r['complete_cases']}/{r['requested_cases']} complete\n{r['invalid_cases']} invalid\nNo full-set mean",
                        transform=ax.get_xaxis_transform(),ha='center',fontsize=7,color='#b33a35')
                continue
            if key!='ground_supported_feet_mean':
                ax.scatter(np.full(len(r['case_details']),i),[c[key]*scale for c in r['case_details']],
                           color='#777777',s=12,alpha=.35,zorder=2)
            ax.scatter([i],[r[key]*scale],color='#d87825' if label.startswith('Seed 1') else '#2369a6',s=48,zorder=3)
            if key=='ground_supported_feet_mean':
                ax.annotate(f"{r['inverted_cases']}/{r['requested_cases']} inv",(i,r[key]*scale),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
        ax.set(xticks=range(len(entries)),xticklabels=[label for label,_ in entries],ylabel=unit,
               xlim=(-.5,len(entries)-.5),ylim=(0,4.5 if key=='ground_supported_feet_mean' else None))
        ax.set_title(unit,loc='left',fontweight='bold')
    fig.suptitle('Independent seed 1: MuJoCo fixed-weight diagnostics on the same sample(16,2027)\nColored dots = full-set means; gray points = individual case means. Existing references reused; training-pool targets.',fontsize=11,fontweight='bold')
    save(fig,'umi_repeated_seeds_mujoco')


def repeated_seed_author_eval():
    base=ROOT/'outputs/isaac/author-validation-seed2027'
    labels=['Official\nours','Seed 0 / 3500\n(post hoc)','Seed 1 / 3500\n(preselected)']
    reports=[json.loads((base/name/'summary.json').read_text()) for name in ('ours','scratch3500','seed1_3500')]
    fig,axes=plt.subplots(2,3,figsize=(12.8,7.2),layout='constrained')
    specs=[('position_error/mean',1000,'EE position error (mm)'),
           ('orientation_error/mean',1,'EE orientation error (rad)'),
           ('timeout_survival_proxy',100,'Timeout survival proxy (%)'),
           ('ground_supported_feet/mean',1,'Feet: ground-filtered normal Fz >1 N'),
           ('zero_ground_supported_fraction/mean',100,'Time with zero ground feet (%)'),
           ('electrical_power/mean',1,'Author electrical estimator (W)')]
    for ax,(key,scale,unit) in zip(axes.flat,specs):
        for i,r in enumerate(reports):
            assert r['included_completed_episodes']==500
            value=r['metrics'][key]*scale
            ax.scatter([i],[value],s=55,color=('#617887','#2369a6','#d87825')[i],zorder=3)
            text=f'{value:.5f}' if key=='orientation_error/mean' else f'{value:.4f}' if key in ('ground_supported_feet/mean','zero_ground_supported_fraction/mean') else f'{value:.2f}'
            if key=='timeout_survival_proxy':text+=f"\n{r['metrics']['inverted_completed_episodes']}/500 inv"
            ax.annotate(text,(i,value),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
        values=[r['metrics'][key]*scale for r in reports]
        upper=110 if key=='timeout_survival_proxy' else 4.35 if key=='ground_supported_feet/mean' else max(values)*1.25
        ax.set(xticks=range(3),xticklabels=labels,xlim=(-.5,2.5),ylim=(0,upper),ylabel=unit)
        ax.set_title(unit,loc='left',fontweight='bold')
    fig.suptitle('Independent training repeat: same author-style evaluation seed 2027, latest 500 completed episodes\nTimeout is a survival proxy; early terminations remain included. Current normal-contact EMD proxy; power is not hardware measurement.',fontsize=11,fontweight='bold')
    save(fig,'umi_repeated_seeds_author_eval')


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,
                         'axes.titlepad':9,'savefig.facecolor':'white'})
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch-through',type=int,help='Saved from-scratch node to summarize')
    parser.add_argument('--scratch-eval',action='store_true',help='Compare official, scratch500 and scratch4000 completed-episode evaluations')
    parser.add_argument('--scratch-lab',action='store_true',help='Compare completed nominal and randomized Isaac Lab paired diagnostics')
    parser.add_argument('--repeated-through',type=int,help='Saved independent seed1 node to compare with seed0 learning curves')
    parser.add_argument('--repeated-eval',action='store_true',help='Compare completed author-style seed2027 official and both training seeds at3500')
    args=parser.parse_args()
    if args.repeated_eval:
        repeated_seed_author_eval()
        print(OUT/'umi_repeated_seeds_author_eval.png')
    elif args.repeated_through is not None:
        repeated_seed_training(args.repeated_through)
        repeated_seed_mujoco(args.repeated_through)
        print(OUT/'umi_repeated_seeds_training.png')
        print(OUT/'umi_repeated_seeds_mujoco.png')
    elif args.scratch_lab:
        scratch_lab_figure()
        print(OUT/'umi_scratch_stage1_lab_paired.png')
    elif args.scratch_eval:
        scratch_eval_figure()
        print(OUT/'umi_scratch_stage1_author_eval.png')
    elif args.scratch_through is not None:
        scratch_figures(args.scratch_through)
        print(OUT/'umi_scratch_stage1_training.png')
        print(OUT/'umi_scratch_stage1_mujoco_paired.png')
    else:
        rows,cases=load_data()
        training_figure(rows)
        paired_figure(cases)
        print(OUT/'umi_transfer_training.png')
        print(OUT/'umi_transfer_nominal_paired.png')


if __name__=='__main__':
    main()
