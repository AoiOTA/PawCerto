"""Bounded CPU comparison of experimental PRE six-axis residual with original Gym."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_deepwbc_wrench import metrics, SCENARIOS, rotation

def analyze(gym_path,lab_path,output):
    gym=np.load(gym_path); lab=np.load(lab_path)
    result={'boundary':'Separate-engine short trajectories; this is a signal hypothesis test, not default signal equivalence or learning acceptance. Nonzero foot COM lever has analytic CPU coverage only.', 'scenarios':{}}
    for name in SCENARIOS:
        gm=gym['scenario']==name; lm=lab['scenario']==name
        if gm.sum()!=lm.sum(): raise ValueError('Protocol lengths differ')
        g=gym['sensor_modes_wrench'][gm]; candidate=lab['candidate'][lm]; residual=lab['solver_only'][lm]
        last=(lab['step'][lm]+(200 if name=='static_contact' else 1)+1)%4==0
        result['scenarios'][name]={
            'all_5ms_both':metrics(candidate,g[:,1]),'consumer_last_5ms_both':metrics(candidate[last],g[last,1]),
            'all_5ms_constraint':metrics(residual,g[:,3]),'consumer_last_5ms_constraint':metrics(residual[last],g[last,3]),
            'fd_gravity':metrics(candidate-residual,g[:,2]),
            'candidate_axis_mean':candidate.mean((0,1)).tolist(),'gym_both_axis_mean':g[:,1].mean((0,1)).tolist(),
            'foot_position_rmse_m':float(np.sqrt(np.mean((gym['link_pose'][gm,...,:3]-lab['link_pose'][lm,...,:3])**2))),
            'lab_vs_prior_rawnet_max_position_difference_m':None,
            'max_lab_foot_linear_speed':float(np.linalg.norm(lab['post_velocity'][lm,...,:3],axis=-1).max()),
            'max_lab_foot_angular_speed':float(np.linalg.norm(lab['post_velocity'][lm,...,3:],axis=-1).max())}
    previous=lab_path.parent.parent/'deepwbc-wrench-lab-20260912'/'samples.npz'
    if previous.exists():
        old=np.load(previous)
        for name in SCENARIOS:
            result['scenarios'][name]['lab_vs_prior_rawnet_max_position_difference_m']=float(np.abs(lab['link_pose'][lab['scenario']==name,...,:3]-old['link_pose'][old['scenario']==name,...,:3]).max())
    properties=json.loads((lab_path.parent/'properties.json').read_text())
    ids=[properties['body_names'].index(n) for n in ('FL_foot','FR_foot','RL_foot','RR_foot')]
    frame_reward={}
    for scene in SCENARIOS:
        lm=lab['scenario']==scene; gm=gym['scenario']==scene
        candidate=lab['candidate'][lm]; original=gym['sensor_modes_wrench'][gm,1]
        last=(lab['step'][lm]+(200 if scene=='static_contact' else 1)+1)%4==0
        lr=lab['pre_link_rotation'][lm,0][:,ids]
        gr=rotation(gym['link_pose'][gm][:-1,:,3:])
        relative=np.swapaxes(gr,-1,-2)@lr[1:]
        aligned=np.concatenate([np.einsum('...ij,...j->...i',relative,candidate[1:,:,:3]),np.einsum('...ij,...j->...i',relative,candidate[1:,:,3:])],-1)
        lv=lab['pre_com_velocity'][lm,0][:,ids][1:]; gv=gym['post_velocity'][gm][:-1]
        squared=original[last,:,2]**2; csquared=candidate[last,:,2]**2
        frame_reward[scene]={
            'frame_boundary':'First row omitted only for frame/velocity analysis: Gym PRE is previous recorded POST. Rotation alignment is diagnosis, not a changed candidate.',
            'original_same_rows':metrics(candidate[1:],original[1:]),'lab_candidate_rotated_to_gym_pre_frame':metrics(aligned,original[1:]),
            'pre_linear_velocity_rmse_m_s':float(np.sqrt(np.mean((lv[...,:3]-gv[...,:3])**2))),
            'pre_angular_velocity_rmse_rad_s':float(np.sqrt(np.mean((lv[...,3:]-gv[...,3:])**2))),
            'consumer_gym_fz_squared_mean':float(squared.mean()),'consumer_gym_fz_squared_median':float(np.median(squared)),
            'consumer_gym_fz_squared_p95':float(np.quantile(squared,.95)),'consumer_gym_fz_squared_max':float(squared.max()),
            'consumer_candidate_fz_squared_mean':float(csquared.mean()),
            'consumer_squared_mae_relative_to_gym_mean':float(np.abs(csquared-squared).mean()/squared.mean()),
            'consumer_squared_signed_bias_relative_to_gym_mean':float((csquared-squared).mean()/squared.mean())}
    (output.parent/'frame-reward-analysis.json').write_text(json.dumps(frame_reward,indent=2)+'\n')
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gym',type=Path,required=True);p.add_argument('--lab',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(analyze(a.gym,a.lab,a.output),indent=2))
