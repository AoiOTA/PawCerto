"""CPU Newton-Euler candidate and bounded Gym/Lab wrench comparison (xyzw)."""
import argparse
import json
from pathlib import Path
import numpy as np

FEET = ('FL_foot', 'FR_foot', 'RL_foot', 'RR_foot')
SCENARIOS = ('static_contact', 'free_fall', 'moving_contact')

def rotation(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x,y,z,w = np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),axis=-1).reshape(q.shape[:-1]+(3,3))

def net_wrench(mass, inertia_b, quat, omega_w, acceleration_w, com_offset_b):
    """Net body force, moment at body origin; no gravity/free-dynamics subtraction.

    inertia_b is the COM inertia expressed in body axes, already a 3x3 matrix.
    acceleration_w contains COM linear and angular acceleration in world axes.
    """
    r = rotation(quat)
    iw = r @ inertia_b @ np.swapaxes(r,-1,-2)
    mv = lambda a,b: np.einsum('...ij,...j->...i',a,b)
    force = np.asarray(mass)[...,None]*acceleration_w[...,:3]
    moment = mv(iw,acceleration_w[...,3:]) + np.cross(omega_w,mv(iw,omega_w)) + np.cross(mv(r,com_offset_b),force)
    return np.concatenate((mv(np.swapaxes(r,-1,-2),force),mv(np.swapaxes(r,-1,-2),moment)),axis=-1)

def metrics(a,b):
    d = a-b
    return dict(axis_mae=np.mean(np.abs(d),axis=(0,1)).tolist(),axis_rmse=np.sqrt(np.mean(d*d,axis=(0,1))).tolist(),axis_max_abs=np.max(np.abs(d),axis=(0,1)).tolist(),contactbit_disagreement=float(np.mean((np.linalg.norm(a,axis=-1)>1.5)!=(np.linalg.norm(b,axis=-1)>1.5))),fz_squared_mae=float(np.mean(np.abs(a[...,2]**2-b[...,2]**2))))

def sensor_ledger(path):
    data=np.load(path)
    meta=json.loads((path.parent/'metadata.json').read_text())
    result={'source':str(path),'boundary':'Same-body simultaneous channels; FD velocity differences are sensitivity only','scenarios':{}}
    for scenario in SCENARIOS:
        select=data['scenario']==scenario
        modes=data['sensor_modes_wrench'][select]
        default,both,fd,constraint=np.moveaxis(modes,1,0)
        result['scenarios'][scenario]={'default_vs_explicit_both':metrics(default,both),'both_vs_fd_plus_constraint':metrics(both,fd+constraint),'default_vs_velocity_difference':metrics(default,data['fd_candidate'][select]),'axis_mean_by_mode':{n:modes[:,i].mean((0,1)).tolist() for i,n in enumerate(meta['sensor_mode_order'])},'max_foot_speed':float(np.linalg.norm(data['post_velocity'][select,:,:3],axis=-1).max()),'contact_fraction':float(np.mean(np.linalg.norm(data['contact_force'][select],axis=-1)>1.5))}
    dof=json.loads((path.parent/'dof-properties.json').read_text()); j=dof['joint_names'].index('widow_waist')
    result['waist']={'asset':{k:v[j] for k,v in dof['asset'].items()},'actor':{k:v[j] for k,v in dof['actor'].items()},'observed_position_min':float(data['joint_pos'][:,j].min()),'observed_position_max':float(data['joint_pos'][:,j].max())}
    return result

def fd_gravity(path):
    data=np.load(path); model=np.load(path.parent/'static.npz')
    result={'gravity':[0,0,-9.81],'lever_note':'Actual four COM offsets are zero, so nonzero COM lever is untested.','frame_note':'Previous recorded post frame is next step pre frame; first sample of each scenario is omitted only for that comparison.','scenarios':{}}
    for name in SCENARIOS:
        select=data['scenario']==name
        r=rotation(data['link_pose'][select,:,3:])
        fw=np.broadcast_to(model['mass'][:,None]*np.array([0,0,-9.81]),(select.sum(),4,3))
        force=np.einsum('...ji,...j->...i',r,fw)
        predicted=np.concatenate((force,np.cross(model['com'],force)),axis=-1)
        actual=data['sensor_modes_wrench'][select,2]
        result['scenarios'][name]={'post_frame_gravity_vs_fd_only':metrics(predicted,actual),'previous_post_frame_gravity_vs_next_fd_only':metrics(predicted[:-1],actual[1:]),'gravity_predicted_axis_mean':predicted.mean((0,1)).tolist(),'fd_only_axis_mean':actual.mean((0,1)).tolist()}
    return result

def friction_slide(path):
    data=np.load(path); meta=json.loads((path.parent/'slide-metadata.json').read_text()); dt=meta['dt']
    result={'metadata':meta,'boundary':'Effective motion and breakaway in these box-plane conditions only; scalar Gym shape friction has no exposed independent dynamic coefficient or combine rule.','materials':[]}
    for i,value in enumerate(meta['values']):
        mask=data['phase']=='slide'; pre=data['pre_state'][mask,i]; post=data['post_state'][mask,i]; contact=data['contact_force'][mask,i]
        moving=(pre[:,7]>.4)&(data['step'][mask]>=5)
        ax=(post[:,7]-pre[:,7])/dt
        dyn={'moving_samples':int(moving.sum()),'mean_deceleration':float(-ax[moving].mean()),'effective_mu_from_deceleration':float(-ax[moving].mean()/9.81),'deceleration_std':float(ax[moving].std()),'initial_vx':float(pre[0,7]),'final_vx':float(post[-1,7]),'distance_x':float(post[-1,0]-pre[0,0]),'mean_contact_force':contact[moving].mean(0).tolist(),'max_vertical_speed':float(np.abs(post[moving,9]).max()),'max_angular_speed':float(np.linalg.norm(post[moving,10:13],axis=-1).max())}
        mask=data['phase']=='push'; state=data['post_state'][mask,i]; applied=data['applied_force'][mask,i,0]; moving=np.flatnonzero(np.abs(state[:,7])>.02)
        onset=int(moving[0]) if len(moving) else None
        push={'slip_vx_threshold':.02,'first_slip_step':onset,'force_over_mg_at_slip':float(applied[onset]/meta['materials'][i]['mass']/9.81) if onset is not None else None,'angular_speed_at_slip':float(np.linalg.norm(state[onset,10:13])) if onset is not None else None}
        result['materials'].append({'requested':value,'dynamic':dyn,'push':push})
    return result

def analyze(gym_path,lab_path,output):
    gym=np.load(gym_path); lab=np.load(lab_path)
    report={'boundary':'Separate engine trajectories; state mismatch is a confound. Native joint reaction and velocity-difference sensitivity are not default Gym equivalence.', 'scenarios':{}}
    for name in SCENARIOS:
        g=gym['scenario']==name; l=lab['scenario']==name
        if g.sum()!=l.sum(): raise ValueError('Mismatched protocol lengths')
        gw=gym['raw_wrench'][g]; cand=lab['candidate'][l]; native=lab['native_wrench'][l]
        policy_last=(lab['step'][l]+(200 if name=='static_contact' else 1)+1)%4==0
        report['scenarios'][name]={'samples':int(g.sum()),'net_vs_gym':metrics(cand,gw),'policy_last_net_vs_gym':metrics(cand[policy_last],gw[policy_last]),'policy_last_native_vs_gym':metrics(native[policy_last],gw[policy_last]),'gym_max_foot_speed':float(np.linalg.norm(gym['post_velocity'][g,...,:3],axis=-1).max()),'lab_max_foot_speed':float(np.linalg.norm(lab['post_velocity'][l,...,:3],axis=-1).max()),'native_vs_gym':metrics(native,gw),'lab_fd_vs_reported':metrics(lab['fd_candidate'][l],cand),'gym_fd_vs_sensor':metrics(gym['fd_candidate'][g],gw),'foot_position_rmse':float(np.sqrt(np.mean((gym['link_pose'][g,...,:3]-lab['link_pose'][l,...,:3])**2))),'gym_wrench_axis_mean':gw.mean((0,1)).tolist(),'lab_candidate_axis_mean':cand.mean((0,1)).tolist(),'gym_contact_fraction':float(np.mean(np.linalg.norm(gym['contact_force'][g],axis=-1)>1.5)),'lab_contact_fraction':float(np.mean(np.linalg.norm(lab['contact_force'][l],axis=-1)>1.5))}
    gs=np.load(gym_path.parent/'static.npz'); ls=np.load(lab_path.parent/'static.npz')
    report['model_difference']={k:dict(gym=gs[k].tolist(),lab=ls[k].tolist(),max_abs_difference=float(np.abs(gs[k]-ls[k]).max())) for k in ('mass','com','inertia')}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--gym',type=Path,required=True); p.add_argument('--lab',type=Path); p.add_argument('--output',type=Path,required=True); p.add_argument('--fd-gravity',action='store_true'); p.add_argument('--friction-slide',action='store_true'); a=p.parse_args()
    if a.lab: analyze(a.gym,a.lab,a.output)
    else:
        result=fd_gravity(a.gym) if a.fd_gravity else friction_slide(a.gym) if a.friction_slide else sensor_ledger(a.gym); a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
