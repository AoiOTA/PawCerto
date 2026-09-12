"""Resolve the released source's actual default task, with no simulator imports."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from .base_config import Cfg
from .released_config import configure_env
from ..actor_critic import AC_Args
from ..ppo import PPO_Args


def as_dict(obj):
    if isinstance(obj, dict): return {k:as_dict(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [as_dict(v) for v in obj]
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,np.generic): return obj.item()
    if isinstance(obj,type) or isinstance(obj,SimpleNamespace):
        return {k:as_dict(v) for k,v in vars(obj).items() if not k.startswith('_') and (not callable(v) or isinstance(v,type))}
    return deepcopy(obj)


def namespace(obj):
    if isinstance(obj,dict):
        mappings={'default_joint_angles','stiffness','damping','sensor_args','privileged_sensor_args','terrain_kwargs'}
        return SimpleNamespace(**{k:deepcopy(v) if k in mappings else namespace(v) for k,v in obj.items()})
    if isinstance(obj,list):return [namespace(v) for v in obj]
    return deepcopy(obj)


def default_config():
    cfg=namespace(as_dict(Cfg));ac=namespace(as_dict(AC_Args))
    configure_env(cfg,ac)
    ppo=as_dict(PPO_Args);ppo['entropy_coef']=0.005
    return {'env':as_dict(cfg),'policy':as_dict(ac),'ppo':ppo,
            'runner':{'num_steps_per_env':48,'save_interval':400},'seed':0}


def resolve_config(config):
    return namespace(config['env'])
