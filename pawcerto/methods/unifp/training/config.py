"""Resolved, JSON-compatible original UniFP configuration."""
from copy import deepcopy
from types import SimpleNamespace
from ..config import B2Z1PosForceRoughCfg, B2Z1PosForceRoughCfgPPO


def class_to_dict(obj):
    if isinstance(obj, dict):
        return {k: class_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [class_to_dict(v) for v in obj]
    if hasattr(obj, '__dict__'):
        return {k: class_to_dict(getattr(obj, k)) for k in dir(obj)
                if not k.startswith('_') and not callable(getattr(obj, k))}
    return obj


def default_config():
    return {'env': class_to_dict(B2Z1PosForceRoughCfg()),
            'train': class_to_dict(B2Z1PosForceRoughCfgPPO())}


def resolve_config(config):
    """Create an independent attribute config while retaining source mapping fields."""
    mapping_fields = {'default_joint_angles', 'stiffness', 'damping', 'terrain_dict', 'terrain_kwargs'}
    def convert(value, key=None):
        if isinstance(value, dict) and key not in mapping_fields:
            return SimpleNamespace(**{k: convert(v, k) for k, v in value.items()})
        return deepcopy(value)
    return convert(config['env'])
