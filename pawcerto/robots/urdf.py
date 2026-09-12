"""Named URDF limits shared by simulator-specific robot adapters."""
import math
import xml.etree.ElementTree as ET


def read_joint_limits(path, joint_names):
    """Return source limits in the caller's joint order, without applying control overrides."""
    names = list(joint_names)
    if len(set(names)) != len(names):
        raise ValueError('Joint order contains duplicate names')
    joints = {joint.get('name'): joint for joint in ET.parse(path).findall('joint')}
    result = {}
    for name in names:
        if name not in joints:
            raise ValueError(f'URDF has no joint {name!r}')
        limit = joints[name].find('limit')
        if limit is None:
            raise ValueError(f'URDF joint {name!r} has no limits')
        try:
            values = {key: float(limit.attrib[key]) for key in ('lower', 'upper', 'effort', 'velocity')}
        except (KeyError, ValueError) as exc:
            raise ValueError(f'Invalid URDF limits for joint {name!r} in {path}: {exc}') from exc
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f'Nonfinite URDF limit for {name!r}')
        result[name] = values
    return result
