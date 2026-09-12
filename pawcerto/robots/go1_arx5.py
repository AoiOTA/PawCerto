"""RoboDuet Go1/ARX5 source data shared by both simulator asset consumers.

The 20 simulation DOFs include two gripper joints. This ordering does not
define the method's 18-dimensional policy action or change its controllers.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / 'third_party/roboduet-reference/resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf'
JOINT_NAMES = [f'{leg}_{joint}_joint' for leg in ('FL', 'FR', 'RL', 'RR') for joint in ('hip', 'thigh', 'calf')] + [f'zarx_j{i}' for i in range(1, 9)]
DEFAULT_POS = [.1, .8, -1.5, -.1, .8, -1.5, .1, 1., -1.5, -.1, 1., -1.5, 0., .8, .8, 0., 0., 0., 0., 0.]


def source_tree(urdf=URDF):
    """Resolve this source URDF's mesh paths relative to its own directory."""
    urdf = Path(urdf)
    tree = ET.parse(urdf)
    for mesh in tree.findall('.//mesh'):
        path = (urdf.parent / mesh.get('filename')).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set('filename', str(path))
    return tree
