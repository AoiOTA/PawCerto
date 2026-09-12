"""Fixed Learning Force Control B1/Z1 asset data, separate from policy ABI."""
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "third_party/learning-force-control-reference/resources/robots/b1/urdf/b1_plus_z1.urdf"
JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")] + [f"joint{i}" for i in range(1, 7)] + ["jointGripper"]
DEFAULT_POS = [.1,.6,-1.3, -.1,.6,-1.3, .1,1.,-1.3, -.1,1.,-1.3, 0.,1.,-1.8,-.1,0.,0.,0.]
P_GAINS = [180.,180.,300.]*4 + [64.,128.,64.,64.,64.,64.,64.]
D_GAINS = [8.,8.,15.]*4 + [1.5,3.,1.5,1.5,1.5,1.5,1.5]

def source_tree(urdf=URDF):
    # Namespace repair is confined to the derived conversion input.
    urdf = Path(urdf)
    source = urdf.read_text().replace("<robot ", '<robot xmlns:xacro="http://www.ros.org/wiki/xacro" ', 1)
    tree = ET.ElementTree(ET.fromstring(source))
    # Three unused includes coexist with already expanded Z1 links. The source
    # references absent files; do not invoke xacro or alter original bytes.
    for include in tree.findall("{http://www.ros.org/wiki/xacro}include"):
        tree.getroot().remove(include)
    seen = set()
    for material in tree.findall("material"):
        name = material.get("name")
        if name in seen:
            # Retain the duplicate color under a unique unused definition.
            material.set("name", name + "_duplicate")
        seen.add(name)
    for mesh in tree.findall(".//mesh"):
        path = (urdf.parent / mesh.get("filename")).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", str(path))
    return tree
