"""CPU preparation of the nominal vendor AS2/Piper-H for official Lab import."""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from pawcerto.mujoco.as2_piper_asset import CONFIG, assemble


def prepare_urdf(output, config_path=CONFIG):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text())
    root = assemble(config=config)
    ET.ElementTree(root).write(output / 'source.urdf', encoding='utf-8', xml_declaration=True)
    # World is a URDF floating-base placeholder, not an inertial body. The
    # gripper's virtual stroke frame is an empty fixed leaf, not a 1 kg body.
    removed = []
    for name, joint_name in [('world', 'floating_base_joint'), ('piper_gripper_link', 'piper_gripper')]:
        link = root.find(f"link[@name='{name}']")
        joint = root.find(f"joint[@name='{joint_name}']")
        if link is None or joint is None or link.find('inertial') is not None or link.find('collision') is not None:
            raise ValueError(f'Unexpected massless placeholder: {name}')
        if name != 'world' and any(j.find('parent').get('link') == name for j in root.findall('joint')):
            raise ValueError(f'Massless placeholder is no longer a leaf: {name}')
        removed.append({'name': name, 'joint': ET.tostring(joint, encoding='unicode')})
        root.remove(link)
        root.remove(joint)
    for extension in root.findall('mujoco'):
        root.remove(extension)
    for link in root.findall('link'):
        if float(link.find('inertial/mass').get('value')) <= 0:
            raise ValueError(f'Nonpositive physical mass: {link.get("name")}')
        # Source collision shapes are the declared rendering representation.
        for collision in link.findall('collision'):
            visual = copy.deepcopy(collision)
            visual.tag = 'visual'
            link.append(visual)
    tree = ET.ElementTree(root)
    tree.write(output / 'merged.urdf', encoding='utf-8', xml_declaration=True)
    manifest = {'root_body': 'base_link', 'body_names': [l.get('name') for l in root.findall('link')],
                'controlled_joint_order': config['controlled_joint_order'], 'tcp': config['tcp'],
                'removed_massless_frames': removed,
                'fixed_joint_names': [j.get('name') for j in root.findall('joint') if j.get('type') == 'fixed'],
                'mass_kg': sum(float(l.find('inertial/mass').get('value')) for l in root.findall('link')),
                'collision_policy': 'Source primitives unchanged; source meshes use convex hull; self collision enabled; official importer/PhysX contact defaults',
                'inertial_policy': 'All positive-mass source links and fixed joints retained; source COM and rotated inertia tensors explicitly authored',
                'merged_urdf': str(output / 'merged.urdf')}
    (output / 'preparation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return tree, manifest
