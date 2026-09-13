"""Prepare the finite AS2 training family while retaining nominal AS2 evaluation."""
from copy import deepcopy
import json
from pathlib import Path

from pawcerto.artifacts import file_identity
from .robot_binding import as2_config, robot_binding


def configure_asset_family(source, manifest_path):
    """Record converted assets consumed by the existing UMI training entrypoint."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    variants = []
    for entry in manifest['variants']:
        urdf = Path(entry['merged_urdf_path'])
        # The official converter writes this pointer next to its prepared URDF.
        pointer = urdf.with_name('usd_path.txt')
        usd = Path(pointer.read_text().strip())
        variants.append(dict(name=entry['name'], urdf=file_identity(urdf), usd=file_identity(usd)))
    names = [entry['name'] for entry in variants]
    if len(names) != len(set(names)) or 'nominal' not in names:
        raise ValueError('AS2 asset family requires unique names and a nominal evaluation asset')
    nominal = variants[names.index('nominal')]
    nominal_path = manifest.get('source', {}).get('nominal_config_path')
    if nominal_path is not None:
        if file_identity(Path(nominal_path))['sha256'] != manifest['source']['nominal_config_sha256']:
            raise ValueError('Changed nominal assembly configuration in AS2 family')
        config = as2_config(source, nominal_path=nominal_path, urdf_path=nominal['urdf']['path'])
    else:
        # Historical manifests precede assembly selection and use the original binding.
        config = deepcopy(source) if robot_binding(source)['name'] == 'as2_piper' else as2_config(source)
    config['pawcerto_asset'] = dict(
        usd_path=nominal['usd']['path'], usd_sha256=nominal['usd']['sha256'],
        urdf_path=nominal['urdf']['path'], urdf_sha256=nominal['urdf']['sha256'])
    config['pawcerto_asset_family'] = dict(
        manifest=file_identity(manifest_path), variants=variants,
        assignment='sequential', evaluation_variant='nominal',
        inertial_randomization='asset family replaces independent mass and COM offsets',
        scope='Finite research hypotheses; not calibrated hardware ranges or unseen-AS2 generalization')
    return config


def training_family(config, training):
    """Family geometry is a training distribution; frozen evaluation uses nominal AS2."""
    family = config.get('pawcerto_asset_family') if training else None
    if family is None:
        return None
    if robot_binding(config)['name'] != 'as2_piper' or family['assignment'] != 'sequential':
        raise ValueError('The finite training family supports AS2 with sequential assignment')
    for variant in family['variants']:
        for key in ('urdf', 'usd'):
            identity = variant[key]
            if file_identity(Path(identity['path']))['sha256'] != identity['sha256']:
                raise ValueError(f"Changed {key} in AS2 training family: {variant['name']}")
    return family


def require_resume_family(source, destination):
    if source.get('pawcerto_asset_family') != destination.get('pawcerto_asset_family'):
        raise ValueError('Resume must retain the AS2 asset family; use explicit weight initialization for adaptation')


def resolve_training_asset(config, source_asset, usd_override=None, urdf_override=None):
    """Weight initialization retains the destination family's nominal evaluation asset."""
    family = config.get('pawcerto_asset_family')
    recorded = config.get('pawcerto_asset') if family else source_asset
    binding = robot_binding(config)
    usd = Path(usd_override or (recorded or {}).get('usd_path') or binding['usd_path'])
    if usd.suffix == '.txt':
        usd = Path(usd.read_text().strip())
    usd_identity = file_identity(usd)
    urdf_identity = file_identity(urdf_override or (recorded or {}).get('urdf_path') or binding['urdf_path'])
    if family:
        nominal = next(v for v in family['variants'] if v['name'] == family['evaluation_variant'])
        if usd_identity != nominal['usd'] or urdf_identity != nominal['urdf']:
            raise ValueError('Training family must retain its declared nominal evaluation asset')
    return dict(usd_path=usd_identity['path'], usd_sha256=usd_identity['sha256'],
                urdf_path=urdf_identity['path'], urdf_sha256=urdf_identity['sha256'])
