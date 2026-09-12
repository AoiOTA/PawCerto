#!/usr/bin/env python3
"""Fetch the fixed Learning Force Control source and official B1/Z1 assets.

No installation is performed. Existing files must match the pinned Git blobs;
partial downloads are never promoted until their content hash is verified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from urllib.request import Request, urlopen

SHA = "c760e1d74ad165d3c069d4f57ab5d066f6a41eb6"
REPO = "Improbable-AI/learning-compliance"


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    with urlopen(Request(url, headers={'User-Agent': 'PawCerto-reference-fetch'}), timeout=60) as response:
        with partial.open('wb') as output:
            shutil.copyfileobj(response, output)
    return partial


def blob_sha(path):
    digest = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def selected(path):
    return True  # Preserve complete official source, assets, and file-level licensing.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'third_party/learning-force-control-reference')
    args = parser.parse_args()
    destination = args.destination.resolve()
    tree_path = destination / '.source-tree.json'
    if not tree_path.exists():
        partial = download(f'https://api.github.com/repos/{REPO}/git/trees/{SHA}?recursive=1', tree_path)
        tree = json.loads(partial.read_text())
        if tree.get('sha') != SHA or tree.get('truncated'):
            raise ValueError('Expected complete pinned Learning Force Control source tree')
        partial.replace(tree_path)
    tree = json.loads(tree_path.read_text())
    if tree.get('sha') != SHA or tree.get('truncated'):
        raise ValueError('Expected complete pinned Learning Force Control source tree')
    files = [entry for entry in tree['tree'] if entry['type'] == 'blob' and selected(entry['path'])]

    def fetch(entry):
        target = destination / entry['path']
        if destination not in target.resolve().parents:
            raise ValueError(f'Path escapes source directory: {entry["path"]}')
        if target.exists():
            if blob_sha(target) != entry['sha']:
                raise ValueError(f'Existing file differs from pinned blob: {target}')
            return 0
        partial = download(f'https://raw.githubusercontent.com/{REPO}/{SHA}/{entry["path"]}', target)
        if blob_sha(partial) != entry['sha']:
            raise ValueError(f'Download differs from pinned blob: {partial}')
        partial.replace(target)
        return 1

    failures, added = [], 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch, entry): entry['path'] for entry in files}
        for future in as_completed(futures):
            try:
                added += future.result()
            except Exception as exc:
                failures.append(f'{futures[future]}: {type(exc).__name__}: {exc}')
    if failures:
        raise RuntimeError('Source fetch incomplete; rerun to fill missing verified files:\n' + '\n'.join(failures))
    urdf = destination / 'resources/robots/b1/urdf/b1_plus_z1.urdf'
    # Upstream has unbound xacro:include tags. Bind only for inventory parsing;
    # preserve the pinned asset bytes and report this simulator-use limitation.
    robot = ET.fromstring(urdf.read_text().replace('<robot ', '<robot xmlns:xacro="http://www.ros.org/wiki/xacro" ', 1))
    joints = [joint.attrib['name'] for joint in robot.findall('joint')
              if joint.attrib['type'] != 'fixed']
    if len(joints) != 19:
        raise ValueError(f'Expected official B1/Z1 19 movable joints, found {len(joints)}')
    meshes = {(urdf.parent / mesh.attrib['filename']).resolve()
              for mesh in robot.findall('.//mesh')}
    if any(not mesh.is_file() or destination not in mesh.parents for mesh in meshes):
        raise ValueError('B1/Z1 URDF mesh closure is incomplete or escapes source tree')
    print(json.dumps({'revision': SHA, 'movable_joints': joints, 'verified_meshes': len(meshes), 'asset_note': 'Original URDF contains unbound xacro prefix; namespace supplied only for inventory parsing', 'destination': str(destination),
                      'verified_files': len(files), 'downloaded_files': added}, indent=2))


if __name__ == '__main__':
    main()
