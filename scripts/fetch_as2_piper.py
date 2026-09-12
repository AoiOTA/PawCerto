#!/usr/bin/env python3
"""Fetch pinned vendor AS2/Piper-H assets; verify Git blobs before local reuse."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ('unitree_as2', 'unitreerobotics/unitree_ros', '7d6075f7f58588b189b940130e3edab3c839b2df', 'robots/as2_description/'),
    ('agilex_piper_h', 'agilexrobotics/agx_arm_urdf', 'f6642ce0d7872c686f29c99e9e10cd23d1d49313', 'piper_h/'),
    ('unitree_as2_dynamics', 'unitreerobotics/unitree_mujoco', '1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d', 'unitree_robots/as2/as2.xml'),
]


def blob_sha(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    with urlopen(Request(url, headers={'User-Agent': 'PawCerto-asset-fetch'}), timeout=60) as response:
        with partial.open('wb') as stream:
            shutil.copyfileobj(response, stream)
    return partial


def fetch(destination, cache=None):
    destination = Path(destination).resolve()
    records = []
    for source_id, repo, revision, prefix in SOURCES:
        tree_path = destination / source_id / '.source-tree.json'
        if not tree_path.exists():
            partial = download(f'https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1', tree_path)
            tree = json.loads(partial.read_text())
            if tree.get('sha') != revision or tree.get('truncated'):
                raise ValueError(f'Incomplete or wrong source tree: {repo}')
            partial.replace(tree_path)
        tree = json.loads(tree_path.read_text())
        if tree.get('sha') != revision or tree.get('truncated'):
            raise ValueError(f'Invalid cached source tree: {tree_path}')
        entries = [e for e in tree['tree'] if e['type'] == 'blob' and
                   (e['path'] in ('LICENSE', 'LICENSE.md', 'LICENSE.txt', 'COPYING',
                                  'piper_h/urdf/piper_h_with_gripper_description.xacro') or
                    (e['path'].startswith(prefix) and Path(e['path']).suffix.lower() in ('.urdf', '.stl', '.xml')))]
        if not entries:
            raise ValueError(f'No selected source files: {repo}')
        for entry in entries:
            relative = Path(source_id) / entry['path']
            target = destination / relative
            if destination not in target.resolve().parents:
                raise ValueError(f'Path escapes destination: {relative}')
            url = f'https://raw.githubusercontent.com/{repo}/{revision}/{entry["path"]}'
            if not target.exists():
                cached = Path(cache) / relative if cache else None
                if cached is not None and cached.is_file():
                    if blob_sha(cached) != entry['sha']:
                        raise ValueError(f'Local input differs from official Git blob: {cached}')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = target.with_name(target.name + '.partial')
                    shutil.copyfile(cached, partial)
                else:
                    partial = download(url, target)
                if blob_sha(partial) != entry['sha']:
                    raise ValueError(f'Download differs from official Git blob: {partial}')
                partial.replace(target)
            if blob_sha(target) != entry['sha']:
                raise ValueError(f'Existing file differs from official Git blob: {target}')
            records.append({'path': str(relative), 'repository': repo, 'revision': revision,
                            'git_blob': entry['sha'], 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                            'bytes': target.stat().st_size, 'url': url})
    manifest = {'files': records}
    (destination / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return {'destination': str(destination), 'verified_files': len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT / 'third_party/as2_piper')
    parser.add_argument('--cache', type=Path, help='Optional existing upstream asset directory; each file is checked against official Git blobs')
    args = parser.parse_args()
    print(json.dumps(fetch(args.destination, args.cache), indent=2))


if __name__ == '__main__':
    main()
