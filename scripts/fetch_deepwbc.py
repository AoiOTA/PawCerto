#!/usr/bin/env python3
"""Fetch the fixed DeepWBC source and official Go1/WidowX assets.

No installation is performed. Existing files must match the pinned Git blobs;
partial downloads are never promoted until their content hash is verified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

SHA = "8159e4ed8695b2d3f62a40d2ab8d88205ac5021a"
REPO = "MarkFzp/Deep-Whole-Body-Control"


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
    return (path in ('LICENSE', 'README.md')
            or path.startswith(('rsl_rl/', 'legged_gym/legged_gym/',
                                'legged_gym/resources/robots/widowGo1/'))
            or path in ('legged_gym/LICENSE', 'legged_gym/README.md', 'legged_gym/setup.py'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'third_party/deepwbc-reference')
    args = parser.parse_args()
    destination = args.destination.resolve()
    tree_path = destination / '.source-tree.json'
    if not tree_path.exists():
        partial = download(f'https://api.github.com/repos/{REPO}/git/trees/{SHA}?recursive=1', tree_path)
        tree = json.loads(partial.read_text())
        if tree.get('sha') != SHA or tree.get('truncated'):
            raise ValueError('Expected complete pinned DeepWBC source tree')
        partial.replace(tree_path)
    tree = json.loads(tree_path.read_text())
    if tree.get('sha') != SHA or tree.get('truncated'):
        raise ValueError('Expected complete pinned DeepWBC source tree')
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
    print(json.dumps({'revision': SHA, 'destination': str(destination),
                      'verified_files': len(files), 'downloaded_files': added}, indent=2))


if __name__ == '__main__':
    main()
