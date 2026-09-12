"""Trajectory identities and grouped partitions; no upstream data is redistributed."""
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_manifest(path, seed=2027):
    with open(path, 'rb') as stream:
        episodes = pickle.load(stream)
    parent = list(range(len(episodes)))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    seen = {}
    rows = []
    recording_keys = ('recording_id', 'source_recording_id', 'source_id')
    for i, ep in enumerate(episodes):
        digest = hashlib.sha256()
        for key in ('ee_pos', 'ee_axis_angle'):
            array = np.ascontiguousarray(ep[key], dtype='<f8')
            digest.update(key.encode()); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
        content = digest.hexdigest()
        recording = {k: str(ep[k]) for k in recording_keys if k in ep}
        for token in [('content', content)] + list(recording.items()):
            if token in seen:
                parent[root(i)] = root(seen[token])
            else:
                seen[token] = i
        rows.append(dict(id=i, content_sha256=content, recording_ids=recording))
    groups = {}
    for row in rows:
        group = root(row['id'])
        row['group'] = group
        groups.setdefault(group, []).append(row['id'])
    order = list(groups)
    random.Random(seed).shuffle(order)
    partitions = dict(train=[], validation=[], test=[])
    targets = dict(train=round(len(rows)*.70), validation=round(len(rows)*.15))
    targets['test'] = len(rows)-sum(targets.values())
    for group in order:
        partition = max(partitions, key=lambda name: targets[name]-len(partitions[name]))
        partitions[partition].extend(groups[group])
    for ids in partitions.values():
        ids.sort()
    return dict(version=1, seed=seed, source_name=Path(path).name, source_sha256=source_hash(path),
                source_count=len(rows), grouping='identical ee_pos + ee_axis_angle, or shared explicit recording ID',
                recording_metadata_available=any(row['recording_ids'] for row in rows),
                trajectories=rows, partitions=partitions)


def resolve_selection(path, manifest_path, partition):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['source_sha256'] != source_hash(path):
        raise ValueError('Trajectory source hash does not match split manifest')
    parts = manifest['partitions']
    ids = [i for values in parts.values() for i in values]
    if set(parts) != {'train', 'validation', 'test'} or sorted(ids) != list(range(manifest['source_count'])):
        raise ValueError('Partitions must cover every source ID exactly once')
    rows = manifest['trajectories']
    if sorted(row['id'] for row in rows) != list(range(manifest['source_count'])):
        raise ValueError('Trajectory metadata must cover every source ID exactly once')
    assignment = {i: name for name, values in parts.items() for i in values}
    groups = {}
    for row in rows:
        group = row['group']
        name = assignment[row['id']]
        if group in groups and groups[group] != name:
            raise ValueError('A trajectory group crosses partition boundaries')
        groups[group] = name
    selected = parts[partition]
    if not selected:
        raise ValueError('Selected partition is empty')
    return dict(manifest_sha256=source_hash(manifest_path), source_sha256=manifest['source_sha256'],
                partition=partition, trajectory_ids=selected, split_seed=manifest['seed'],
                recording_metadata_available=manifest['recording_metadata_available'])


def configure_selection(config, path, manifest_path=None, partition=None):
    sampler = config['env']['tasks']['reaching']['sequence_sampler']
    sampler.pop('trajectory_selection', None)
    selection = resolve_selection(path, manifest_path, partition) if manifest_path else None
    if selection:
        sampler['trajectory_selection'] = selection
    return selection


def checkpoint_training_selection(checkpoint):
    """Read training identity from loaded weights, never adjacent execution config."""
    return checkpoint.get('config', {}).get('env', {}).get('tasks', {}).get('reaching', {}).get('sequence_sampler', {}).get('trajectory_selection')


def evaluation_identity(training_selection, evaluation_selection):
    eligible = bool(training_selection and evaluation_selection
                    and training_selection['partition'] == 'train'
                    and evaluation_selection['partition'] in ('validation', 'test')
                    and training_selection['manifest_sha256'] == evaluation_selection['manifest_sha256'])
    return dict(training_selection=training_selection, evaluation_selection=evaluation_selection,
                held_out_from_recorded_training_partition=eligible,
                limitation='Partition identity is not proof of untouched test selection or unknown recording-source independence.')
