import json
import importlib.util
import io
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pawcerto.methods.umi_on_legs import PoseSequence, UmiPolicy
from pawcerto.methods.umi_on_legs.data_split import make_manifest, resolve_selection, evaluation_identity, checkpoint_training_selection


class DataSplitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'source.pkl'
        episodes = [dict(ee_pos=np.full((10,3), i, dtype=float), ee_axis_angle=np.zeros((10,3))) for i in range(20)]
        episodes[1] = dict(episodes[0])
        episodes[2]['recording_id'] = episodes[3]['recording_id'] = 'recording-a'
        self.path.write_bytes(pickle.dumps(episodes))
        self.manifest = make_manifest(self.path)
        self.mp = Path(self.tmp.name)/'split.json'
        self.mp.write_text(json.dumps(self.manifest))
        self.config = dict(dt=.02, episode_length_s=.2, planar_center=False, add_random_height_range=None)

    def test_grouping_and_partition_coverage(self):
        self.assertEqual(self.manifest, make_manifest(self.path))
        rows = self.manifest['trajectories']
        self.assertEqual(rows[0]['group'], rows[1]['group'])
        self.assertEqual(rows[2]['group'], rows[3]['group'])
        parts = self.manifest['partitions']
        self.assertEqual(sorted(sum(parts.values(), [])), list(range(20)))
        for name in parts:
            resolve_selection(self.path, self.mp, name)

    def test_source_and_leakage_rejected(self):
        self.path.write_bytes(self.path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError, 'hash'):
            resolve_selection(self.path, self.mp, 'train')

    def test_group_leakage_rejected(self):
        parts = self.manifest['partitions']
        current = next(k for k,v in parts.items() if 0 in v)
        other = next(k for k in parts if k != current)
        parts[current].remove(0); parts[other].append(0)
        self.mp.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'group'):
            resolve_selection(self.path, self.mp, 'train')

    def test_selection_and_original_sampler(self):
        original = PoseSequence(self.path, self.config)
        generator = torch.Generator().manual_seed(7)
        expected_ids = torch.randint(20, (8,), generator=generator)
        self.assertTrue(torch.equal(original.sample(8,7)[0], original.positions[expected_ids]))
        selection = resolve_selection(self.path,self.mp,'train')
        selected = PoseSequence(self.path, dict(self.config, trajectory_selection=selection))
        ids = selection['trajectory_ids']
        pos, rot = selected.select_ids(ids)
        self.assertTrue(torch.equal(pos, original.positions[ids]))
        self.assertEqual(len(rot),len(ids))
        excluded = next(i for i in range(20) if i not in ids)
        with self.assertRaises(ValueError):
            selected.select_ids([excluded])

    def test_legacy_not_relabelled(self):
        test = resolve_selection(self.path,self.mp,'test')
        train = resolve_selection(self.path,self.mp,'train')
        self.assertFalse(evaluation_identity(None,test)['held_out_from_recorded_training_partition'])
        self.assertTrue(evaluation_identity(train,test)['held_out_from_recorded_training_partition'])
        train['manifest_sha256'] = 'another-split'
        self.assertFalse(evaluation_identity(train,test)['held_out_from_recorded_training_partition'])

    def test_training_identity_belongs_to_weights_not_sibling_config(self):
        train = resolve_selection(self.path, self.mp, 'train')
        validation = resolve_selection(self.path, self.mp, 'validation')
        config = {'env': {'tasks': {'reaching': {'sequence_sampler': {'trajectory_selection': train}}}},
                  'runner': {'alg': {'actor_critic': {'actor': {'_args_': [
                      {'_target_': 'torch.nn.Linear', 'in_features': 1, 'out_features': 1}]}}}}}
        directory = Path(self.tmp.name)/'policy'
        directory.mkdir()
        (directory/'config.json').write_text(json.dumps(config))
        actor = torch.nn.Sequential(torch.nn.Linear(1, 1))
        checkpoint = {'model_state_dict': {'actor.'+k: v for k,v in actor.state_dict().items()}}
        weights = directory/'model_4000.pt'
        torch.save(checkpoint, weights)
        policy = UmiPolicy(weights)
        self.assertIsNone(policy.training_selection)
        self.assertFalse(evaluation_identity(policy.training_selection, validation)['held_out_from_recorded_training_partition'])
        # The Lab entry consumes this same helper on its already-loaded checkpoint.
        self.assertIsNone(checkpoint_training_selection(torch.load(weights, weights_only=False)))
        checkpoint['config'] = config
        torch.save(checkpoint, weights)
        self.assertTrue(evaluation_identity(UmiPolicy(weights).training_selection, validation)['held_out_from_recorded_training_partition'])
        config['env']['tasks']['reaching']['sequence_sampler']['trajectory_selection'] = dict(train, manifest_sha256='different-split')
        torch.save(checkpoint, weights)
        self.assertFalse(evaluation_identity(UmiPolicy(weights).training_selection, validation)['held_out_from_recorded_training_partition'])
        # An actor-only bundle does not acquire training provenance from config.json.
        torch.jit.script(actor).save(str(directory/'actor.ts'))
        self.assertIsNone(UmiPolicy(directory).training_selection)


    def test_split_cli_preserves_failures_then_exits_nonzero(self):
        script = Path(__file__).resolve().parents[1]/'scripts/eval_umi_split_mujoco.py'
        spec = importlib.util.spec_from_file_location('split_cli_under_test', script)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        ids = self.manifest['partitions']['validation']
        sampler = Mock()
        sampler.select_ids.return_value = (torch.zeros(len(ids),1,3), torch.zeros(len(ids),1,3,3))
        policy = SimpleNamespace(training_selection=None, config={'env': {'cfg': {'sim': {'dt': .005}},
            'controller': {'decimation_count': 4}, 'tasks': {'reaching': {'sequence_sampler': {}}}}},
            trajectories=Mock(return_value=sampler))
        output = Path(self.tmp.name)/'failed-evaluation'
        args = ['eval_umi_split_mujoco.py', '--checkpoint', 'fixture-policy',
                '--trajectory', str(self.path), '--split-manifest', str(self.mp), '--output', str(output)]
        # Exercise real partition loading, CLI loop, summary aggregation and serialization.
        # Only the physical evaluation is substituted by a numerical-failure report.
        failure = {'kind': 'mujoco_numerical_failure', 'new_warnings': ['mjWARN_BADQACC']}
        with patch.object(cli, 'UmiPolicy', return_value=policy), patch.object(cli, 'evaluate',
                side_effect=lambda *a, **k: dict(completion_status='invalid', failure=failure)) as evaluate, \
                patch('sys.argv', args), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                cli.main()
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(evaluate.call_count, len(ids))
        sampler.select_ids.assert_called_once_with(ids, 2027)
        summary = json.loads((output/'summary.json').read_text())
        self.assertEqual(summary['completion_status'], 'incomplete')
        self.assertEqual(summary['invalid_cases'], len(ids))
        self.assertEqual(summary['evaluated_trajectory_ids'], ids)
        self.assertEqual(summary['seed'], 2027)
        for trajectory_id in ids:
            saved = json.loads((output/f'trajectory_{trajectory_id:03d}.json').read_text())
            self.assertEqual(saved['trajectory_id'], trajectory_id)
            self.assertEqual(saved['failure'], failure)


if __name__ == '__main__':
    unittest.main()
