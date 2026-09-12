"""CPU numerical comparison against the pinned original DeepWBC source."""
from copy import deepcopy
import importlib
import hashlib
import json
from pathlib import Path
import torch
import pytest
from pawcerto.methods.deepwbc.config import default_config
from pawcerto.methods.deepwbc.policy import build_model
from pawcerto.methods.deepwbc.training import PPO, OnPolicyRunner

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'third_party/deepwbc-reference'


def test_pinned_source_blobs():
    manifest = json.loads((REFERENCE / '.source-tree.json').read_text())
    assert manifest['sha'] == '8159e4ed8695b2d3f62a40d2ab8d88205ac5021a'
    entries = {entry['path']: entry['sha'] for entry in manifest['tree']}
    for filename in ('algorithms/ppo.py', 'storage/rollout_storage.py',
                     'utils/utils.py', 'runners/on_policy_runner.py'):
        path = 'rsl_rl/rsl_rl/' + filename
        data = (REFERENCE / path).read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == entries[path]

@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def algorithm_config():
    cfg = default_config()
    cfg['algorithm']['num_learning_epochs'] = 2
    cfg['algorithm']['num_mini_batches'] = 2
    cfg['runner']['num_steps_per_env'] = 4
    return cfg


def populate(alg, seed=12):
    torch.manual_seed(seed)
    with torch.inference_mode():
        for step in range(4):
            obs = torch.randn(4, 860)
            alg.act(obs, obs, hist_encoding=False)
            alg.process_env_step(torch.randn(4), torch.randn(4),
                                 torch.tensor([step == 2, False, False, step == 3]),
                                 {'time_outs': torch.tensor([0., 0., 0., float(step == 3)])})
        alg.compute_returns(torch.randn(4, 860))


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('stage', ['ppo', 'dagger'])
def test_original_update_optimizer_and_returns_parity(stage, monkeypatch):
    assert REFERENCE.exists(), 'Pinned source required for numerical parity'
    monkeypatch.syspath_prepend(str(REFERENCE / 'rsl_rl'))
    upstream = importlib.import_module('rsl_rl.algorithms.ppo').PPO
    cfg = algorithm_config()
    torch.manual_seed(7)
    model = build_model(cfg)
    original = upstream(deepcopy(model), device='cpu', **cfg['algorithm'])
    adapted = PPO(deepcopy(model), device='cpu', **cfg['algorithm'])
    for alg in (original, adapted):
        alg.init_storage(4, 4, [860], [None], [18])
        alg.counter = 4500  # exercise nonzero mixing and privileged regularization
        populate(alg)
    for key in ('rewards', 'returns', 'advantages', 'actions_log_prob'):
        assert_tree_equal(getattr(original.storage, key), getattr(adapted.storage, key))
    before = deepcopy(adapted.actor_critic.state_dict())
    torch.manual_seed(99)
    expected = original.update() if stage == 'ppo' else original.update_dagger()
    torch.manual_seed(99)
    actual = adapted.update() if stage == 'ppo' else adapted.update_dagger()
    assert actual == expected
    assert_tree_equal(original.actor_critic.state_dict(), adapted.actor_critic.state_dict())
    assert_tree_equal(original.optimizer.state_dict(), adapted.optimizer.state_dict())
    assert_tree_equal(original.hist_encoder_optimizer.state_dict(), adapted.hist_encoder_optimizer.state_dict())
    assert original.counter == adapted.counter == 4501
    assert any(not torch.equal(before[k], v) for k, v in adapted.actor_critic.state_dict().items())
    if stage == 'dagger':
        assert all(torch.equal(before[k], v) for k, v in adapted.actor_critic.state_dict().items()
                   if not k.startswith('actor.history_encoder.'))


class DeterministicEnv:
    """Interface fixture only; it provides no physics or learning evidence."""
    num_envs = 4
    num_obs = 860
    num_privileged_obs = None
    num_actions = 18
    def reset(self):
        self.obs = torch.linspace(-1, 1, 3440).reshape(4, 860)
        self.curriculum_calls = 0
        return self.obs, None
    def get_observations(self):
        return self.obs
    def get_privileged_observations(self):
        return None
    def update_command_curriculum(self):
        self.curriculum_calls += 1
    def step(self, actions):
        return self.obs, None, actions[:, :12].mean(-1), actions[:, 12:].mean(-1), torch.zeros(4), {}


def test_runner_stage_checkpoint_resume_and_export(tmp_path):
    cfg = algorithm_config()
    torch.manual_seed(11)
    runner = OnPolicyRunner(DeterministicEnv(), cfg)
    # Both phase boundaries and PPO are exercised through actual update calls.
    result = runner.learn(21)
    assert [r['iteration'] for r in result if r['stage'] == 'dagger'] == [0, 20]
    assert runner.alg.counter == runner.current_learning_iteration == 21
    checkpoint = tmp_path / 'checkpoint.pt'
    runner.save(checkpoint, {'test': True})
    runner.learn(1)
    continued = deepcopy(runner.alg.actor_critic.state_dict())
    optimizer = deepcopy(runner.alg.optimizer.state_dict())
    resumed = OnPolicyRunner(DeterministicEnv(), cfg)
    assert resumed.load(checkpoint) == {'test': True}
    assert resumed.current_learning_iteration == resumed.alg.counter == 21
    assert_tree_equal(runner.alg.hist_encoder_optimizer.state_dict(), resumed.alg.hist_encoder_optimizer.state_dict())
    assert resumed.learn(1)[0]['stage'] == 'ppo'
    assert_tree_equal(continued, resumed.alg.actor_critic.state_dict())
    assert_tree_equal(optimizer, resumed.alg.optimizer.state_dict())
    exported = resumed.export(tmp_path / 'history.pt')
    obs = torch.randn(3, 860)
    with torch.no_grad():
        expected = resumed.alg.actor_critic.act_inference(obs, hist_encoding=True)
        torch.testing.assert_close(exported(obs), expected)
        torch.testing.assert_close(torch.jit.load(str(tmp_path / 'history.pt'))(obs), expected)
