"""CPU original-source parity and checkpoint contract, not learning evidence."""
from pathlib import Path
import torch
import pytest
from pawcerto.methods.learning_force_control import ActorCritic, PPO, PPO_Args, RolloutStorage
from pawcerto.methods.learning_force_control.config import training_config, build_model
from pawcerto.methods.learning_force_control.export import load_checkpoint, export_policy, JitPolicy
from pawcerto.methods.learning_force_control.history import ObservationHistory
from pawcerto.methods.learning_force_control.observations import actor_observation, force_in_base_yaw

SOURCE = Path(__file__).resolve().parents[1] / 'third_party/learning-force-control-reference'


@pytest.fixture(autouse=True)
def restore_original_torch_typo():
    # The original constructor overwrites an inherited torch API; keep this
    # source-parity probe from contaminating unrelated tests in the process.
    normal = torch.distributions.Normal
    previous = normal.__dict__.get('set_default_validate_args')
    yield
    if previous is None:
        if 'set_default_validate_args' in normal.__dict__:
            delattr(normal, 'set_default_validate_args')
    else:
        normal.set_default_validate_args = previous


def original_classes():
    if not SOURCE.exists():
        pytest.skip('Run scripts/fetch_learning_force_control.py for source parity')
    # Execute actual fixed original algorithm source. Remove only package/config
    # imports so Isaac Gym, logging and CLI metaclasses are not CPU prerequisites.
    ac = (SOURCE/'b1_gym_learn/ppo_cse/actor_critic.py').read_text()
    ac = ac.replace('from params_proto import PrefixProto', '').replace('class AC_Args(PrefixProto, cli=False):','class AC_Args:')
    ns = {}; exec(compile(ac, 'original/actor_critic.py', 'exec'), ns)
    storage = (SOURCE/'b1_gym_learn/ppo_cse/rollout_storage.py').read_text().replace('from b1_gym_learn.utils import split_and_pad_trajectories','')
    exec(compile(storage, 'original/rollout_storage.py','exec'), ns)
    ppo = (SOURCE/'b1_gym_learn/ppo_cse/ppo.py').read_text()
    for line in ['from params_proto import PrefixProto', 'from b1_gym_learn.ppo_cse import ActorCritic', 'from b1_gym_learn.ppo_cse import RolloutStorage', 'from b1_gym_learn.ppo_cse import caches']:
        ppo = ppo.replace(line, '')
    ppo = ppo.replace('class PPO_Args(PrefixProto):', 'class PPO_Args:')
    exec(compile(ppo, 'original/ppo.py','exec'), ns)
    return ns


def test_original_network_checkpoint_and_export(tmp_path):
    ns = original_classes(); cfg = training_config()
    for k,v in cfg['policy'].items(): setattr(ns['AC_Args'], k, v)
    torch.manual_seed(21)
    original = ns['ActorCritic'](87,16,870,19)
    checkpoint = tmp_path/'ac_weights_latest.pt'
    torch.save(original.state_dict(), checkpoint)
    loaded = load_checkpoint(checkpoint, config=cfg)
    history = torch.randn(3,870); privileged = torch.randn(3,16)
    torch.testing.assert_close(original.act_student(history),loaded.act_student(history),rtol=0,atol=0)
    torch.testing.assert_close(original.act_teacher(history,privileged),loaded.act_teacher(history,privileged),rtol=0,atol=0)
    torch.testing.assert_close(original.evaluate(history,privileged),loaded.evaluate(history,privileged),rtol=0,atol=0)
    export_policy(loaded,tmp_path/'jit')
    torch.testing.assert_close(JitPolicy(tmp_path/'jit')(history),loaded.act_student(history),rtol=0,atol=0)
    cfg['num_actions']=17
    with pytest.raises(RuntimeError):load_checkpoint(checkpoint,config=cfg)


def test_original_ppo_storage_update_parity():
    ns=original_classes(); cfg=training_config()
    for k,v in cfg['policy'].items():setattr(ns['AC_Args'],k,v)
    torch.manual_seed(32); original=ns['ActorCritic'](87,16,870,19)
    port=build_model(cfg);port.load_state_dict(original.state_dict())
    opts=PPO_Args();opts.num_learning_epochs=1;opts.num_mini_batches=1
    ns['PPO_Args'].num_learning_epochs=1;ns['PPO_Args'].num_mini_batches=1
    a=ns['PPO'](original);b=PPO(port,config=opts)
    for alg in (a,b):alg.init_storage(4,2,[87],[16],[870],[19])
    for t in range(2):
        obs=torch.randn(4,87);priv=torch.randn(4,16);hist=torch.randn(4,870)
        reward=torch.randn(4);done=torch.tensor([0,0,0,t],dtype=torch.uint8)
        for alg in (a,b):
            torch.manual_seed(90+t);alg.act(obs,priv,hist)
            alg.process_env_step(reward,done,{'env_bins':torch.zeros(4),'time_outs':torch.tensor([0,0,0,t])})
    last=torch.randn(4,870);priv=torch.randn(4,16)
    for alg in (a,b):alg.compute_returns(last,priv)
    torch.testing.assert_close(a.storage.returns,b.storage.returns,rtol=0,atol=0)
    torch.manual_seed(7);ra=a.update()
    torch.manual_seed(7);rb=b.update()
    assert ra==rb
    for k,v in original.state_dict().items():torch.testing.assert_close(v,port.state_dict()[k],rtol=0,atol=0)
    assert b.storage.step==0
    b.test_mode();assert not port.training


def test_history_and_force_input_semantics():
    h=ObservationHistory(2,num_obs=1,history_length=3,frame_skip=2)
    for i in range(1,7):result=h.append(torch.full((2,1),float(i)))
    torch.testing.assert_close(result,torch.tensor([[2.,4.,6.],[2.,4.,6.]]))
    h.reset_rows([1]);assert h.buffer[1].count_nonzero()==0
    command=torch.ones(2,23);zero=torch.zeros(2,19)
    result=actor_observation(torch.zeros(2,3),command,torch.ones(23),zero,zero,zero,zero,torch.zeros(2,4),torch.tensor([0,1]))
    assert result.shape==(2,87)
    assert result[1,18:21].count_nonzero()==0
    assert torch.equal(command,torch.ones(2,23))
    torch.testing.assert_close(force_in_base_yaw(torch.tensor([[1.,0.,2.]]),torch.tensor([torch.pi/2])),torch.tensor([[0.,-1.,2.]]),atol=1e-6,rtol=0)
