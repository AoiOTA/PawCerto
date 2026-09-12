"""Authored ABI tests; these do not establish original ReLIC behavior."""

import numpy as np
import pytest
import torch

from pawcerto.methods.relic.external_export import run_external_policy, validate_observations


@pytest.mark.parametrize("shape", [(84,), (2, 84), (1, 83)])
def test_reject_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        validate_observations(np.zeros(shape, dtype=np.float32))


def test_reject_dtype_and_nonfinite():
    with pytest.raises(TypeError, match="float32"):
        validate_observations(np.zeros((1, 84)))
    obs = np.zeros((1, 84), dtype=np.float32)
    obs[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_observations(obs)


def test_explicit_cpu_torchscript(tmp_path):
    model = torch.nn.Linear(84, 12)
    path = tmp_path / "authored.pt"
    torch.jit.script(model).save(str(path))
    obs = np.linspace(-0.5, 0.5, 84, dtype=np.float32).reshape(1, 84)
    np.testing.assert_allclose(run_external_policy(path, obs), model(torch.from_numpy(obs)).detach().numpy())


def test_reject_wrong_output(tmp_path):
    path = tmp_path / "wrong.pt"
    torch.jit.script(torch.nn.Identity()).save(str(path))
    with pytest.raises(ValueError, match="output"):
        run_external_policy(path, np.zeros((1, 84), dtype=np.float32))
