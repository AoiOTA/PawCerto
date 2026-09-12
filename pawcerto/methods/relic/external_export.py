"""CPU forward of a caller-supplied ReLIC actor export, not a controller."""

from pathlib import Path

import numpy as np


def validate_observations(observations: np.ndarray) -> np.ndarray:
    """Require the exported single-sample ABI, without guessing robot ordering."""
    if not isinstance(observations, np.ndarray) or observations.dtype != np.float32:
        raise TypeError("observations must be a float32 NumPy array")
    if observations.shape != (1, 84):
        raise ValueError("observations must have shape (1, 84)")
    if not np.isfinite(observations).all():
        raise ValueError("observations must be finite")
    return np.ascontiguousarray(observations)


def run_external_policy(policy: str | Path, observations: np.ndarray) -> np.ndarray:
    """Load an explicit trusted .pt or .onnx path and return raw 12-leg actions.

    TorchScript is executable content; only load exports you trust. No robot
    targets, command interpolation, or simulation are implemented here.
    """
    observations = validate_observations(observations)
    policy = Path(policy)
    if policy.suffix == ".pt":
        import torch

        model = torch.jit.load(str(policy), map_location="cpu").eval()
        with torch.inference_mode():
            actions = model(torch.from_numpy(observations)).detach().cpu().numpy()
    elif policy.suffix == ".onnx":
        import onnxruntime as ort

        session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
        actions = session.run(["actions"], {"obs": observations})[0]
    else:
        raise ValueError("policy must be an explicit .pt or .onnx export")
    if actions.shape != (1, 12) or actions.dtype != np.float32:
        raise ValueError("external policy must output float32 actions with shape (1, 12)")
    if not np.isfinite(actions).all():
        raise ValueError("external policy returned nonfinite actions")
    return actions
