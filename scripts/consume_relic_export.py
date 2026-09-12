"""Consume an explicitly supplied external ReLIC export on CPU."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pawcerto.methods.relic.external_export import run_external_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True, help="float32 (1, 84) .npy")
    parser.add_argument("--output", type=Path, required=True, help="JSON output path")
    parser.add_argument("--compare-policy", type=Path, help="optional second export; errors propagate")
    args = parser.parse_args()
    observations = np.load(args.observations, allow_pickle=False)
    actions = run_external_policy(args.policy, observations)
    result = {
        "evidence": "CPU exported-actor forward only; no robot or controller validation",
        "policy": str(args.policy),
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "observations": str(args.observations),
        "observations_sha256": hashlib.sha256(args.observations.read_bytes()).hexdigest(),
        "actions": actions.tolist(),
    }
    if args.compare_policy is not None:
        other = run_external_policy(args.compare_policy, observations)
        result["comparison"] = {
            "policy": str(args.compare_policy),
            "policy_sha256": hashlib.sha256(args.compare_policy.read_bytes()).hexdigest(),
            "actions": other.tolist(),
            "max_abs_difference": float(np.max(np.abs(actions - other))),
            "allclose_rtol_1e-5_atol_1e-6": bool(np.allclose(actions, other, rtol=1e-5, atol=1e-6)),
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.compare_policy is not None and not result["comparison"]["allclose_rtol_1e-5_atol_1e-6"]:
        raise SystemExit("external export parity failed; results retained in output")


if __name__ == "__main__":
    main()
