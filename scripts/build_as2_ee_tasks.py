"""Build eight FK-constructed AS2/Piper-H EE tasks; CPU only, no training.

Requires a prepared assembly and existing asset-family training configuration.
The output recipe retains the caller's runner settings without authorizing a run.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pawcerto.methods.umi_on_legs.as2_tasks import build_as2_ee_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/as2_general_ee_tasks.json")
    parser.add_argument("--assembly-config", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True,
                        help="Existing family recipe; its manifest path resolves relative to this file")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_as2_ee_tasks(args.task_config, args.assembly_config, args.training_config, args.output)
    print(json.dumps({"trajectories": len(result["trajectories"]),
                      "frames": result["sampler_contract"]["frames"],
                      "output": str(args.output.resolve()),
                      "validation": result["validation"]}, indent=2))


if __name__ == "__main__":
    main()
