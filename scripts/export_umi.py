"""Export one UMI checkpoint as a CPU TorchScript actor with execution config."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Original checkpoint directory or training model_N.pt beside config.json')
    parser.add_argument('--joint-names', type=Path,
                        default=None,
                        help='Optional exact order; defaults to the saved robot binding')
    parser.add_argument('--output', type=Path, required=True, help='New export directory')
    args = parser.parse_args()
    from pawcerto.methods.umi_on_legs import UmiPolicy
    policy = UmiPolicy(args.checkpoint)
    from pawcerto.methods.umi_on_legs.robot_binding import joint_order
    result = policy.export(args.output, joint_order(policy.config,args.joint_names))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
