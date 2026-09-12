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
                        default=ROOT / 'configs/umi_go2_arx5_joint_names.json',
                        help='Measured original Go2 + ARX5 policy joint order')
    parser.add_argument('--output', type=Path, required=True, help='New export directory')
    args = parser.parse_args()
    from pawcerto.methods.umi_on_legs import UmiPolicy
    policy = UmiPolicy(args.checkpoint)
    result = policy.export(args.output, json.loads(args.joint_names.read_text()))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
