"""Export a RoboDuet checkpoint as all five original-named CPU JIT modules."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New export directory')
    parser.add_argument('--joint-names', type=Path, default=ROOT / 'configs/roboduet_go1_arx5_joint_names.json')
    parser.add_argument('--arm-output-mode', choices=('official_play', 'training_mean'), default='official_play',
                        help='Original playback leaves body guidance raw; training_mean applies the training mean tanh')
    args = parser.parse_args()
    from pawcerto.methods.roboduet.export import export_checkpoint
    result = export_checkpoint(args.checkpoint, args.output,
                               json.loads(args.joint_names.read_text()), mode=args.arm_output_mode)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
