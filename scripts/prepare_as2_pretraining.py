"""Write a UMI configuration consuming a converted finite AS2 asset family."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'reference/checkpoints/tossing/ours/config.json')
    parser.add_argument('--family-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from pawcerto.methods.umi_on_legs.pretraining import configure_asset_family
    config = configure_asset_family(json.loads(args.config.read_text()), args.family_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + '\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
