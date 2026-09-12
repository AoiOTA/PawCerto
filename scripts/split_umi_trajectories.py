"""Generate source-hashed grouped UMI trajectory IDs, without copying source data."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pawcerto.methods.umi_on_legs.data_split import make_manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', type=Path, default=Path('reference/data/tossing.pkl'))
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = make_manifest(args.trajectory, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(manifest, stream, indent=2); stream.write('\n')
    print(json.dumps({name: len(ids) for name, ids in manifest['partitions'].items()}))
