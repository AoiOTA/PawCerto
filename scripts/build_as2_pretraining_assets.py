"""Generate a finite research AS2/Piper URDF family without starting Isaac."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pawcerto.as2_pretraining_assets import FAMILY_CONFIG, build_family, build_mujoco_family
from pawcerto.mujoco.as2_piper_asset import CONFIG

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=FAMILY_CONFIG)
    parser.add_argument('--nominal-config', type=Path, default=CONFIG,
                        help='Assembly configuration used by every family member; default preserves the historical nominal model')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mujoco-only', action='store_true', help='Read existing frozen manifest, emit original AS2 MuJoCo models and separate mujoco-manifest.json; static validation only')
    args = parser.parse_args()
    result = build_mujoco_family(args.output) if args.mujoco_only else build_family(args.output, args.config, nominal_config_path=args.nominal_config)
    for variant in result['variants']:
        print(variant['name'], variant['total_mass_kg'], variant['mujoco_path'] if args.mujoco_only else variant['urdf_path'])
