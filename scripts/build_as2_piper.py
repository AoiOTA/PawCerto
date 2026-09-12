#!/usr/bin/env python3
"""Build and audit a nominal AS2/Piper-H MuJoCo asset without GPU/Isaac/ROS."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pawcerto.mujoco.as2_piper_asset import CONFIG, ROOT, SOURCES, build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, default=SOURCES)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, default=ROOT / 'reference/as2_piper')
    args = parser.parse_args()
    report = build(args.output, args.sources, args.config)
    print(json.dumps({k: v for k, v in report.items() if k not in ('source_provenance', 'source_joint_limits', 'config')}, indent=2))


if __name__ == '__main__':
    main()
