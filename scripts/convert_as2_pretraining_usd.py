"""Convert a finite AS2 family through the existing official-Lab asset path."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family-manifest', type=Path, required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    manifest = json.loads(args.family_manifest.read_text())
    launcher = AppLauncher(args)
    try:
        from convert_as2_piper_usd import convert_prepared
        for entry in manifest['variants']:
            report = convert_prepared(Path(entry['merged_urdf_path']).parent)
            print(json.dumps(dict(name=entry['name'], usd_path=report['usd_path'])), flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))


if __name__ == '__main__':
    main()
