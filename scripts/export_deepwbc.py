"""Export a trusted DeepWBC checkpoint and verify standalone CPU action parity."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Trusted full runner checkpoint')
    parser.add_argument('--config', type=Path, help='Complete JSON config, required for upstream checkpoints without config')
    parser.add_argument('--output', type=Path, required=True, help='New export directory (must not exist)')
    parser.add_argument('--observations', type=Path, help='Optional .npy [batch,860] or evaluation .npz containing observations')
    args = parser.parse_args(argv)
    import numpy as np
    import torch
    from pawcerto.methods.deepwbc.export import load_checkpoint, export_policy
    torch.set_num_threads(1)
    config = json.loads(args.config.read_text()) if args.config else None
    loaded = load_checkpoint(args.checkpoint, config=config)
    observations = None
    if args.observations:
        array = np.load(args.observations, allow_pickle=False)
        if isinstance(array, np.lib.npyio.NpzFile):
            with array:
                array = array['observations'].copy()
        if array.ndim == 3:
            array = array.reshape(-1, array.shape[-1])
        observations = torch.from_numpy(array)
    report = export_policy(loaded, args.output, observations)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
