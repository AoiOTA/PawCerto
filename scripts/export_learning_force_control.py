"""Export a specified full Learning Force Control Lab checkpoint."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observations', type=Path, help='.npy histories or evaluation .npz with observations')
    args = parser.parse_args(argv)
    import numpy as np
    import torch
    from pawcerto.methods.learning_force_control.export import load_training_checkpoint, export_training_checkpoint
    torch.set_num_threads(1)
    loaded = load_training_checkpoint(args.checkpoint)
    histories = None
    if args.observations:
        array = np.load(args.observations, allow_pickle=False)
        if isinstance(array, np.lib.npyio.NpzFile):
            with array:
                array = array['observations'].copy()
        if array.ndim == 3:
            array = array.reshape(-1, array.shape[-1])
        histories = torch.from_numpy(array)
    print(json.dumps(export_training_checkpoint(loaded, args.output, histories), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
