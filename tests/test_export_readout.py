"""Package parity must reject mismatched or incomplete saved executions."""
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('verify_export', Path(__file__).parents[1] / 'scripts/verify_umi_export.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SavedExportValidation(unittest.TestCase):
    def test_contact_readout_rejects_failed_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'input'
            inputs.mkdir()
            (inputs / 'case_00.json').write_text(json.dumps({'completion_status': 'invalid', 'failure': {'reason': 'numerical'}}))
            result = subprocess.run([sys.executable, str(Path(__file__).parents[1] / 'scripts/readout_umi_contacts.py'),
                                     '--policy', str(inputs), '--reference', str(inputs), '--output', str(root / 'output')],
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires all 16 complete cases', result.stderr)
            self.assertFalse((root / 'output/contact-quality-readout.json').exists())

    def test_failures_cannot_pass_equal_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f'{name}.json' for name in ('original', 'exported')]
            for path in paths:
                np.savez(path.with_suffix('.npz'), metrics=np.array([[.04, 1.]]))
                path.write_text(json.dumps({'checkpoint': str(path), 'completion_status': 'invalid', 'failure': {'reason': 'numerical'}}))
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                module.verify_saved(*paths)

    def test_arrays_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f'{name}.json' for name in ('original', 'exported')]
            for i, path in enumerate(paths):
                np.savez(path.with_suffix('.npz'), metrics=np.array([[.04, float(i)]]))
                path.write_text(json.dumps({'checkpoint': str(path), 'completion_status': 'complete', 'failure': None}))
            with self.assertRaisesRegex(ValueError, 'arrays differ'):
                module.verify_saved(*paths)


if __name__ == '__main__':
    unittest.main()
