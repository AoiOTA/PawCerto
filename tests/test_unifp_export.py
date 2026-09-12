"""Check exported deployment functions against the full original network path."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from pawcerto.methods.unifp.algorithm import ActorCritic
from pawcerto.methods.unifp.export import ExportedUniFP, export_policy, load_checkpoint
from pawcerto.methods.unifp.training.config import default_config


class UniFPExportTests(unittest.TestCase):
    def test_full_network_checkpoint_and_standalone_dynamic_batch(self):
        torch.set_num_threads(1)
        torch.manual_seed(37)
        config = default_config()
        with contextlib.redirect_stdout(io.StringIO()):
            model = ActorCritic(2336, 447, 12, 73, 17, **config['train']['policy']).eval()
            with tempfile.TemporaryDirectory() as temp:
                temp = Path(temp)
                checkpoint = temp / 'model.pt'
                torch.save({'model_state_dict': model.state_dict(), 'config': config,
                            'iter': 21, 'global_steps': 504}, checkpoint)
                loaded = load_checkpoint(checkpoint)
                result = export_policy(loaded, temp / 'export')
                self.assertTrue(result['actions']['passed'])
                self.assertTrue(result['estimates']['passed'])
                exported = ExportedUniFP(temp / 'export')
                for batch in (1, 3, 11):
                    observations = torch.randn(batch, 2336)
                    with torch.inference_mode():
                        expected = model.act_inference({'obs': observations}, policy_info={})
                        estimates = model.get_student_latent(observations)
                    torch.testing.assert_close(exported.act_inference({'obs': observations}), expected)
                    torch.testing.assert_close(exported.get_student_latent(observations), estimates)
                for shape in ((1, 73), (0, 2336), (2336,)):
                    with self.assertRaises((RuntimeError, torch.jit.Error)):
                        exported.act_inference({'obs': torch.zeros(shape)})
                contract = json.loads((temp / 'export/contract.json').read_text())
                self.assertEqual(contract['global_steps'], 504)
                self.assertEqual(contract['checkpoint'], loaded.identity)
                # Legacy policy-only inputs cannot silently acquire a default config.
                torch.save({'model_state_dict': model.state_dict()}, checkpoint)
                with self.assertRaisesRegex(ValueError, 'lacks config'):
                    load_checkpoint(checkpoint)
                load_checkpoint(checkpoint, config=config)


if __name__ == '__main__':
    unittest.main()
