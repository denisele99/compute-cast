"""Optional example checks: python -m unittest discover -s examples/toy_mlp -v."""
import importlib.util
from pathlib import Path
import unittest

import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('toy_training_example', HERE / 'train.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class ExampleTests(unittest.TestCase):
    def test_cpu_step_changes_weights_and_has_finite_loss(self):
        architecture, settings = example.load_settings(HERE)
        sample = example.synthetic_data(architecture, settings)
        step, network = example.make_step(HERE / 'model.py', sample, 'cpu')
        self.assertEqual(sum(p.numel() for p in network.parameters()), 6529)
        before = [p.detach().clone() for p in network.parameters()]
        loss = step()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, network.parameters())))
        self.assertTrue(all(torch.isfinite(p).all() for p in network.parameters()))

    def test_meta_factory_and_sample_shapes_match_configs(self):
        architecture, settings = example.load_settings(HERE)
        spec = importlib.util.spec_from_file_location('toy_model_example', HERE / 'model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with torch.device('meta'):
            network = module.build_model(architecture)
        self.assertEqual(sum(p.numel() for p in network.parameters()), architecture['total_params'])
        data = example.synthetic_data(architecture, settings)
        self.assertEqual(tuple(data['features'].shape), (64, 100))
        self.assertEqual(tuple(data['targets'].shape), (64, 1))
