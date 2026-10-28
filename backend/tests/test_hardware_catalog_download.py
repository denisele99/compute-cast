import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'download_hardware_catalog.py'
spec = importlib.util.spec_from_file_location('hardware_downloader', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    return {'_header': {'gpu_memory': {'unit': 'GB'}, 'memory_bandwidth': {'unit': 'GB/s'},
                        **{field: {'unit': 'TFLOPS'} for field in module.COMPUTE_FIELDS.values()}},
            'test_gpu': {'name': 'Example GPU', 'manufacturer': 'NVIDIA', 'architecture': 'Example',
                         'gpu_memory': 24, 'memory_bandwidth': 1000, 'fp32': 50,
                         'fp16_tensor_core': None, 'fp16_tensor_core_sparsity': 1000,
                         'sources': ['https://example.com/datasheet']}}


class CatalogDownloadTests(unittest.TestCase):
    def test_maps_schema_without_using_sparse_peaks_or_guessing_variant(self):
        result = module.convert(fixture(), '2026-10-28')['gpus']['nvidia_test_gpu']
        self.assertEqual(result['memory']['advertised_capacity_gb'], 24)
        self.assertEqual(result['memory']['bandwidth_gb_per_second'], 1000)
        self.assertEqual(result['peak_compute_tflops']['fp32'], 50)
        self.assertIsNone(result['peak_compute_tflops']['fp16_tensor_dense'])
        self.assertIsNone(result['form_factor'])
        self.assertIsNone(result['source']['last_verified'])
        self.assertNotIn('price', json.dumps(result))

    def test_ambiguous_capacity_is_not_silently_chosen(self):
        data = fixture()
        data['test_gpu']['gpu_memory'] = '16/32'
        memory = module.convert(data, '2026-10-28')['gpus']['nvidia_test_gpu']['memory']
        self.assertIsNone(memory['advertised_capacity_gb'])
        self.assertEqual(memory['advertised_capacity_options_gb'], [16, 32])

    def test_schema_drift_bad_units_and_invalid_numbers_fail(self):
        for value in (-1, float('nan'), True, '1000 GB/s'):
            data = fixture()
            data['test_gpu']['memory_bandwidth'] = value
            with self.assertRaises(ValueError):
                module.convert(data, '2026-10-28')
        data = fixture()
        data['_header']['memory_bandwidth']['unit'] = 'TB/s'
        with self.assertRaises(ValueError):
            module.convert(data, '2026-10-28')

    def test_curated_entries_win_without_mutation(self):
        imported = module.convert(fixture(), '2026-10-28')
        curated = {'schema_version': 1, 'gpus': {'nvidia_test_gpu': {'name': 'Verified', 'source': {'last_verified': '2026-10-27'}}}}
        original = copy.deepcopy(curated)
        result = module.merge_curated(imported, curated)
        self.assertEqual(result['gpus']['nvidia_test_gpu'], curated['gpus']['nvidia_test_gpu'])
        self.assertEqual(curated, original)

    def test_download_verifies_checksum(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def geturl(self): return 'https://example.com/data.json'
            def read(self, count): return b'example'
        with patch.object(module, 'urlopen', return_value=Response()):
            self.assertEqual(module.download('https://example.com/data.json', hashlib.sha256(b'example').hexdigest()), b'example')
            with self.assertRaises(ValueError):
                module.download('https://example.com/data.json', '0' * 64)

    def test_cli_writes_json_and_license_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'hardware_catalog.json'
            with patch.object(module, 'download', side_effect=[json.dumps(fixture()).encode(), b'Example licence']):
                module.main(['--output', str(output), '--no-merge'])
            self.assertEqual(json.loads(output.read_text())['schema_version'], 1)
            self.assertEqual(output.with_name('hardware_catalog.LICENSE.txt').read_text(), 'Example licence')
            before = output.read_bytes()
            with self.assertRaises(SystemExit), patch.object(module, 'download') as network:
                module.main(['--output', str(output), '--no-merge'])
            network.assert_not_called()
            self.assertEqual(output.read_bytes(), before)
