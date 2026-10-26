import unittest
from fastapi.testclient import TestClient
from app.main import create_app


class APITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app())

    def test_existing_frontend_contract(self):
        response = self.client.post("/api/estimate", json={})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["training_steps"], 62500)
        self.assertEqual(result["training_cost_usd"], 3.47)
        self.assertEqual(result["memory_status"], "unknown")

    def test_partial_batches(self):
        result = self.client.post("/api/estimate", json={"examples": 17, "batch_size": 16, "epochs": 2}).json()
        self.assertEqual(result["training_steps"], 4)

    def test_validation(self):
        self.assertEqual(self.client.post("/api/estimate", json={"batch_size": 0}).status_code, 422)

    def test_memory_limit(self):
        result = self.client.post("/api/estimate", json={"measured_peak_vram_gb": 24}).json()
        self.assertEqual(result["memory_status"], "exceeds_capacity")

    def test_catalog_is_explicitly_illustrative(self):
        result = self.client.get("/api/gpus").json()
        self.assertEqual(len(result["items"]), 3)
        self.assertTrue(all(item["illustrative"] for item in result["items"]))
        self.assertTrue(all(item["hourly_price_usd"] is None for item in result["items"]))

    def test_placeholders_never_claim_success(self):
        for path, payload in [
            ("/api/benchmarks", {"workload_description": "test", "gpu_option_id": "demo-24gb"}),
            ("/api/workloads/parse", {"description": "test"}),
        ]:
            self.assertEqual(self.client.post(path, json=payload).status_code, 501)

    def test_health_and_docs(self):
        self.assertEqual(self.client.get("/api/health").json(), {"status": "ok"})
        self.assertEqual(self.client.get("/openapi.json").status_code, 200)
