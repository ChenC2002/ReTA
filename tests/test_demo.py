"""The demo serves real predictor responses and rejects invalid histories."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('fastapi') and importlib.util.find_spec('httpx'),
                     'Optional demo dependencies are required')
class DemoAPITests(unittest.TestCase):
    def test_model_loaded_once_and_inputs_validated(self):
        from fastapi.testclient import TestClient
        from reta.demo.app import create_app
        calls = []
        class Model:
            def metadata(self):
                return {'version': 'V2.0'}
            def predict(self, visits, top_k):
                calls.append((visits, top_k))
                if visits == [['UNKNOWN']]:
                    raise ValueError('Unknown ICD-9-CM codes: UNKNOWN')
                return {'steps': [], 'elapsed_ms': 1}
        loaded = []
        def factory():
            loaded.append(True)
            return Model()
        with TestClient(create_app(factory)) as client:
            self.assertEqual(client.get('/').status_code, 200)
            self.assertEqual(client.get('/api/meta').json()['version'], 'V2.0')
            self.assertEqual(client.post('/api/predict', json={'visits': [['25000']]}).status_code, 200)
            self.assertEqual(client.post('/api/predict', json={'visits': []}).status_code, 422)
            self.assertEqual(client.post('/api/predict', json={'visits': [['25000']] * 7}).status_code, 422)
            self.assertEqual(client.post('/api/predict', json={'visits': [['25000']], 'top_k': 1000}).status_code, 422)
            response = client.post('/api/predict', json={'visits': [['UNKNOWN']]})
            self.assertEqual(response.status_code, 422)
            self.assertIn('Unknown ICD', response.json()['detail'])
        self.assertEqual(len(loaded), 1)
        self.assertEqual(len(calls), 2)
