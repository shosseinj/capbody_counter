import io
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from cap_body import match_detections, pair_cost, predict_cap_body
import local_service


def detection(box, score=.8):
    return {'box': box, 'score': score}


class MatchingTests(unittest.TestCase):
    def test_body_cap_counted_once(self):
        result = match_detections([detection([0, 20, 100, 300])], [detection([25, 15, 75, 45])])
        self.assertEqual(result['summary']['automatic_count'], 1)
        self.assertEqual(result['summary']['pending_review'], 0)
        self.assertEqual(len(result['links']), 1)
        self.assertEqual(result['objects'][1]['status'], 'linked')

    def test_duplicates_are_suppressed(self):
        result = match_detections([detection([0, 20, 100, 300]), detection([1, 21, 101, 301], .7)],
                                  [detection([25, 15, 75, 45]), detection([26, 16, 76, 46], .7)])
        self.assertEqual(len(result['objects']), 2)
        self.assertEqual(result['summary']['automatic_count'], 1)

    def test_cap_only_never_auto_counted(self):
        result = match_detections([], [detection([25, 15, 75, 45])])
        self.assertEqual(result['summary']['automatic_count'], 0)
        self.assertEqual(result['summary']['pending_review'], 1)

    def test_ambiguous_cap_not_attached(self):
        result = match_detections([detection([0, 20, 100, 300]), detection([40, 20, 140, 300])],
                                  [detection([50, 15, 90, 45])])
        self.assertEqual(result['links'], [])
        self.assertEqual(result['objects'][-1]['reason'], 'ambiguous_match')
        self.assertEqual(result['summary']['automatic_count'], 2)

    def test_one_to_one_assignment(self):
        result = match_detections([detection([0, 20, 100, 300]), detection([120, 20, 220, 300])],
                                  [detection([25, 15, 75, 45]), detection([145, 15, 195, 45])])
        self.assertEqual(len(result['links']), 2)
        self.assertEqual(len({l['body_id'] for l in result['links']}), 2)
        self.assertEqual(len({l['cap_id'] for l in result['links']}), 2)

    def test_short_body_requires_review(self):
        result = match_detections([detection([0, 0, 50, 40])], [])
        self.assertEqual(result['summary']['automatic_count'], 0)
        self.assertEqual(result['objects'][0]['reason'], 'short_body')

    def test_invalid_geometry_not_paired(self):
        self.assertIsNone(pair_cost([0, 20, 100, 300], [200, 15, 250, 45]))
        self.assertIsNone(pair_cost([0, 20, 100, 300], [25, 200, 75, 230]))

    def test_empty_detections(self):
        result = match_detections([], [])
        self.assertEqual(result['objects'], [])
        self.assertEqual(result['summary']['automatic_count'], 0)

    def test_examples_validated_before_inference(self):
        class NeverCalled:
            def predict(self, *args, **kwargs):
                raise AssertionError('Invalid input reached inference')
        for boxes in ([], [[0, 0, 999, 20]], [[0, 0, 1, 1]]*4):
            with self.subTest(boxes=boxes), self.assertRaises(ValueError):
                predict_cap_body(NeverCalled(), Image.new('RGB', (100, 100)), boxes)


class FakeCounter:
    device = 'cpu'
    parameter_count = 123
    source = {'repo_id': 'test-double'}

    def predict(self, image, text='bottle', **kwargs):
        rows = [detection([25, 15, 75, 45])] if text == 'bottle cap' else [detection([0, 20, 100, 300])]
        return {'count': len(rows), 'detections': rows, 'backend': 'countgdplusplus',
                'target': text, 'width': image.width, 'height': image.height}


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch.object(local_service, 'Counter', FakeCounter)
        self.patch.start()
        self.client = TestClient(local_service.app)
        self.client.__enter__()
        buffer = io.BytesIO()
        Image.new('RGB', (300, 400)).save(buffer, format='PNG')
        self.payload = buffer.getvalue()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.patch.stop()

    def post(self, endpoint, **fields):
        return self.client.post(endpoint, files={'file': ('test.png', self.payload, 'image/png')}, data=fields)

    def test_workspace_and_health(self):
        self.assertIn('Count with', self.client.get('/').text)
        self.assertEqual(self.client.get('/assets/state.mjs').status_code, 200)
        self.assertTrue(self.client.get('/health').json()['ready'])

    def test_original_api_unchanged(self):
        response = self.post('/count', prompt='bottle', include_image='false')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['count'], 1)
        self.assertNotIn('annotated_image_base64', response.json())

    def test_cap_body_api(self):
        response = self.post('/count/cap-body', cap_exemplars='[[25,15,75,45]]')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['summary']['automatic_count'], 1)
        self.assertEqual(len(response.json()['links']), 1)

    def test_bad_cap_input_returns_422(self):
        for value in ('[]', 'false', '[', '[[0,0,999,20]]'):
            with self.subTest(value=value):
                self.assertEqual(self.post('/count/cap-body', cap_exemplars=value).status_code, 422)

    def test_exif_image_orientation(self):
        buffer = io.BytesIO()
        exif = Image.Exif(); exif[274] = 6
        Image.new('RGB', (40, 20)).save(buffer, format='JPEG', exif=exif)
        from starlette.datastructures import UploadFile
        image = local_service.read_image(UploadFile(io.BytesIO(buffer.getvalue()), filename='rotated.jpg'))
        self.assertEqual(image.size, (20, 40))


if __name__ == '__main__':
    unittest.main()
