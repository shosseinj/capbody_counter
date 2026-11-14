import unittest
from evaluate_counts import evaluate


class EvaluationTests(unittest.TestCase):
    def test_metrics_against_independent_counts(self):
        report = evaluate({'a': 10, 'b': 8}, {'a': 12, 'b': 6})
        self.assertEqual(report, {'images': 2, 'mae': 2, 'rmse': 2, 'bias': 0})

    def test_missing_images_cannot_be_silently_dropped(self):
        with self.assertRaises(ValueError):
            evaluate({'a': 10, 'b': 8}, {'a': 10})

    def test_invalid_counts(self):
        for table in ({'a': -1}, {'a': True}, {'a': 1.5}, [], {}):
            with self.subTest(table=table), self.assertRaises(ValueError):
                evaluate({'a': 10}, table)
