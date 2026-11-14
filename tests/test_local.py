import unittest

import torch
from transformers import BertConfig, BertModel

from local_count import filter_predictions, validate_boxes
from models.GroundingDINO.bertwarper import BertModelWarper


class LocalInferenceTests(unittest.TestCase):
    def test_positive_negative_filtering(self):
        # Third token is the positive/negative separator, fourth a negative category.
        probabilities = torch.tensor([[[.01, .8, .01, .2, .01],
                                       [.01, .9, .01, .95, .01],
                                       [.01, .1, .01, .05, .01]]])
        output = {"input_ids": torch.tensor([[101, 200, 1012, 201, 102]]),
                  "pred_logits": torch.logit(probabilities), "pred_boxes": torch.rand(1, 3, 4)}
        boxes, scores = filter_predictions(output, .23)
        self.assertEqual(len(boxes), 1)
        torch.testing.assert_close(boxes, output["pred_boxes"][0, :1])
        self.assertAlmostEqual(float(scores[0]), .8)

    def test_no_separator_rejected(self):
        with self.assertRaises(ValueError):
            filter_predictions({"input_ids": torch.tensor([[101, 102]])}, .23)

    def test_invalid_box_inputs(self):
        for boxes in ("bad", [[0, 0, float("nan"), 10]], [[10, 0, 5, 10]], [[0, 0, 101, 10]]):
            with self.subTest(boxes=boxes), self.assertRaises(ValueError):
                validate_boxes(boxes, (100, 100))

    def test_bert_mask_isolation_and_checkpoint_keys(self):
        model = BertModel(BertConfig(hidden_size=32, intermediate_size=64,
                                     num_hidden_layers=1, num_attention_heads=4)).eval()
        wrapper = BertModelWarper(model).eval()
        self.assertEqual(set(model.state_dict()), set(wrapper.state_dict()))
        ids = torch.tensor([[101, 20, 30, 102]])
        mask = torch.ones(1, 4, 4, dtype=torch.bool)
        mask[:, :, 2] = False
        with torch.inference_mode():
            first = wrapper(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
            ids[0, 2] = 40
            second = wrapper(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
        torch.testing.assert_close(first[:, [0, 1, 3]], second[:, [0, 1, 3]])


if __name__ == "__main__":
    unittest.main()
