"""CountGD++ single-image inference using the official trained weights."""
import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageOps
import torch
from safetensors.torch import load_model
from torchvision.transforms import functional as F

ROOT = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get("COUNTGD_WEIGHTS", str(ROOT / "checkpoints/model.safetensors")))


def validate_boxes(boxes, size):
    if not isinstance(boxes, list) or len(boxes) > 20:
        raise ValueError("Exemplars must be a list of at most 20 [x1,y1,x2,y2] pixel boxes")
    width, height = size
    for box in boxes:
        if (not isinstance(box, (list, tuple)) or len(box) != 4 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)):
            raise ValueError("Each box needs four finite pixel coordinates")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Invalid box {box} for {width}x{height} image")
    return boxes


def filter_predictions(output, threshold):
    """Apply the positive/negative filtering used by the official demo."""
    input_ids = output["input_ids"][0]
    split = (input_ids == 1012).nonzero(as_tuple=False)
    if not len(split):
        raise ValueError("Caption has no positive/negative separator")
    split = int(split[0, 0])
    logits = output["pred_logits"][0].sigmoid()
    positive = logits[:, :split+1].max(-1).values
    negative_logits = logits[:, split+1:]
    keep = positive > threshold
    if negative_logits.shape[-1]:
        keep &= positive > negative_logits.max(-1).values
    return output["pred_boxes"][0][keep].cpu(), positive[keep].cpu()


class Counter:
    def __init__(self, device=None):
        from groundingdino.util.slconfig import SLConfig
        from models.GroundingDINO.backbone import build_backbone
        from models.GroundingDINO.transformer import build_transformer
        from models.GroundingDINO.groundingdino_app import GroundingDINO

        if not WEIGHTS.is_file():
            raise FileNotFoundError(f"Official model weights missing: {WEIGHTS}")
        requested = device or os.environ.get("COUNT_DEVICE", "cpu")
        self.device = requested
        if requested == "auto":
            self.device = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 5*1024**3 else "cpu"
        torch.manual_seed(42)
        cfg = SLConfig.fromfile(str(ROOT / "cfg_app.py"))
        args = argparse.Namespace(**cfg._cfg_dict.to_dict())
        args.text_encoder_type = str(ROOT / "checkpoints/bert-base-uncased")
        self.model = GroundingDINO(
            build_backbone(args), build_transformer(args),
            num_queries=args.num_queries, aux_loss=args.aux_loss, iter_update=True,
            query_dim=4, num_feature_levels=args.num_feature_levels, nheads=args.nheads,
            dec_pred_bbox_embed_share=args.dec_pred_bbox_embed_share,
            two_stage_type=args.two_stage_type, two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
            two_stage_class_embed_share=args.two_stage_class_embed_share, num_patterns=args.num_patterns,
            dn_number=0, dn_box_noise_scale=args.dn_box_noise_scale,
            dn_label_noise_ratio=args.dn_label_noise_ratio, dn_labelbook_size=args.dn_labelbook_size,
            text_encoder_type=args.text_encoder_type, sub_sentence_present=args.sub_sentence_present,
            max_text_len=args.max_text_len, load_text_encoder_weights=False,
        )
        missing, unexpected = load_model(self.model, str(WEIGHTS), strict=False)
        # Older BERT versions persisted these generated, non-learned buffers.
        optional = ("position_ids", "token_type_ids")
        missing = [key for key in missing if not key.endswith(optional)]
        unexpected = [key for key in unexpected if not key.endswith(optional)]
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")
        self.model.to(self.device).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        if WEIGHTS.resolve() == (ROOT / "checkpoints/model.safetensors").resolve():
            self.source = json.loads((ROOT / "checkpoints/model-source.json").read_text())
        else:
            self.source = {"weights": str(WEIGHTS.resolve()), "provenance": "custom weights"}

    def _prepare(self, image, boxes):
        resized = F.resize(image, 800, max_size=1333)
        tensor = F.normalize(F.to_tensor(resized), [.485, .456, .406], [.229, .224, .225]).to(self.device)
        exemplars = torch.tensor(boxes, dtype=torch.float32, device=self.device).reshape(-1, 4)
        exemplars = exemplars * exemplars.new_tensor([resized.width/image.width, resized.height/image.height]*2)
        return tensor, exemplars

    def predict(self, image, text="bottle", exemplars=None, negative_texts=None,
                negative_exemplars=None, threshold=0.23):
        from groundingdino.util.misc import nested_tensor_from_tensor_list
        start = time.monotonic()
        exemplars = validate_boxes([] if exemplars is None else exemplars, image.size)
        if not isinstance(text, str) or len(text) > 500 or not (text.strip() or exemplars):
            raise ValueError("Provide a positive text prompt or exemplar boxes")
        negative_texts = [] if negative_texts is None else negative_texts
        if not isinstance(negative_texts, list) or len(negative_texts) > 10 or any(
                not isinstance(t, str) or len(t) > 500 for t in negative_texts):
            raise ValueError("negative_texts must be a list of at most 10 short category names")
        texts = [text.strip()] + [t.strip() for t in negative_texts]
        if any("." in t for t in texts):
            raise ValueError("Use category names without periods; negative categories have a separate field")
        caption = " . ".join(texts) + " . "
        neg_boxes = [[] for _ in negative_texts] if negative_exemplars is None else negative_exemplars
        if not isinstance(neg_boxes, list) or len(neg_boxes) != len(negative_texts):
            raise ValueError("Provide one negative exemplar list per negative text category")
        for group in neg_boxes:
            validate_boxes(group, image.size)
        token_count = len(self.model.tokenizer(caption)["input_ids"]) + len(exemplars) + sum(map(len, neg_boxes))
        if token_count > 256:
            raise ValueError("Combined prompts exceed the 256-token model limit")
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be between 0 and 1")
        tensor, pos_boxes = self._prepare(image, exemplars)
        nested = nested_tensor_from_tensor_list(tensor.unsqueeze(0))
        negative_images, negative_boxes = [], []
        for group in neg_boxes:
            if group:
                _, boxes = self._prepare(image, group)
                negative_images.append(nested)
            else:
                boxes = torch.empty((0, 4), device=self.device)
                negative_images.append(None)
            negative_boxes.append([boxes])
        with torch.inference_mode():
            output = self.model(nested, nested, [pos_boxes], negative_images, negative_boxes, captions=[caption])
            boxes, scores = filter_predictions(output, threshold)
        detections = []
        width, height = image.size
        for (cx, cy, bw, bh), score in zip(boxes.tolist(), scores.tolist()):
            detections.append({"box": [max(0, (cx-bw/2)*width), max(0, (cy-bh/2)*height),
                                       min(width, (cx+bw/2)*width), min(height, (cy+bh/2)*height)], "score": score})
        return {"backend": "countgdplusplus", "target": text, "negative_texts": negative_texts,
                "exemplars": exemplars, "negative_exemplars": neg_boxes,
                "count": len(detections), "count_type": "visible_object_detections",
                "occluded_count": None, "threshold": threshold, "device": self.device,
                "width": width, "height": height, "model_source": self.source,
                "elapsed_seconds": round(time.monotonic()-start, 3),
                "note": "Detectable partly visible objects may be counted. Fully hidden stock and physical shelf depth are not estimated. Predictions require review.",
                "detections": detections}


def annotate(image, result):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for number, detection in enumerate(result["detections"], 1):
        box = detection["box"]
        draw.rectangle(box, outline="#00ff88", width=2)
        draw.text((box[0]+2, max(25, box[1])), str(number), fill="white", stroke_width=2, stroke_fill="black")
    draw.rectangle((0, 0, image.width, 24), fill="black")
    draw.text((8, 5), f"CountGD++ | {result['target']} | predictions: {result['count']}", fill="white")
    return image


def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--prompt", default="bottle")
    parser.add_argument('--mode', choices=['standard', 'cap_body'], default='standard')
    parser.add_argument('--cap-exemplars', default='[]', help='One to three cap boxes in cap_body mode')
    parser.add_argument('--cap-threshold', default=.23, type=float)
    parser.add_argument("--exemplars", default="[]")
    parser.add_argument("--negative-texts", default="[]", help="JSON list of categories to exclude")
    parser.add_argument("--negative-exemplars", default="null", help="JSON lists of exemplar boxes, one list per negative category")
    parser.add_argument("--threshold", default=0.23, type=float)
    parser.add_argument("--device", default=os.environ.get("COUNT_DEVICE", "cpu"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/result")
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("COUNT_THREADS", "4")))
    try:
        with Image.open(args.image) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        print("Loading official CountGD++ weights...", flush=True)
        counter = Counter(args.device)
        print(f"Loaded {counter.parameter_count:,} parameters on {counter.device}", flush=True)
        if args.mode == 'cap_body':
            from cap_body import predict_cap_body, annotate_evidence
            result = predict_cap_body(counter, image, json.loads(args.cap_exemplars),
                                      json.loads(args.exemplars), args.threshold, args.cap_threshold)
            annotated = annotate_evidence(image, result)
        else:
            result = counter.predict(image, text=args.prompt, threshold=args.threshold,
                                     exemplars=json.loads(args.exemplars), negative_texts=json.loads(args.negative_texts),
                                     negative_exemplars=json.loads(args.negative_exemplars))
            annotated = annotate(image, result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Path(str(args.output)+".json").write_text(json.dumps(result, indent=2)+"\n")
        annotated.save(str(args.output)+".png")
        print(json.dumps({k: v for k, v in result.items() if k not in ('detections', 'objects', 'links')}, indent=2))
        print(f"Saved {args.output}.json and {args.output}.png", flush=True)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Counting failed: {exc}\n")


if __name__ == "__main__":
    main()
