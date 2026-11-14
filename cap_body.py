"""CapBody Counter: evidence matching and conservative review proposals.

Copyright (c) 2026 CapBody Counter contributors. Distributed under MIT.
"""
from dataclasses import asdict, dataclass
import math
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from local_count import validate_boxes


@dataclass(frozen=True)
class MatchingConfig:
    nms_iou: float = 0.5
    min_body_aspect: float = 1.6
    max_horizontal_offset: float = 0.6
    cap_above_body: float = 0.15
    cap_below_body: float = 0.35
    min_cap_width_ratio: float = 0.15
    max_cap_width_ratio: float = 0.95
    max_cap_height_ratio: float = 0.35
    max_pair_cost: float = 0.65
    ambiguity_margin: float = 0.15
    horizontal_weight: float = 0.6


CONFIG = MatchingConfig()


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    return intersection / max(area_a+area_b-intersection, 1e-12)


def deduplicate(detections, threshold):
    ordered = sorted(detections, key=lambda d: (-d['score'], *d['box']))
    kept = []
    for detection in ordered:
        if all(iou(detection['box'], other['box']) <= threshold for other in kept):
            kept.append(detection)
    # Spatial ordering provides reproducible IDs independently of model query order.
    return sorted(kept, key=lambda d: (d['box'][1], d['box'][0], *d['box'][2:]))


def pair_cost(body, cap, config=CONFIG):
    x1, y1, x2, y2 = body
    cx1, cy1, cx2, cy2 = cap
    width, height = x2-x1, y2-y1
    if width <= 0 or height <= 0:
        return None
    cx, cy = (cx1+cx2)/2, (cy1+cy2)/2
    dx, dy = abs(cx-(x1+x2)/2)/width, (cy-y1)/height
    ratio = (cx2-cx1)/width
    if (dx > config.max_horizontal_offset or not -config.cap_above_body <= dy <= config.cap_below_body
            or not config.min_cap_width_ratio <= ratio <= config.max_cap_width_ratio
            or (cy2-cy1)/height > config.max_cap_height_ratio):
        return None
    cost = config.horizontal_weight * dx/config.max_horizontal_offset
    cost += (1-config.horizontal_weight) * abs(dy)/config.cap_below_body
    return cost if cost <= config.max_pair_cost else None


def match_detections(body_detections, cap_detections, config=CONFIG):
    objects, bodies, caps = [], [], []
    for kind, rows, destination, prefix in (
        ('body', body_detections, bodies, 'b'), ('cap', cap_detections, caps, 'c')
    ):
        for index, detection in enumerate(deduplicate(rows, config.nms_iou), 1):
            box = list(detection['box'])
            short = kind == 'body' and (box[3]-box[1])/max(box[2]-box[0], 1e-12) < config.min_body_aspect
            item = {'id': f'{prefix}{index:03d}', 'kind': kind, 'box': box,
                    'score': float(detection['score']), 'status': 'accepted' if kind == 'body' and not short else 'review',
                    'partner_id': None, 'source': 'model',
                    'reason': 'short_body' if short else 'unmatched_cap' if kind == 'cap' else 'body_detection'}
            objects.append(item)
            if not short:
                destination.append(item)
    links = []
    if bodies and caps:
        costs = np.full((len(bodies), len(caps)), np.inf)
        for bi, body in enumerate(bodies):
            for ci, cap in enumerate(caps):
                value = pair_cost(body['box'], cap['box'], config)
                if value is not None:
                    costs[bi, ci] = value
        # Dummy columns allow every body to remain unmatched.
        augmented = np.full((len(bodies), len(caps)+len(bodies)), config.max_pair_cost+1)
        augmented[:, :len(caps)] = np.where(np.isfinite(costs), costs, 1e6)
        assigned_bodies, assigned_caps = linear_sum_assignment(augmented)
        for bi, ci in zip(assigned_bodies, assigned_caps):
            if ci >= len(caps) or not math.isfinite(costs[bi, ci]):
                continue
            cost = costs[bi, ci]
            alternatives = list(np.delete(costs[bi, :], ci)) + list(np.delete(costs[:, ci], bi))
            ambiguous = any(math.isfinite(v) and v < cost+config.ambiguity_margin for v in alternatives)
            body, cap = bodies[bi], caps[ci]
            if ambiguous:
                cap['reason'] = 'ambiguous_match'
                cap['suggested_body_id'] = body['id']
                continue
            body['partner_id'], cap['partner_id'] = cap['id'], body['id']
            cap.update(status='linked', reason='matched_cap')
            links.append({'body_id': body['id'], 'cap_id': cap['id'], 'cost': round(float(cost), 4), 'source': 'auto'})
    return {'objects': objects, 'links': links,
            'summary': {'automatic_count': sum(o['status'] == 'accepted' for o in objects),
                        'pending_review': sum(o['status'] == 'review' for o in objects),
                        'matched_pairs': len(links),
                        'body_proposals': len(body_detections), 'cap_proposals': len(cap_detections)}}


def predict_cap_body(counter, image, cap_exemplars, body_exemplars=None,
                     body_threshold=0.23, cap_threshold=0.23):
    start = time.monotonic()
    validate_boxes(cap_exemplars, image.size)
    if not 1 <= len(cap_exemplars) <= 3:
        raise ValueError('Mark one to three cap examples before analysis')
    body_exemplars = [] if body_exemplars is None else body_exemplars
    validate_boxes(body_exemplars, image.size)
    if len(body_exemplars) > 3:
        raise ValueError('Use at most three body examples')
    for threshold in (body_threshold, cap_threshold):
        if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError('Thresholds must be between 0 and 1')
    bodies = counter.predict(image, text='plastic bottle', exemplars=body_exemplars, threshold=body_threshold)
    caps = counter.predict(image, text='bottle cap', exemplars=cap_exemplars, threshold=cap_threshold)
    result = match_detections(bodies['detections'], caps['detections'])
    result.update(application='CapBody Counter', schema_version=1, mode='cap_body',
                  width=image.width, height=image.height, device=counter.device,
                  cap_exemplars=cap_exemplars, body_exemplars=body_exemplars,
                  body_threshold=body_threshold, cap_threshold=cap_threshold,
                  matching_config=asdict(CONFIG), model_source=counter.source,
                  baseline={'prompt': 'plastic bottle', 'count': bodies['count']},
                  elapsed_seconds=round(time.monotonic()-start, 3),
                  note='Unmatched caps require review. Fully hidden stock is not estimated.')
    return result


def annotate_evidence(image, result):
    from PIL import ImageDraw
    image = image.copy()
    draw = ImageDraw.Draw(image)
    by_id = {o['id']: o for o in result['objects']}
    for link in result['links']:
        a, b = by_id[link['body_id']]['box'], by_id[link['cap_id']]['box']
        draw.line(((a[0]+a[2])/2, a[1], (b[0]+b[2])/2, (b[1]+b[3])/2), fill='#17978c', width=2)
    colors = {'accepted': '#16845a', 'linked': '#17978c', 'review': '#ce8a17', 'rejected': '#86918b'}
    for obj in result['objects']:
        draw.rectangle(obj['box'], outline=colors[obj['status']], width=2)
        draw.text((obj['box'][0]+2, obj['box'][1]+2), obj['id'], fill='white', stroke_width=2, stroke_fill='black')
    draw.rectangle((0, 0, image.width, 25), fill='#143a32')
    summary = result['summary']
    draw.text((8, 5), f"CapBody | automatic: {summary['automatic_count']} | review: {summary['pending_review']}", fill='white')
    return image
