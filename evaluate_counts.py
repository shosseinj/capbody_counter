"""Compare count predictions against independently annotated ground truth.

Each JSON input is a mapping such as {"image-01": 12, "image-02": 8}.
"""
import argparse
import json
import math
from pathlib import Path


def evaluate(truth, predictions):
    if not isinstance(truth, dict) or not isinstance(predictions, dict) or not truth or truth.keys() != predictions.keys():
        raise ValueError('Ground truth and predictions must have the same nonempty image IDs')
    for table in (truth, predictions):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in table.values()):
            raise ValueError('Counts must be nonnegative integers')
    errors = [predictions[k]-truth[k] for k in sorted(truth)]
    return {'images': len(errors), 'mae': sum(map(abs, errors))/len(errors),
            'rmse': math.sqrt(sum(e*e for e in errors)/len(errors)), 'bias': sum(errors)/len(errors)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth', required=True, type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--cap-body', required=True, type=Path)
    args = parser.parse_args()
    try:
        truth, baseline, cap_body = [json.loads(p.read_text()) for p in (args.truth, args.baseline, args.cap_body)]
        report = {'baseline': evaluate(truth, baseline), 'cap_body': evaluate(truth, cap_body),
                  'note': 'Use independent annotations and separate reviewed results from automatic predictions.'}
        report['mae_change'] = report['cap_body']['mae']-report['baseline']['mae']
        print(json.dumps(report, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Evaluation failed: {exc}\n')


if __name__ == '__main__':
    main()
