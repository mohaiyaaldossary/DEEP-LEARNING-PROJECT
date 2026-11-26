import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('--select-only', action='store_true')
args = parser.parse_args()
root = args.root
models = ('previous_fine_tuned', 'original', 'new_fine_tuned')

def metrics(stage, name):
    folder = f'{stage}_{name}'
    if stage == 'validation' and name == 'new_fine_tuned':
        chosen = json.loads((root / 'checkpoint_selection.json').read_text(encoding='utf-8'))
        folder = chosen['winner']['validation_directory']
    return json.loads((root / folder / 'metrics.json').read_text(encoding='utf-8'))

def f1(row):
    p, r = row['precision'], row['recall']
    return 2 * p * r / (p + r) if p + r else 0.0

validation = {name: metrics('validation', name) for name in models}
selection = {
    name: max(result['operating_points'], key=lambda row: (f1(row), row['recall'], row['confidence']))
    for name, result in validation.items()
}
(root / 'validation_thresholds.json').write_text(json.dumps({
    'rule': 'Per-model maximum F1 on the 240 validation images, IoU 0.50. Ties prefer higher recall and then higher confidence.',
    'models': {name: {**row, 'f1': f1(row), 'weights_sha256': validation[name]['weights_sha256']}
               for name, row in selection.items()},
}, indent=2)+'\n', encoding='utf-8')
thresholds = sorted(set([0.07, 0.30, *(row['confidence'] for row in selection.values())]))
if args.select_only:
    print(' '.join(str(value) for value in thresholds))
    raise SystemExit

test = {name: metrics('test', name) for name in models}
official = {name: metrics('official', name) for name in models}
protocol = ('images', 'category', 'image_size', 'ap_score_floor', 'nms_iou',
            'max_detections_per_image', 'precision_recall_iou', 'device',
            'ultralytics_version', 'pycocotools_version')
for collection in (validation, test, official):
    for name, result in collection.items():
        assert all(result[key] == collection['original'][key] for key in protocol), name
for name in models:
    assert test[name]['split'] == 'test' and validation[name]['split'] == 'val'
    assert test[name]['manifest_sha256'] == validation[name]['manifest_sha256'] == test['original']['manifest_sha256']
    assert test[name]['weights_sha256'] == validation[name]['weights_sha256'] == official[name]['weights_sha256']
assert test['original']['images'] == 240 and official['original']['images'] == 5000
assert len({result['weights_sha256'] for result in test.values()}) == 3

def format_result(dataset, name, label, result, point, confidence):
    return {
        'dataset': dataset, 'images': result['images'], 'model': name,
        'operating_point': label, 'confidence': confidence,
        'mAP_50_95_percent': result['mAP_50_95'] * 100, 'mAP_50_percent': result['mAP_50'] * 100,
        'precision_percent': point['precision'] * 100, 'recall_percent': point['recall'] * 100,
        'f1_percent': f1(point) * 100, 'true_positives': point['true_positives'],
        'false_positives': point['false_positives'], 'false_negatives': point['false_negatives'],
    }

rows = []
for name in models:
    test_points = {p['confidence']: p for p in test[name]['operating_points']}
    chosen = selection[name]['confidence']
    for label, conf in [('shared_default', 0.30), ('validation_f1', chosen), ('shared_low', 0.07)]:
        rows.append(format_result('project_test', name, label, test[name], test_points[conf], conf))
    rows.append(format_result('official_val2017', name, 'shared_default', official[name], official[name], 0.30))
with (root / 'comparison.csv').open('w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows({k:f'{v:.6f}' if isinstance(v, float) else v for k,v in row.items()} for row in rows)
report = {
    'project_test_note': 'Images were held out of project fine-tuning and threshold selection, but are from COCO train2017, the pretraining data source for the original weight.',
    'official_val2017_note': 'These 5,000 images were not used to train, select checkpoints, or select operating thresholds in this project fine-tune. The original upstream model uses COCO train2017 for training and the upstream configuration uses val2017 for validation.',
    'mAP_note': 'Person bounding-box AP, calculated from all confidence scores at or above the floor 0.001.',
    'checkpoints': {name: result['weights_sha256'] for name, result in test.items()},
    'test_protocol': {key: test['original'][key] for key in protocol},
    'split_manifest_sha256': test['original']['manifest_sha256'],
    'comparison': rows,
}
(root / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print((root / 'comparison.csv').read_text())

