import argparse
import contextlib
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=Path, required=True)
parser.add_argument('--dataset', type=Path, required=True)
parser.add_argument('--annotations', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--split', choices=['val', 'test'], required=True)
parser.add_argument('--confidence', type=float, nargs='*', default=[])
args = parser.parse_args()
if args.split == 'val' and args.confidence:
    parser.error('Validation uses the predeclared confidence sweep')
args.out.mkdir(parents=True, exist_ok=False)
person_id = 1
score_floor = 0.001
torch.set_num_threads(6)
cv2.setNumThreads(1)

manifest_path = args.dataset / 'dataset_summary.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
split_ids = set(manifest[args.split]['image_ids'])
assert len(split_ids) == 240
for other in {'train', 'val', 'test'} - {args.split}:
    assert split_ids.isdisjoint(manifest[other]['image_ids'])
with zipfile.ZipFile(args.annotations) as archive:
    source = json.loads(archive.read('annotations/instances_train2017.json'))
member = {
    'info': source['info'], 'categories': source['categories'],
    'images': [item for item in source['images'] if item['id'] in split_ids],
    'annotations': [
        item for item in source['annotations']
        if item['image_id'] in split_ids and item['category_id'] == person_id
    ],
}
annotation_file = args.out.parent / (args.out.name + '_instances.json')
annotation_file.write_text(json.dumps(member), encoding='utf-8')
gt = COCO(str(annotation_file))
image_ids = sorted(gt.getImgIds())
assert len(image_ids) == len(split_ids) and gt.cats[person_id]['name'] == 'person'
model = YOLO(str(args.weights))
assert model.names[0] == 'person', model.names
predictions = []
image_dir = args.dataset / 'images' / args.split
for start in range(0, len(image_ids), 16):
    batch_ids = image_ids[start:start + 16]
    images = []
    for image_id in batch_ids:
        filename = gt.imgs[image_id]['file_name']
        encoded = (image_dir / filename).read_bytes()
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert image is not None, filename
        images.append(image)
    output = model.predict(images, classes=[0], imgsz=320, conf=score_floor, iou=0.7, max_det=100, device='cpu', verbose=False)
    for image_id, result in zip(batch_ids, output, strict=True):
        for (x1, y1, x2, y2), score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=True):
            predictions.append({'image_id': image_id, 'category_id': person_id, 'bbox': [x1, y1, x2-x1, y2-y1], 'score': score})
    if start % 400 == 0 or start + len(batch_ids) == len(image_ids):
        print(f'Processed {start + len(batch_ids)}/{len(image_ids)} images', flush=True)

assert predictions
dt = gt.loadRes(predictions)
evaluation = COCOeval(gt, dt, 'bbox')
evaluation.params.imgIds = image_ids
evaluation.params.catIds = [person_id]
with (args.out / 'coco_summary.txt').open('w', encoding='utf-8') as summary:
    with contextlib.redirect_stdout(summary):
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()

score_chunks, match_chunks = [], []
total_ground_truth = 0
for item in evaluation.evalImgs:
    if item is None or item['aRng'] != evaluation.params.areaRng[0]:
        continue
    usable = ~np.asarray(item['dtIgnore'][0], dtype=bool)
    score_chunks.append(np.asarray(item['dtScores'], dtype=float)[usable])
    match_chunks.append((item['dtMatches'][0] > 0)[usable])
    total_ground_truth += int(np.count_nonzero(~np.asarray(item['gtIgnore'], dtype=bool)))
scores = np.concatenate(score_chunks)
matches = np.concatenate(match_chunks)

def score_at(confidence):
    kept = scores >= confidence
    tp = int(np.count_nonzero(kept & matches))
    fp = int(np.count_nonzero(kept & ~matches))
    fn = total_ground_truth - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_ground_truth
    f2 = 5 * precision * recall / (4 * precision + recall) if precision or recall else 0.0
    return {
        'confidence': confidence, 'precision': precision, 'recall': recall, 'f2': f2,
        'true_positives': tp, 'false_positives': fp, 'false_negatives': fn,
    }

if args.split == 'val':
    thresholds = [i / 100 for i in range(1, 51)]
else:
    thresholds = sorted(set([0.30, *args.confidence]))
    assert all(score_floor <= confidence <= 1 for confidence in thresholds)
rows = [score_at(confidence) for confidence in thresholds]
results = {
    'dataset': 'COCO train2017 project split', 'split': args.split,
    'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    'images': len(image_ids), 'category': 'person', 'ground_truth_non_crowd': total_ground_truth,
    'weights_sha256': hashlib.sha256(args.weights.read_bytes()).hexdigest(),
    'mAP_50_95': float(evaluation.stats[0]), 'mAP_50': float(evaluation.stats[1]),
    'image_size': 320, 'ap_score_floor': score_floor, 'nms_iou': 0.7,
    'max_detections_per_image': 100, 'precision_recall_iou': 0.50,
    'device': 'cpu', 'ultralytics_version': version('ultralytics'),
    'pycocotools_version': version('pycocotools'),
    'operating_points': rows,
}
if args.split == 'val':
    eligible = [row for row in rows if row['precision'] >= 0.25]
    results['selection'] = {
        'rule': 'Maximize F2 at validation precision >= 25%. If unavailable, maximize F2 over all declared thresholds.',
        'precision_floor_met': bool(eligible),
        'chosen': max(eligible or rows, key=lambda row: (row['f2'], row['recall'], row['confidence'])),
    }
(args.out / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
with (args.out / 'operating_points.csv').open('w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print('mAP50:95', results['mAP_50_95'], 'mAP50', results['mAP_50'], flush=True)
print(results.get('selection', rows), flush=True)

