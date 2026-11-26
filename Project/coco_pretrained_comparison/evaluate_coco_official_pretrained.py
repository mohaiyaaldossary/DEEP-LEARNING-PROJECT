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
parser.add_argument('--images', type=Path, required=True)
parser.add_argument('--annotations', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--test-manifest', type=Path)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
person_id = 1
score_floor = 0.001
report_conf = 0.30
torch.set_num_threads(4)
cv2.setNumThreads(1)

with zipfile.ZipFile(args.annotations) as archive:
    if args.test_manifest:
        manifest = json.loads(args.test_manifest.read_text(encoding='utf-8'))
        holdout_ids = set(manifest['test']['image_ids'])
        assert len(holdout_ids) == 240
        for split in ('train', 'val'):
            assert holdout_ids.isdisjoint(manifest[split]['image_ids'])
        source = json.loads(archive.read('annotations/instances_train2017.json'))
        member = {
            'info': source['info'],
            'categories': source['categories'],
            'images': [item for item in source['images'] if item['id'] in holdout_ids],
            'annotations': [item for item in source['annotations'] if item['image_id'] in holdout_ids and item['category_id'] == person_id],
        }
        annotation_data = json.dumps(member).encode('utf-8')
        dataset_name = 'COCO train2017 (project test holdout)'
        expected_size = 240
    else:
        annotation_data = archive.read('annotations/instances_val2017.json')
        dataset_name = 'COCO val2017'
        expected_size = 5000
annotation_file = args.out.parent / (args.out.name + '_instances.json')
annotation_file.write_bytes(annotation_data)
gt = COCO(str(annotation_file))
image_ids = sorted(gt.getImgIds())
assert len(image_ids) == expected_size and gt.cats[person_id]['name'] == 'person'
model = YOLO(str(args.weights))
assert model.names[0] == 'person', model.names
predictions = []
image_context = contextlib.nullcontext() if args.test_manifest else zipfile.ZipFile(args.images)
with image_context as archive:
    for start in range(0, len(image_ids), 16):
        batch_ids = image_ids[start:start + 16]
        images = []
        for image_id in batch_ids:
            filename = gt.imgs[image_id]['file_name']
            encoded = (args.images / filename).read_bytes() if args.test_manifest else archive.read('val2017/' + filename)
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

tp = fp = fn = 0
for item in evaluation.evalImgs:
    if item is None or item['aRng'] != evaluation.params.areaRng[0]:
        continue
    valid = (np.asarray(item['dtScores']) >= report_conf) & ~np.asarray(item['dtIgnore'][0], dtype=bool)
    matched = item['dtMatches'][0] > 0
    image_tp = np.count_nonzero(valid & matched)
    tp += int(image_tp)
    fp += int(np.count_nonzero(valid & ~matched))
    fn += int(np.count_nonzero(~np.asarray(item['gtIgnore'], dtype=bool)) - image_tp)

results = {
    'dataset': dataset_name, 'category': 'person', 'images': len(image_ids),
    'weights': args.weights.name, 'weights_sha256': hashlib.sha256(args.weights.read_bytes()).hexdigest(),
    'image_size': 320, 'device': 'cpu', 'ap_score_floor': score_floor,
    'nms_iou': 0.7, 'max_detections_per_image': 100,
    'precision_recall_confidence': report_conf, 'precision_recall_iou': 0.50,
    'mAP_50_95': float(evaluation.stats[0]), 'mAP_50': float(evaluation.stats[1]),
    'mAP_75': float(evaluation.stats[2]),
    'precision': tp / (tp + fp) if tp + fp else 0.0,
    'recall': tp / (tp + fn) if tp + fn else 0.0,
    'true_positives': tp, 'false_positives': fp, 'false_negatives': fn,
    'ultralytics_version': version('ultralytics'), 'pycocotools_version': version('pycocotools'),
    'image_source': ('https://cocodataset.org/#download' if args.test_manifest else 'http://images.cocodataset.org/zips/val2017.zip'),
    'annotation_archive': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
}
(args.out / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
with (args.out / 'metrics.csv').open('w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=results)
    writer.writeheader()
    writer.writerow(results)
print(json.dumps(results, indent=2))



