import argparse
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils import YAML

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=Path, required=True)
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--project', type=Path, required=True)
args = parser.parse_args()

torch.set_num_threads(4)
model = YOLO(str(args.weights.resolve()))
assert model.names[0] == 'person' and len(model.names) == 80
source = args.data.resolve()
data = YAML.load(source)
data['path'] = source.parent.as_posix()
data['names'] = model.names
args.project.mkdir(parents=True, exist_ok=True)
training_config = args.project / 'coco_person_preserved_head.yaml'
YAML.save(training_config, data)
model.train(
    data=str(training_config.resolve()), pretrained=True, device='cpu',
    epochs=2, imgsz=320, batch=8, workers=0, plots=False, amp=False, seed=42,
    freeze=23, optimizer='AdamW', lr0=0.0001, lrf=0.5, warmup_epochs=0,
    mosaic=0, close_mosaic=0, mixup=0, scale=0.1, translate=0.05,
    project=str(args.project.resolve()), name='yolo11n_pretrained_preserved_head',
)
