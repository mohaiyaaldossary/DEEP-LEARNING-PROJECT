import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=Path, required=True)
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--project', type=Path, required=True)
args = parser.parse_args()

torch.set_num_threads(6)
model = YOLO(str(args.weights.resolve()))
model.train(
    data=str(args.data.resolve()), pretrained=True, device='cpu', epochs=5,
    imgsz=384, batch=8, workers=0, plots=False, amp=False, seed=42,
    optimizer='AdamW', lr0=0.001, lrf=0.15, warmup_epochs=0.5,
    mosaic=0.5, close_mosaic=2, project=str(args.project.resolve()),
    name='yolo11n_coco_recall_384',
)
