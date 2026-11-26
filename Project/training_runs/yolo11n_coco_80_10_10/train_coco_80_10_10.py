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
model.train(data=str(args.data.resolve()), pretrained=True, device='cpu',
            epochs=6, imgsz=320, batch=8, workers=0, plots=False, amp=False,
            seed=42, close_mosaic=0, project=str(args.project.resolve()),
            name='yolo11n_coco_80_10_10')

