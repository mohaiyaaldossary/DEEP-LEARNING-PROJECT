import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('comparison', type=Path)
parser.add_argument('training', type=Path)
args = parser.parse_args()
candidates = ('yolo11n_pretrained_coco_320', 'yolo11n_pretrained_preserved_head')

def f1(point):
    p, r = point['precision'], point['recall']
    return 2*p*r/(p+r) if p+r else 0

rows = []
for name in candidates:
    folder = args.comparison / f'validation_{name}'
    report = json.loads((folder / 'metrics.json').read_text())
    selected = max(report['operating_points'], key=lambda point: (f1(point), point['recall'], point['confidence']))
    rows.append({
        'training_run': name, 'validation_directory': folder.name,
        'weight_sha256': report['weights_sha256'],
        'mAP_50_95': report['mAP_50_95'], 'mAP_50': report['mAP_50'],
        'best_f1': f1(selected), 'confidence': selected['confidence'],
        'precision': selected['precision'], 'recall': selected['recall'],
    })
winner = max(rows, key=lambda row: (row['best_f1'], row['mAP_50_95']))
record = {
    'rule': 'Select among fine-tuned run best.pt checkpoints by maximum validation F1, breaking ties by validation mAP50-95. Each run best.pt was selected by its validation detection fitness.',
    'winner': winner, 'candidates': rows,
}
(args.comparison / 'checkpoint_selection.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print((args.training / winner['training_run'] / 'weights/best.pt').resolve())
