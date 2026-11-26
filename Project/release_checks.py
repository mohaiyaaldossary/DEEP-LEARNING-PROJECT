"""Lock the image preset, measure latency serially, then replay and score Walk3."""
from pathlib import Path
from time import perf_counter
import csv, json, subprocess, sys, statistics, hashlib
import cv2
import torch
from ultralytics import YOLO

root=Path(__file__).resolve().parent
folders=['verified_benchmark','capacity_benchmark','high_resolution_benchmark','large_benchmark']
folders.append('yolo26_benchmark')
rows=[]
for folder in folders:
    with (root/folder/'comparison.csv').open() as source:
        rows.extend(csv.DictReader(source))
winner=max((r for r in rows if r['split']=='val'),key=lambda r:(float(r['f1']),float(r['map50_95']),r['model']))
assert winner['model'] in ('yolo11m_640','yolo11m_960','yolo11l_640','yolo26m_640')
model_name, size=winner['model'].split('_')
cfg=json.loads((root/'final_config.json').read_text())
cfg.update(model=model_name+'.pt',imgsz=int(size),conf=float(winner['confidence']),reason='Highest validation F1 of ten measured configurations')
(root/'accuracy_config.json').write_text(json.dumps(cfg,indent=2))
with (root/'evaluation/release_comparison.csv').open('w') as out:
    wr=csv.DictWriter(out,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
torch.set_num_threads(4)
cv2.setNumThreads(1)
paths=sorted((root/'COCO2017_person_80_10_10/images/val').glob('*.jpg'))[:45]
images=[cv2.imread(str(p)) for p in paths]
timings={}
for weight, size in [('yolo11m.pt',640),('yolo11m.pt',960),('yolo11l.pt',640),('yolo26m.pt',640)]:
    model=YOLO(str(root/weight))
    elapsed=[]
    for frame in images:
        start=perf_counter()
        model.predict(frame,imgsz=size,conf=.001,iou=.7,max_det=100,classes=[0],device='cpu',verbose=False)
        elapsed.append(perf_counter()-start)
    timings[f'{Path(weight).stem}_{size}']=1/statistics.mean(elapsed[5:])
(root/'evaluation/release_timing.json').write_text(json.dumps(timings,indent=2))
for video in ['CAVIAR_Walk2_Reentry.mp4','CAVIAR_Walk3_Test.mp4']:
    out=root/'accuracy_runs'/Path(video).stem
    subprocess.run([sys.executable,str(root/'run_project.py'),video,'--preset','accuracy','--headless','--save-video','--out',str(out)],cwd=root,check=True)
    run=max(out.iterdir(),key=lambda p:p.stat().st_mtime)
    cmd=[sys.executable,str(root/'evaluate_video.py'),str(run)]
    if 'Walk3' in video:cmd+=['--annotations',str(root/'evaluation/wk3gt.xml'),'--start-frame','0']
    subprocess.run(cmd,cwd=root,check=True)
    print(video,run.name,flush=True)
