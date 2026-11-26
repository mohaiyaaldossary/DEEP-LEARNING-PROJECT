"""Reproducible CPU comparison. Labels are project YOLO labels, not official COCO GT."""
import argparse, contextlib, csv, hashlib, io, json, platform, time
from pathlib import Path
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.transforms.functional import to_tensor

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'COCO2017_person_80_10_10'
CANDIDATES=[('yolo_original_320','yolo11n.pt',320),('yolo_original_640','yolo11n.pt',640),('yolo_finetuned_320','training_runs/yolo11n_pretrained_preserved_head/weights/best.pt',320),('yolo_finetuned_640','training_runs/yolo11n_pretrained_preserved_head/weights/best.pt',640),('ssdlite_320',None,320)]

def ground_truth(split):
    images=[]; anns=[]
    for p in sorted((DATA/'images'/split).glob('*.jpg')):
        im=cv2.imread(str(p));h,w=im.shape[:2];iid=int(p.stem)
        images.append(dict(id=iid,file_name=p.name,width=w,height=h))
        for line in (DATA/'labels'/split/(p.stem+'.txt')).read_text().splitlines():
            c,x,y,bw,bh=map(float,line.split());assert c==0
            box=[(x-bw/2)*w,(y-bh/2)*h,bw*w,bh*h]
            anns.append(dict(id=len(anns)+1,image_id=iid,category_id=1,bbox=box,area=box[2]*box[3],iscrowd=0))
    gt=COCO();gt.dataset=dict(info={},images=images,annotations=anns,categories=[dict(id=1,name='person')]);gt.createIndex()
    return gt

def evaluate(gt,preds):
    with contextlib.redirect_stdout(io.StringIO()):
        dt=gt.loadRes(preds);ev=COCOeval(gt,dt,'bbox');ev.params.catIds=[1];ev.evaluate();ev.accumulate();ev.summarize()
    scores=[];matches=[];ngt=0
    for item in ev.evalImgs:
        if item is None or item['aRng'] != ev.params.areaRng[0]:continue
        usable=~np.asarray(item['dtIgnore'][0],bool)
        scores.extend(np.asarray(item['dtScores'])[usable]);matches.extend((item['dtMatches'][0]>0)[usable])
        ngt+=int(np.count_nonzero(~np.asarray(item['gtIgnore'],bool)))
    scores=np.asarray(scores);matches=np.asarray(matches,bool)
    rows=[]
    for t in np.arange(0.01,0.91,0.01):
        keep=scores>=t;tp=int((keep&matches).sum());fp=int((keep&~matches).sum());fn=ngt-tp
        p=tp/(tp+fp) if tp+fp else 0;r=tp/ngt if ngt else 0
        rows.append(dict(confidence=round(float(t),2),precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0,tp=tp,fp=fp,fn=fn))
    return dict(map50_95=float(ev.stats[0]),map50=float(ev.stats[1]),ground_truth=ngt,operating_points=rows)

def main():
    global CANDIDATES
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'verified_benchmark');ap.add_argument('--suite',choices=['base','capacity'],default='base');args=ap.parse_args();args.out.mkdir(exist_ok=True)
    if args.suite=='capacity':CANDIDATES=[('yolo11s_640','yolo11s.pt',640),('yolo11m_640','yolo11m.pt',640)]
    if (ROOT / "model_cache/checkpoints/ssdlite320_mobilenet_v3_large_coco-a79551df.pth").is_file():
        torch.hub.set_dir(str(ROOT / "model_cache"))
    torch.set_num_threads(4);cv2.setNumThreads(1)
    manifest=json.loads((DATA/'dataset_summary.json').read_text())
    for a,b in [('train','val'),('train','test'),('val','test')]:assert set(manifest[a]['image_ids']).isdisjoint(manifest[b]['image_ids'])
    protocol=dict(candidates=CANDIDATES,selection='Maximum validation F1 over confidence 0.01 to 0.90; ties by mAP then name. Test never selects.',labels='Project YOLO labels converted to COCO; crowd metadata unavailable; not official COCO metrics.',device='cpu',threads=4,platform=platform.platform(),torch=torch.__version__,manifest_sha256=hashlib.sha256((DATA/'dataset_summary.json').read_bytes()).hexdigest())
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    summary={}
    for split in ['val','test']:
        gt=ground_truth(split)
        for name,weight,size in CANDIDATES:
            output=args.out/f'{split}_{name}.json'
            if output.exists():summary[f'{split}_{name}']=json.loads(output.read_text());continue
            if weight:model=YOLO(str(ROOT/weight))
            else:model=ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,score_thresh=0.001,detections_per_img=100).eval()
            preds=[];timings=[]
            for i,imeta in enumerate(gt.dataset['images']):
                im=cv2.imread(str(DATA/'images'/split/imeta['file_name']))
                t=time.perf_counter()
                if weight:
                    result=model.predict(im,classes=[0],imgsz=size,conf=0.001,iou=0.7,max_det=100,device='cpu',verbose=False)[0]
                    boxes=result.boxes.xyxy.cpu().numpy();scores=result.boxes.conf.cpu().numpy()
                else:
                    with torch.inference_mode():result=model([to_tensor(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))])[0]
                    mask=result['labels']==1;boxes=result['boxes'][mask].numpy();scores=result['scores'][mask].numpy()
                timings.append(time.perf_counter()-t)
                for (x1,y1,x2,y2),score in zip(boxes,scores):
                    preds.append(dict(image_id=imeta['id'],category_id=1,bbox=[float(x1),float(y1),float(x2-x1),float(y2-y1)],score=float(score)))
                if i%60==0:print(split,name,i,'/',len(gt.imgs),flush=True)
            (args.out/f'{split}_{name}_predictions.json').write_text(json.dumps(preds))
            result=evaluate(gt,preds);result.update(model=name,split=split,images=len(gt.imgs),mean_inference_seconds=float(np.mean(timings[1:])),image_size=size)
            if split=='val':result['chosen']=max(result['operating_points'],key=lambda r:(r['f1'],r['confidence']))
            else:
                threshold=summary[f'val_{name}']['chosen']['confidence'];result['chosen']=next(r for r in result['operating_points'] if r['confidence']==threshold)
                # Test exposes only the locked operating point to avoid selecting on it.
                result.pop('operating_points')
            output.write_text(json.dumps(result,indent=2));summary[f'{split}_{name}']=result
        if split=='val':
            winner=max(CANDIDATES,key=lambda c:(summary['val_'+c[0]]['chosen']['f1'],summary['val_'+c[0]]['map50_95'],c[0]))[0]
            (args.out/'selection.json').write_text(json.dumps(dict(winner=winner,validation=summary['val_'+winner]['chosen']),indent=2))
    rows=[]
    for key,r in summary.items():rows.append(dict(model=r['model'],split=r['split'],map50_95=r['map50_95'],map50=r['map50'],fps=1/r['mean_inference_seconds'],**r['chosen']))
    with (args.out/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('DONE',flush=True)
if __name__=='__main__':main()
