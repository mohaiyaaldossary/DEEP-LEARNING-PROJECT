"""Generate report figures and paired image-bootstrap uncertainty from saved outputs."""
from pathlib import Path
import json,csv,contextlib,io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycocotools.cocoeval import COCOeval
from benchmark_detectors import ground_truth
R=Path(__file__).resolve().parent;O=R/'evaluation';F=O/'figures';F.mkdir(exist_ok=True)
files=list((R/'verified_benchmark').glob('val_*.json'))+list((R/'capacity_benchmark').glob('val_*.json'))
vals=[json.loads(p.read_text()) for p in files if 'predictions' not in p.name]
labels={'yolo_original_320':'YOLO11n 320','yolo_original_640':'YOLO11n 640','yolo_finetuned_320':'Tuned YOLO11n 320','yolo_finetuned_640':'Tuned YOLO11n 640','ssdlite_320':'SSDLite 320','yolo11s_640':'YOLO11s 640','yolo11m_640':'YOLO11m 640'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(8,4.5))
for r in sorted(vals,key=lambda r:r['model']):
 rows=r['operating_points'];ax.plot([x['confidence'] for x in rows],[x['f1']*100 for x in rows],label=labels[r['model']])
ax.set(xlabel='Confidence threshold selected on validation only',ylabel='Validation F1 (%)',ylim=(0,90));ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(F/'threshold_search.png',dpi=180);plt.close(fig)
rows=[]
for folder in ['verified_benchmark','capacity_benchmark']:
 rows+=list(csv.DictReader((R/folder/'comparison.csv').open()))
tests=[r for r in rows if r['split']=='test'];tests.sort(key=lambda r:float(r['f1']))
fig,ax=plt.subplots(figsize=(8,4.3));yy=np.arange(len(tests));ax.barh(yy-.17,[float(r['f1'])*100 for r in tests],height=.32,label='F1 at validation threshold');ax.barh(yy+.17,[float(r['map50_95'])*100 for r in tests],height=.32,label='AP50 to 95');ax.set_yticks(yy,[labels[r['model']] for r in tests]);ax.set(xlabel='Project test score (%)',xlim=(0,100));ax.legend(fontsize=8);fig.tight_layout();fig.savefig(F/'model_comparison.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(7.5,4))
for r in tests:
 x=float(r['fps']);y=float(r['f1'])*100;ax.scatter(x,y,s=55)
 offsets={'yolo_original_640':(10,12),'yolo_finetuned_640':(10,-20),'yolo_original_320':(-65,12),'yolo_finetuned_320':(-65,-20)}
 ax.annotate(labels[r['model']],(x,y),xytext=offsets.get(r['model'],(5,7)),textcoords='offset points',fontsize=8,arrowprops=dict(arrowstyle='-',color='#888888',lw=.5))
ax.set(xlabel='Approximate CPU inference FPS (single pass)',ylabel='Project test F1 (%)',xlim=(0,145),ylim=(52,90));ax.grid(alpha=.2);fig.tight_layout();fig.savefig(F/'speed_accuracy.png',dpi=180);plt.close(fig)
# Paired bootstrap over images, not individual boxes; same sample indices for both models.
gt=ground_truth('test');counts=[]
for folder,name in [('verified_benchmark','yolo_original_320'),('capacity_benchmark','yolo11m_640')]:
 result=json.loads((R/folder/f'test_{name}.json').read_text());pred=json.loads((R/folder/f'test_{name}_predictions.json').read_text());threshold=result['chosen']['confidence']
 with contextlib.redirect_stdout(io.StringIO()):
  dt=gt.loadRes(pred);ev=COCOeval(gt,dt,'bbox');ev.params.catIds=[1];ev.evaluate()
 per={}
 for r in ev.evalImgs:
  if r is None or r['aRng']!=ev.params.areaRng[0]:continue
  keep=(np.asarray(r['dtScores'])>=threshold)&~np.asarray(r['dtIgnore'][0],bool);match=r['dtMatches'][0]>0;tp=int((keep&match).sum());fp=int((keep&~match).sum());ng=int((~np.asarray(r['gtIgnore'],bool)).sum());per[r['image_id']]=[tp,fp,ng-tp]
 counts.append(np.array([per.get(i,[0,0,0]) for i in sorted(gt.imgs)]))
rng=np.random.default_rng(42);idx=rng.integers(0,len(gt.imgs),(2000,len(gt.imgs)));scores=[]
for c in counts:
 s=c[idx].sum(axis=1);scores.append(2*s[:,0]/np.maximum(1,2*s[:,0]+s[:,1]+s[:,2]))
ci=dict(method='Paired image cluster bootstrap; 2000 replicates; seed 42; percentile 95% interval; conditional on this COCO-derived subset.',baseline_f1_ci95=np.percentile(scores[0],[2.5,97.5]).tolist(),selected_f1_ci95=np.percentile(scores[1],[2.5,97.5]).tolist(),f1_difference_ci95=np.percentile(scores[1]-scores[0],[2.5,97.5]).tolist())
(O/'uncertainty.json').write_text(json.dumps(ci,indent=2))
with (O/'all_comparisons.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print(json.dumps(ci,indent=2))
