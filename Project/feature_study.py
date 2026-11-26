"""Grouped cross-validation for a post-detection candidate classifier."""
from pathlib import Path
import contextlib, io, json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from pycocotools.cocoeval import COCOeval
from benchmark_detectors import ground_truth
ROOT=Path(__file__).resolve().parent
FEATURES=['confidence','relative area','log aspect ratio','vertical center']

def examples(split):
    gt=ground_truth(split)
    pred=json.loads((ROOT/'capacity_benchmark'/f'{split}_yolo11m_640_predictions.json').read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        dt=gt.loadRes(pred)
        ev=COCOeval(gt,dt,'bbox');ev.params.catIds=[1];ev.evaluate()
    rows=[]
    for item in ev.evalImgs:
        if item is None or item['aRng'] != ev.params.areaRng[0]:continue
        image=gt.imgs[item['image_id']];w,h=image['width'],image['height']
        for i,dt_id in enumerate(item['dtIds']):
            if item['dtIgnore'][0,i]:continue
            box=dt.anns[dt_id];x,y,bw,bh=box['bbox']
            rows.append([image['id'],box['score'],bw*bh/(w*h),np.log(max(bw,1e-8)/max(bh,1e-8)),(y+bh/2)/h,int(item['dtMatches'][0,i]>0)])
    a=np.asarray(rows)
    return a[:,0].astype(int),a[:,1:5],a[:,-1]

def fit(x,y,columns):
    z=x[:,columns];mean=z.mean(0);std=np.maximum(z.std(0),1e-8);z=(z-mean)/std
    def objective(theta):
        s=z@theta[:-1]+theta[-1];e=expit(s)-y
        return ((np.logaddexp(0,s)-y*s).mean()+.0005*np.sum(theta[:-1]**2),np.r_[z.T@e/len(y)+.001*theta[:-1],e.mean()])
    opt=minimize(objective,np.zeros(len(columns)+1),jac=True,method='L-BFGS-B')
    if not opt.success:raise RuntimeError(opt.message)
    return opt.x,mean,std

def rfe(x,y,k):
    cols=list(range(4))
    while len(cols)>k:
        theta,*_=fit(x,y,cols);cols.pop(int(np.argmin(np.abs(theta[:-1]))))
    return cols,fit(x,y,cols)

def predict(model,x):
    cols,(theta,mean,std)=model
    return expit((x[:,cols]-mean)/std@theta[:-1]+theta[-1])

def counts(y,s,t):
    keep=s>=t;tp=int(y[keep].sum());fp=int(keep.sum()-tp);fn=int(y.sum()-tp)
    return {'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/(2*tp+fp+fn)}
def main():
    ids,x,y=examples('val');ids_unique=np.unique(ids);np.random.default_rng(42).shuffle(ids_unique)
    folds=np.array_split(ids_unique,5);out=[]
    for k in [4,2,1]:
        scores=np.zeros(len(y))
        for group in folds:
            hold=np.isin(ids,group);scores[hold]=predict(rfe(x[~hold],y[~hold],k),x[hold])
        best=max((dict(confidence=t,**counts(y,scores,t)) for t in np.arange(.01,.91,.01)),key=lambda a:a['f1'])
        out.append(dict(size=k,validation=best))
    best=max(out,key=lambda a:(a['validation']['f1'],-a['size']))
    model=rfe(x,y,best['size']);_,xt,yt=examples('test')
    result=dict(target='Whether a supplied candidate matched a non-ignored person under original COCO score ordering at IoU 0.50. Candidate classification F1 excludes undetected ground truths.',folds='5 validation folds grouped by image, seed 42. RFE removes smallest absolute standardized logistic coefficient inside each training fold. Mean NLL with L2 coefficient 0.0005.',sizes=out,chosen_features=[FEATURES[i] for i in model[0]],threshold=best['validation']['confidence'],test=counts(yt,predict(model,xt),best['validation']['confidence']),test_candidates=len(yt),correlations={f:float(np.corrcoef(x[:,i],y)[0,1]) for i,f in enumerate(FEATURES)})
    (ROOT/'evaluation/feature_study.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
