"""Evaluate on transformed publisher annotations; development clip, not an unseen test."""
import argparse,csv,json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parent

def iou(a,b):
    x=max(a[0],b[0]);y=max(a[1],b[1]);r=min(a[2],b[2]);d=min(a[3],b[3]);inter=max(0,r-x)*max(0,d-y)
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0

def match_matrix(values,threshold):
    if values.size==0:return []
    valid=values>=threshold
    cost=-(valid.astype(float)*(min(values.shape)+1)+values)
    a,b=linear_sum_assignment(cost)
    return [(int(i),int(j)) for i,j in zip(a,b) if valid[i,j]]

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--annotations',type=Path,default=ROOT/'evaluation/wk2gt.xml');p.add_argument('--start-frame',type=int,default=80);a=p.parse_args()
    info=json.loads((a.run/'video_info.json').read_text());nframes=info['frames'];fps=info['fps']
    zone=json.loads((a.run/'zone_used.json').read_text());poly=np.asarray(zone['points'],np.int32)
    gt={};events=[];states={}
    source=a.annotations
    for f in ET.parse(source).getroot().findall('frame'):
        n=int(f.attrib['number'])-a.start_frame
        if not 0<=n<nframes:continue
        gt[n]=[]
        for o in f.findall('./objectlist/object'):
            pid=int(o.attrib['id']);b=o.find('box').attrib;xc,yc,w,h=[float(b[k]) for k in ('xc','yc','w','h')]
            box=[288-(yc+h/2),xc-w/2,288-(yc-h/2),xc+w/2];gt[n].append((pid,box))
            inside=cv2.pointPolygonTest(poly,((box[0]+box[2])/2,box[3]),False)>=0
            if pid not in states:
                states[pid]=[inside,0]
                if inside:events.append(dict(person_id=pid,frame=n,time=n/fps,event_type='initial_presence'))
            else:
                s=states[pid]
                if inside:
                    if not s[0]:events.append(dict(person_id=pid,frame=n,time=n/fps,event_type='entry'))
                    s[:]=[True,0]
                else:
                    s[1]+=1
                    if s[1]>=15:s[0]=False
    pred={}
    for r in csv.DictReader((a.run/'tracks.csv').open()):pred.setdefault(int(r['frame']),[]).append((int(r['track_id']),[float(r[k]) for k in ('x1','y1','x2','y2')]))
    tp=fp=fn=switches=0;last={};mapping={}
    for frame in range(nframes):
        g=gt.get(frame,[]);d=pred.get(frame,[])
        mat=np.array([[iou(x[1],y[1]) for y in d] for x in g]).reshape(len(g),len(d))
        pairs=match_matrix(mat,.5);tp+=len(pairs);fp+=len(d)-len(pairs);fn+=len(g)-len(pairs)
        for i,j in pairs:
            gid,tid=g[i][0],d[j][0];mapping[(frame,tid)]=gid
            if gid in last and last[gid]!=tid:switches+=1
            last[gid]=tid
    predicted=[]
    for r in csv.DictReader((a.run/'events.csv').open()):
        mm,ss=map(float,r['Elapsed video time'].split(':'));t=float(r['Time seconds']) if r.get('Time seconds') else mm*60+ss;tid=int(r['Track ID'])
        nearby=[(abs(fr/fps-t),gid) for (fr,track),gid in mapping.items() if track==tid and abs(fr/fps-t)<=.12]
        gid=min(nearby)[1] if nearby else None
        predicted.append(dict(time=t,event_type=r['Event type'],track_id=tid,person_id=gid))
    # Lock tolerance to 0.5 seconds; match identity and event type as well as time.
    quality=np.zeros((len(events),len(predicted)))
    for i,g in enumerate(events):
        for j,d in enumerate(predicted):
            delta=abs(g['time']-d['time'])
            if g['person_id']==d['person_id'] and g['event_type']==d['event_type'] and delta<=.5:quality[i,j]=1-delta
    pairs=match_matrix(quality,.5);etp=len(pairs);efp=len(predicted)-etp;efn=len(events)-etp
    result=dict(scope=f"{info['file']}; publisher boxes rotated 90 degrees clockwise; source start frame {a.start_frame}.",annotation_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),frames=nframes,person_instances=sum(map(len,gt.values())),ground_truth_events=events,predicted_events=predicted,detection=dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0),events=dict(tp=etp,fp=efp,fn=efn,precision=etp/(etp+efp) if etp+efp else None,recall=etp/(etp+efn) if etp+efn else None,tolerance_seconds=.5,signed_delays_seconds=[predicted[j]['time']-events[i]['time'] for i,j in pairs]),matched_identity_switches=switches,tracking_note='Switch count follows consecutive successful matches per GT identity, including gaps; not standardized MOTA/IDF1.')
    (a.run/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
