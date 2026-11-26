"""Audit split IDs, exact image duplicates and normalized person-label statistics."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
root=ROOT/'COCO2017_person_80_10_10'
manifest=json.loads((root/'dataset_summary.json').read_text())
for a,b in [('train','val'),('train','test'),('val','test')]:
    assert set(manifest[a]['image_ids']).isdisjoint(manifest[b]['image_ids'])
def stats(a):
    return dict(mean=float(np.mean(a)),median=float(np.median(a)),std=float(np.std(a)),min=float(min(a)),max=float(max(a)))
data={};seen={};duplicates=[]
for split in ['train','val','test']:
    counts=[];widths=[];heights=[];areas=[];invalid=0
    images=sorted((root/'images'/split).glob('*.jpg'))
    assert {int(p.stem) for p in images}==set(manifest[split]['image_ids'])
    for image in images:
        digest=hashlib.sha256(image.read_bytes()).hexdigest()
        if digest in seen and seen[digest][0]!=split:duplicates.append([seen[digest],[split,image.name]])
        seen[digest]=[split,image.name]
        lines=(root/'labels'/split/(image.stem+'.txt')).read_text().splitlines();counts.append(len(lines))
        for line in lines:
            c,x,y,w,h=map(float,line.split())
            invalid+=int(c!=0 or not(0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1))
            widths.append(w);heights.append(h);areas.append(w*h)
    data[split]=dict(images=len(images),boxes=sum(counts),invalid=invalid,persons_per_image=stats(counts),normalized_area=stats(areas),normalized_width=stats(widths),normalized_height=stats(heights))
data['cross_split_duplicate_image_hashes']=duplicates
out=ROOT/'evaluation/dataset_audit.json';out.write_text(json.dumps(data,indent=2));print(out)
