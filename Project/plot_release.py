"""Plot the four final validation curves from the archived threshold scores."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
fig,ax=plt.subplots(figsize=(7.6,3.25))
for folder,name,label in [
    ('capacity_benchmark','yolo11m_640','YOLO11m 640'),
    ('high_resolution_benchmark','yolo11m_960','YOLO11m 960'),
    ('large_benchmark','yolo11l_640','YOLO11l 640'),
    ('yolo26_benchmark','yolo26m_640','YOLO26m 640')]:
    record=json.loads((root/folder/f'val_{name}.json').read_text())
    rows=record['operating_points'];chosen=record['chosen']
    line,=ax.plot([x['confidence'] for x in rows],[100*x['f1'] for x in rows],label=label)
    ax.scatter(chosen['confidence'],100*chosen['f1'],s=22,color=line.get_color())
ax.set(xlabel='Detection confidence',ylabel='Validation F1 (%)')
ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.18);fig.tight_layout()
fig.savefig(root/'evaluation/figures/release_search.png',dpi=180)
