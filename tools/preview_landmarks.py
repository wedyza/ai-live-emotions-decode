"""Draw full-frame points and prepared faces; no GUI or matplotlib required."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from common.face_preprocessing import prepare_face, ALIGNED

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('landmark_preview.jpg'))
p.add_argument('--limit',type=int,default=20);a=p.parse_args()
if not 1<=a.limit<=100:p.error('--limit must be 1..100')
with (a.data/'labels.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
groups=defaultdict(list)
for row in rows:groups[row['label']].append(row)
chosen=[]
while len(chosen)<a.limit and any(groups.values()):
 for label in sorted(groups):
  if groups[label] and len(chosen)<a.limit:chosen.append(groups[label].pop(0))
if not chosen:raise SystemExit('No examples')
canvas=Image.new('RGB',(700,len(chosen)*260),'white');draw=ImageDraw.Draw(canvas)
for i,row in enumerate(chosen):
 with Image.open(a.data/row['file']) as image:rgb=np.asarray(image.convert('RGB')).copy()
 points=json.loads(row['landmarks']);bbox=tuple(int(row[k]) for k in ['x','y','width','height'])
 annotated=Image.fromarray(rgb);d=ImageDraw.Draw(annotated);x,y,w,h=bbox;d.rectangle((x,y,x+w,y+h),outline='red',width=2)
 for j,(u,v) in enumerate(points):d.ellipse((u-3,v-3,u+3,v+3),fill='lime');d.text((u+4,v),str(j),fill='yellow')
 aligned=Image.fromarray(prepare_face(rgb,bbox,points,ALIGNED))
 canvas.paste(ImageOps.contain(annotated,(440,224)),(0,i*260+30))
 canvas.paste(aligned,(460,i*260+30));draw.text((5,i*260+5),f'{row["label"]} | {row["split"]} | {row["id"][:45]}',fill='black')
a.out.parent.mkdir(parents=True,exist_ok=True);canvas.save(a.out);print(a.out.resolve())
