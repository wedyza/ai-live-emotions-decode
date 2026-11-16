"""Check a local photo using the reliable pipeline; no upload or dataset needed."""
import argparse,io,json,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
os.environ['ORT_DISABLE_TELEMETRY']='1'
os.environ['AI_EMOTION_MODEL']='reliable'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode',choices=['reliable','expanded'],default='reliable')
p.add_argument('image',type=Path);p.add_argument('--output',type=Path)
a=p.parse_args()
os.environ['AI_RELIABILITY_PATH']='models/emotion_consensus' if a.mode=='expanded' else 'models/emotion_reliable'
if a.output and a.output.resolve()==a.image.resolve():p.error('Output must differ from original photo')
from ai_service.analyzer import _recognizer
from common.face_preprocessing import prepare_face
from cv_service.detector import detect_captures
from model.yunet import get_yunet
with Image.open(a.image) as im:im=im.convert('RGB')
b=io.BytesIO();im.save(b,'JPEG',quality=95);jpeg=b.getvalue()
rgb=np.asarray(Image.open(io.BytesIO(jpeg)).convert('RGB')).copy()
faces=detect_captures(jpeg,get_yunet());recognizer=_recognizer();records=[];draw=ImageDraw.Draw(im)
for i,f in enumerate(faces,1):
 box=f.bbox;coords=(box.x,box.y,box.width,box.height)
 try:
  crop=prepare_face(rgb,coords,f.landmarks.points,recognizer.crop_mode)
  d=recognizer.predict_decision(crop)
  row={'face':i,'accepted':d.accepted,'expression':d.label if d.accepted else None,'reason':d.reason,'model_score':float(d.scores.max()) if d.accepted else None,'bbox':list(coords),'landmarks':f.landmarks.points}
 except ValueError:
  row={'face':i,'accepted':False,'expression':None,'reason':'invalid_face_geometry','bbox':list(coords)}
 records.append(row)
 color='lime' if row['accepted'] else 'orange';draw.rectangle([box.x,box.y,box.x+box.width,box.y+box.height],outline=color,width=2)
 draw.text((box.x,max(0,box.y-12)),row['expression'] or 'uncertain',fill=color)
 if f.landmarks:
  for x,y in f.landmarks.points:draw.ellipse([x-2,y-2,x+2,y+2],fill='red')
if a.output:
 a.output.parent.mkdir(parents=True,exist_ok=True);im.save(a.output)
print(json.dumps({'faces_detected':len(faces),'results':records,'note':'Expressions are model categories, not a diagnosis of mood. uncertain means abstention.'},ensure_ascii=False,indent=2))
