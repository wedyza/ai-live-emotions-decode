"""Compare exported candidate against its saved PyTorch checkpoint on validation rows."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import v2 as T
import timm
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'libs/common/src'),str(ROOT/'ai-service/src')]
from common.face_preprocessing import prepare_face,ALIGNED
from ai_service.local_emotion_model import LocalEmotionModel
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
p.add_argument('--out',type=Path)
a=p.parse_args();torch.set_num_threads(2)
ckpt=torch.load(a.run/'model.pth',map_location='cpu',weights_only=True)
model=timm.create_model(ckpt['arch'],pretrained=False,num_classes=len(ckpt['classes']))
model.load_state_dict(ckpt['state_dict']);model.eval()
local=LocalEmotionModel(str(a.run/'model.onnx'))
assert local.classes==ckpt['classes'] and local.crop_mode==ckpt['crop']
tf=T.Compose([T.Resize((local.size,local.size)),T.ToImage(),T.ToDtype(torch.float32,scale=True),T.Normalize(local.mean.tolist(),local.std.tolist())])
with (a.data/'labels.csv').open(encoding='utf-8',newline='') as f:rows=[r for r in csv.DictReader(f) if r['split']=='val' and r['label'] in local.classes and r['is_dup'].lower()=='false']
maxdiff=0.;same=0
for row in rows:
 with Image.open(a.data/row['file']) as image:rgb=np.asarray(image.convert('RGB')).copy()
 points=json.loads(row['landmarks']) if local.crop_mode==ALIGNED else None
 face=prepare_face(rgb,tuple(int(row[k]) for k in ['x','y','width','height']),points,local.crop_mode)
 x=tf(Image.fromarray(face)).unsqueeze(0)
 with torch.no_grad():reference=model(x).numpy()[0]
 label,actual=local.predict_emotions(face,logits=True)
 np.testing.assert_allclose(actual,reference,rtol=1e-3,atol=1e-4)
 same+=int(reference.argmax()==actual.argmax());maxdiff=max(maxdiff,float(np.abs(actual-reference).max()))
assert rows and same==len(rows),'Top-class mismatch or empty validation'
report=dict(validation_images=len(rows),matching_classes=same,max_absolute_difference=maxdiff,preprocessing=local.crop_mode,rtol=1e-3,atol=1e-4)
(a.out or a.run/'export_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
