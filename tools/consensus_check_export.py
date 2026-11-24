import os
os.environ['ORT_DISABLE_TELEMETRY']='1'
from pathlib import Path
import json,sys,io
import numpy as np
import torch,pyarrow.parquet as pq
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
P=Path(os.environ.get('EMO_CONSENSUS_DIR',ROOT/'datasets/ferplus_consensus')).resolve()
OLD=Path(os.environ.get('EMO_BENCHMARK_DIR',ROOT/'datasets/fer2013_benchmark')).resolve()
sys.path[:0]=[str(ROOT/'libs/common/src'),str(ROOT/'cv-service/src'),str(ROOT/'ai-service/src')]
from ai_service.reliable_emotion_model import ReliableEmotionModel
from common.face_preprocessing import prepare_face,ALIGNED
from cv_service.detector import detect_captures
from model.yunet import get_yunet
m=ReliableEmotionModel(P/'candidate');h=torch.load(P/'candidate/head_state.pth',map_location='cpu',weights_only=True);features=np.load(P/'features.npz');lookup={str(v):i for i,v in enumerate(features['hashes'])};feature_matrix=features['features'];protocol=json.loads((P/'protocol.json').read_text());policy=json.loads((P/'candidate/policy.json').read_text());rows=pq.read_table(next(OLD.glob('publicTest-*.parquet'))).to_pylist();detector=get_yunet();maxdiff=0.;checked=0
for r in protocol['selection']:
 if r['hash'] not in lookup:continue
 im=Image.open(io.BytesIO(rows[r['row']]['image']['bytes'])).convert('RGB').resize((256,256),Image.Resampling.BILINEAR);b=io.BytesIO();im.save(b,'JPEG',quality=95);jpeg=b.getvalue();rgb=np.asarray(Image.open(io.BytesIO(jpeg)).convert('RGB')).copy();faces=detect_captures(jpeg,detector);f=max(faces,key=lambda f:f.bbox.width*f.bbox.height);box=f.bbox;face=prepare_face(rgb,(box.x,box.y,box.width,box.height),f.landmarks.points,ALIGNED)
 with torch.no_grad():reference=torch.nn.functional.linear(torch.tensor(feature_matrix[lookup[r['hash']]]),h['weight'],h['bias']).numpy()
 for i,view in enumerate([face,np.ascontiguousarray(face[:,::-1])]):
  actual=m.model.predict_emotions(view,logits=True)[1];np.testing.assert_allclose(actual,reference[i],rtol=1e-3,atol=1e-4);maxdiff=max(maxdiff,float(np.abs(actual-reference[i]).max()))
 a=reference.mean(0).astype(float)/policy['temperature'];v=np.exp(a-a.max());v/=v.sum();label=policy['classes'][int(v.argmax())];cut=policy['class_thresholds'][label];expected=bool(cut is not None and reference[0].argmax()==reference[1].argmax() and v.max()>=cut)
 d=m.predict_decision(face);assert d.label==label and d.accepted==expected;np.testing.assert_allclose(d.scores,v,rtol=1e-3,atol=1e-4);checked+=1
 if checked>=20:break
result={'checked_images':checked,'checked_views':checked*2,'decisions_matching':checked,'max_absolute_difference':maxdiff,'rtol':1e-3,'atol':1e-4};(P/'export_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
