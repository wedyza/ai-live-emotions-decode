import os
os.environ['ORT_DISABLE_TELEMETRY']='1'
from pathlib import Path
import sys,json,io
import numpy as np,torch,pyarrow.parquet as pq
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
P=Path(os.environ.get('EMO_VARIANT2_DIR',ROOT/'datasets/variant2_experiment')).resolve();P.mkdir(parents=True,exist_ok=True)
OLD=Path(os.environ.get('EMO_BENCHMARK_DIR',ROOT/'datasets/fer2013_benchmark')).resolve()
V1=Path(os.environ.get('EMO_CONSENSUS_DIR',ROOT/'datasets/ferplus_consensus')).resolve()
sys.path[:0]=[str(ROOT/'libs/common/src'),str(ROOT/'cv-service/src'),str(ROOT/'ai-service/src'),str(ROOT/'tools')]
from common.landmark_features import landmark_features,flip_geometry
from common.face_preprocessing import prepare_face,ALIGNED
from model.yunet import get_yunet
from cv_service.detector import detect_captures
from ai_service.fusion_emotion_model import ReliableFusionModel
from variant2_arch import LandmarkFusionHead
torch.set_num_threads(2);m=ReliableFusionModel(P/'candidate');state=torch.load(P/'candidate/fusion_state.pth',map_location='cpu',weights_only=True);head=LandmarkFusionHead(state['image_weight'],state['image_bias'],state['geometry_mean'],state['geometry_std']);head.load_state_dict(state);head.eval();cache=np.load(P/'cache.npz');features=cache['features'];lookup={str(h):i for i,h in enumerate(cache['hashes'])};rows=pq.read_table(next(OLD.glob('publicTest-*.parquet'))).to_pylist();protocol=json.loads((P/'protocol.json').read_text());det=get_yunet();maxdiff=0.;checked=0;geometry_effect=0.
for r in protocol['selection']:
 if r['hash'] not in lookup:continue
 im=Image.open(io.BytesIO(rows[r['row']]['image']['bytes'])).convert('RGB').resize((256,256),Image.Resampling.BILINEAR);b=io.BytesIO();im.save(b,'JPEG',quality=95);jpeg=b.getvalue();rgb=np.asarray(Image.open(io.BytesIO(jpeg)).convert('RGB')).copy();fs=detect_captures(jpeg,det);f=max(fs,key=lambda f:f.bbox.width*f.bbox.height);box=f.bbox;face=prepare_face(rgb,(box.x,box.y,box.width,box.height),f.landmarks.points,ALIGNED);g=landmark_features(f.landmarks.points);gs=np.array([g,flip_geometry(g)])
 with torch.no_grad():expected=head(torch.tensor(features[lookup[r['hash']]]),torch.tensor(gs)).numpy()
 for i,view in enumerate([face,np.ascontiguousarray(face[:,::-1])]):
  actual=m.model.predict_emotions(view,gs[i],logits=True)[1];np.testing.assert_allclose(actual,expected[i],rtol=1e-3,atol=1e-4);maxdiff=max(maxdiff,float(np.abs(actual-expected[i]).max()))
 decision=m.predict_decision(face,g);reference=m._decision_from_logits(expected[0],expected[1]);assert decision.label==reference.label and decision.accepted==reference.accepted;np.testing.assert_allclose(decision.scores,reference.scores,rtol=1e-3,atol=1e-4)
 # Changing only coordinates must change logits: the second input is not decorative.
 changed=g.copy();changed[5]+=.1;changed[7]+=.1;changed[9]+=.1
 a=m.model.predict_emotions(face,g,logits=True)[1];c=m.model.predict_emotions(face,changed,logits=True)[1];geometry_effect=max(geometry_effect,float(np.abs(a-c).max()));checked+=1
 if checked>=20:break
assert geometry_effect>1e-6
try:m.predict_decision(face)
except ValueError:pass
else:raise AssertionError('Missing geometry unexpectedly accepted')
report={'checked_images':checked,'checked_views':checked*2,'matching_decisions':checked,'max_absolute_difference':maxdiff,'coordinates_only_logit_change_max':geometry_effect,'onnx_inputs':{i.name:i.shape for i in m.model.session.get_inputs()},'missing_geometry_rejected':True};(P/'export_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
