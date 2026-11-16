import os
os.environ['ORT_DISABLE_TELEMETRY']='1'
from pathlib import Path
import sys,json,io,time
import numpy as np,pyarrow.parquet as pq
from PIL import Image
import onnx,onnxruntime as ort
ort.disable_telemetry_events()
ROOT=Path(__file__).resolve().parents[1]
P=Path(os.environ.get('EMO_VARIANT2_DIR',ROOT/'datasets/variant2_experiment')).resolve();P.mkdir(parents=True,exist_ok=True)
OLD=Path(os.environ.get('EMO_BENCHMARK_DIR',ROOT/'datasets/fer2013_benchmark')).resolve()
V1=Path(os.environ.get('EMO_CONSENSUS_DIR',ROOT/'datasets/ferplus_consensus')).resolve()
sys.path[:0]=[str(ROOT/'libs/common/src'),str(ROOT/'cv-service/src')]
import cv2;cv2.setNumThreads(2)
from common.landmark_features import landmark_features,flip_geometry
from common.face_preprocessing import prepare_face,ALIGNED
from cv_service.detector import detect_captures
from model.yunet import get_yunet
protocol=json.loads((P/'protocol.json').read_text());records={r['hash']:r for v in protocol.values() for r in v};previous_features=None;lookup={}
if (V1/'features.npz').exists():
 report=json.loads((V1/'features_report.json').read_text())
 import hashlib
 if report['backbone_sha256']!=hashlib.sha256((ROOT/'models/emotion_reference/model.onnx').read_bytes()).hexdigest():raise ValueError('Feature cache belongs to different backbone')
 previous=np.load(V1/'features.npz');previous_features=previous['features'];lookup={str(h):i for i,h in enumerate(previous['hashes'])}
data={s:pq.read_table(next(OLD.glob(s+'-*.parquet'))).to_pylist() for s in ['train','publicTest']}
model=onnx.load(ROOT/'models/emotion_reference/model.onnx');embedding=model.graph.node[-1].input[0];model.graph.output.append(onnx.helper.make_tensor_value_info(embedding,onnx.TensorProto.FLOAT,[1,1280]));onnx.save(model,P/'embedding.onnx');options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1;session=ort.InferenceSession(str(P/'embedding.onnx'),sess_options=options,providers=['CPUExecutionProvider']);detector=get_yunet();mean=np.array([.485,.456,.406],np.float32);std=np.array([.229,.224,.225],np.float32);keys=[];features=[];geometry=[];failed=[];start=time.perf_counter();new_count=0
for number,(h,r) in enumerate(records.items()):
 im=Image.open(io.BytesIO(data[r['split']][r['row']]['image']['bytes'])).convert('RGB').resize((256,256),Image.Resampling.BILINEAR);b=io.BytesIO();im.save(b,'JPEG',quality=95);jpeg=b.getvalue();rgb=np.asarray(Image.open(io.BytesIO(jpeg)).convert('RGB')).copy();faces=detect_captures(jpeg,detector)
 if not faces:failed.append({'hash':h,'reason':'no_face'});continue
 f=max(faces,key=lambda f:f.bbox.width*f.bbox.height);box=f.bbox
 try:g=landmark_features(f.landmarks.points);face=prepare_face(rgb,(box.x,box.y,box.width,box.height),f.landmarks.points,ALIGNED)
 except ValueError as e:failed.append({'hash':h,'reason':str(e)});continue
 if h in lookup:z=previous_features[lookup[h]]
 else:
  z=[]
  for v in [face,np.ascontiguousarray(face[:,::-1])]:
   image=Image.fromarray(v).resize((224,224),Image.Resampling.BILINEAR);a=(np.asarray(image,np.float32)/255-mean)/std;a=np.ascontiguousarray(a.transpose(2,0,1)[None]);z.append(session.run([embedding],{'input':a})[0][0])
  z=np.array(z);new_count+=1
 keys.append(h);features.append(z);geometry.append([g,flip_geometry(g)])
 if number%300==0:print(number,'/',len(records),'seconds',round(time.perf_counter()-start,1),flush=True)
np.savez_compressed(P/'cache.npz',hashes=keys,features=np.stack(features),geometry=np.array(geometry,dtype=np.float32))
report={'required':len(records),'usable':len(keys),'fresh_feature_pairs':new_count,'failed':failed,'roles_usable':{k:sum(r['hash'] in set(keys) for r in v) for k,v in protocol.items()},'coordinates':'actual YuNet five points, normalized BEFORE image alignment, not canonical template'};(P/'cache_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
