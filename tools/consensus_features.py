import os
os.environ['ORT_DISABLE_TELEMETRY']='1'
import sys,json,io,time,hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import onnx,onnxruntime as ort
ort.disable_telemetry_events()
ROOT=Path(__file__).resolve().parents[1]
P=Path(os.environ.get('EMO_CONSENSUS_DIR',ROOT/'datasets/ferplus_consensus')).resolve()
OLD=Path(os.environ.get('EMO_BENCHMARK_DIR',ROOT/'datasets/fer2013_benchmark')).resolve()
sys.path[:0]=[str(ROOT/'libs/common/src'),str(ROOT/'cv-service/src')]
import cv2;cv2.setNumThreads(2)
from cv_service.detector import detect_captures
from model.yunet import get_yunet
from common.face_preprocessing import prepare_face,ALIGNED
protocol=json.loads((P/'protocol.json').read_text());records={r['hash']:r for v in protocol.values() for r in v};cache={};failed=[]
reused=len(cache);m=onnx.load(ROOT/'models/emotion_reference/model.onnx');assert m.graph.node[-1].op_type=='Gemm'
embedding=m.graph.node[-1].input[0];m.graph.output.append(onnx.helper.make_tensor_value_info(embedding,onnx.TensorProto.FLOAT,[1,1280]));onnx.save(m,P/'embedding.onnx')
options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
session=ort.InferenceSession(str(P/'embedding.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
data={name:pq.read_table(next(OLD.glob(name+'-*.parquet'))).to_pylist() for name in ['train','publicTest']};detector=get_yunet();start=time.perf_counter();max_reuse_difference=0.;compared=0
mean=np.array([.485,.456,.406],np.float32);std=np.array([.229,.224,.225],np.float32)
for number,(h,row) in enumerate(records.items()):
 reused_sample=h in cache
 if reused_sample and compared>=20:continue
 raw=data[row['split']][row['row']]
 im=Image.open(io.BytesIO(raw['image']['bytes'])).convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
 b=io.BytesIO();im.save(b,'JPEG',quality=95);jpeg=b.getvalue();rgb=np.asarray(Image.open(io.BytesIO(jpeg)).convert('RGB')).copy();faces=detect_captures(jpeg,detector)
 if not faces:failed.append(h);continue
 f=max(faces,key=lambda f:f.bbox.width*f.bbox.height);box=f.bbox
 try:face=prepare_face(rgb,(box.x,box.y,box.width,box.height),f.landmarks.points,ALIGNED)
 except ValueError:failed.append(h);continue
 values=[]
 for view in [face,np.ascontiguousarray(face[:,::-1])]:
  im=Image.fromarray(view).resize((224,224),Image.Resampling.BILINEAR);a=(np.asarray(im,np.float32)/255-mean)/std;inp=np.ascontiguousarray(a.transpose(2,0,1)[None]);values.append(session.run([embedding],{'input':inp})[0][0])
 values=np.array(values)
 if reused_sample:
  np.testing.assert_allclose(values,cache[h],rtol=1e-3,atol=1e-4);max_reuse_difference=max(max_reuse_difference,float(np.abs(values-cache[h]).max()));compared+=1
 cache[h]=values
 if number%200==0:print(number,'/',len(records),'computed',len(cache)-reused,'s',round(time.perf_counter()-start,1),flush=True)
keys=sorted(cache);np.savez_compressed(P/'features.npz',hashes=keys,features=np.stack([cache[h] for h in keys]))
report={'required':len(records),'available':len(cache),'reused':reused,'new_features':len(cache)-reused,'failed':failed,'onnx_vs_torch_reuse_checked':compared,'reuse_max_difference':max_reuse_difference,'backbone_sha256':hashlib.sha256((ROOT/'models/emotion_reference/model.onnx').read_bytes()).hexdigest()}
(P/'features_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
