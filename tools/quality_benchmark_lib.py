"""PublicTest selection/calibration, then frozen PrivateTest assessment (FER2013 mirror)."""
import os
os.environ["ORT_DISABLE_TELEMETRY"]="1"
import argparse,io,json,hashlib,time,sys
from pathlib import Path
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1]
P=Path(os.environ.get('EMO_BENCHMARK_DIR',ROOT/'datasets/fer2013_benchmark')).resolve()
sys.path[:0]=[str(ROOT/'libs/common/src'),str(ROOT/'ai-service/src'),str(ROOT/'cv-service/src')]
import cv2;cv2.setNumThreads(2)
import onnxruntime as ort
ort.disable_telemetry_events()
OriginalSession=ort.InferenceSession
def session(*args,**kwargs):
 options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1;kwargs['sess_options']=options;return OriginalSession(*args,**kwargs)
ort.InferenceSession=session
from model.yunet import YuNet
from cv_service.detector import detect_captures
from common.face_preprocessing import prepare_face,BBOX,ALIGNED
from ai_service.local_emotion_model import LocalEmotionModel
CLASSES=['anger','contempt','disgust','fear','happiness','neutral','sadness','surprise'];FER=['anger','disgust','fear','happiness','sadness','surprise','neutral']
paths={'pretrained':ROOT/'models/emotion_reference/model.onnx','finetuned_bbox':ROOT/'models/emotion_experiment/model.onnx','finetuned_aligned':ROOT/'models/emotion_aligned/model.onnx','head_candidate':(P/'head_candidate/model.onnx' if (P/'head_candidate/model.onnx').exists() else ROOT/'models/emotion_head_candidate/model.onnx')}
candidates={'pretrained_bbox':('pretrained',BBOX),'pretrained_aligned':('pretrained',ALIGNED),'finetuned_bbox':('finetuned_bbox',BBOX),'finetuned_aligned':('finetuned_aligned',ALIGNED),'head_candidate':('head_candidate',ALIGNED)}
detector=YuNet(str(ROOT/'cv-service/src/model/weights/face_detection_yunet_2023mar.onnx'))
models={}
def model(name):
 if name not in models:models[name]=LocalEmotionModel(str(paths[name]))
 return models[name]
def read(split):
 file=next(P.glob(split+'-*.parquet'))
 sources=json.loads((P/'benchmark_sources.json').read_text());expected=next(r['sha256'] for r in sources['files'] if r['file']==file.name)
 if hashlib.sha256(file.read_bytes()).hexdigest()!=expected:raise ValueError('Benchmark file checksum mismatch')
 rows=pq.read_table(file).to_pylist();y=[];hashes=[]
 for r in rows:
  with Image.open(io.BytesIO(r['image']['bytes'])) as im:a=np.asarray(im.convert('RGB'))
  hashes.append(hashlib.sha256(a.tobytes()).hexdigest());y.append(CLASSES.index(FER[r['label']]))
 return rows,np.array(y),np.array(hashes)
def image_face(row,mode):
 with Image.open(io.BytesIO(row['image']['bytes'])) as im:im=im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
 b=io.BytesIO();im.save(b,'JPEG',quality=95);rgb=np.asarray(Image.open(io.BytesIO(b.getvalue())).convert('RGB')).copy();faces=detect_captures(b.getvalue(),detector)
 if not faces:return None
 f=max(faces,key=lambda f:f.bbox.width*f.bbox.height);box=f.bbox
 return rgb,(box.x,box.y,box.width,box.height),f.landmarks.points
def softmax(a,T=1):
 z=a/T;z=z-np.max(z,axis=-1,keepdims=True);e=np.exp(z);return e/e.sum(axis=-1,keepdims=True)
def metrics(y,p,valid=None):
 valid=np.ones(len(y),bool) if valid is None else valid
 ok=p[valid]==y[valid]
 return {'n':int(len(y)),'answered':int(valid.sum()),'coverage':float(valid.mean()),'accuracy_answered':float(ok.mean()) if valid.any() else None,'correct_all':int(ok.sum()),'wrong_all':int(valid.sum()-ok.sum()),'correct_fraction_all':float(ok.sum()/len(y)),
 'macro_f1_answered':float(f1_score(y[valid],p[valid],labels=[CLASSES.index(c) for c in FER],average='macro',zero_division=0)) if valid.any() else None}
def wilson(k,n):
 if not n:return 0
 z=1.96;v=k/n;return float((v+z*z/(2*n)-z*np.sqrt(v*(1-v)/n+z*z/(4*n*n)))/(1+z*z/n))
def infer(rows,keys,flip=False):
 output={k:np.full((len(rows),8),np.nan,np.float32) for k in keys};flips={k:np.full((len(rows),8),np.nan,np.float32) for k in keys}
 start=time.perf_counter()
 for i,r in enumerate(rows):
  data=image_face(r,BBOX)
  if data:
   rgb,box,points=data
   for key in keys:
    name,mode=candidates[key]
    try:face=prepare_face(rgb,box,points,mode)
    except ValueError:continue
    output[key][i]=model(name).predict_emotions(face,logits=True)[1]
    if flip is True or (isinstance(flip,str) and key==flip):flips[key][i]=model(name).predict_emotions(np.ascontiguousarray(face[:,::-1]),logits=True)[1]
  if i%200==0:print(i,'/',len(rows),'seconds',round(time.perf_counter()-start,1),flush=True)
 return output,flips
