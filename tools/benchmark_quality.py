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
p=argparse.ArgumentParser();p.add_argument('phase',choices=['public','private']);a=p.parse_args()
if a.phase=='public':
 rows,y,hashes=read('publicTest');unique=np.unique(hashes,return_index=True)[1];unique.sort();rows=[rows[i] for i in unique];y=y[unique];hashes=hashes[unique]
 sel,rest=train_test_split(np.arange(len(y)),test_size=.6,stratify=y,random_state=4242)
 cal,policy=train_test_split(rest,test_size=.5,stratify=y[rest],random_state=4243)
 out={}
 if (P/'public_logits.npz').exists():
  cached=np.load(P/'public_logits.npz');assert np.array_equal(cached['hashes'],hashes) and np.array_equal(cached['y'],y)
  old_hashes=json.loads((P/'public_model_hashes.json').read_text()) if (P/'public_model_hashes.json').exists() else {}
  out={k:cached[k] for k in candidates if k in cached.files and old_hashes.get(k)==hashlib.sha256(paths[candidates[k][0]].read_bytes()).hexdigest()}
 missing=[k for k in candidates if k not in out]
 if missing:
  fresh,_=infer(rows,missing);out.update(fresh)
 np.savez_compressed(P/'public_logits.npz',y=y,hashes=hashes,selection=sel,calibration=cal,policy=policy,**out)
 (P/'public_model_hashes.json').write_text(json.dumps({k:hashlib.sha256(paths[n].read_bytes()).hexdigest() for k,(n,_) in candidates.items()},indent=2))
 scores={}
 for k,z in out.items():
  valid=np.isfinite(z[sel]).all(1);pred=np.nan_to_num(z[sel],nan=-1e9).argmax(1)
  m=metrics(y[sel],pred,valid);m['macro_f1_all']=float(f1_score(y[sel],np.where(valid,pred,-1),labels=[CLASSES.index(c) for c in FER],average='macro',zero_division=0));scores[k]=m
 chosen=max(scores,key=lambda k:scores[k]['macro_f1_all']);print('CHOSEN',chosen,scores,flush=True)
 # Independent calibration/policy subsets: selected model only with horizontal-flip consistency.
 extra,flips=infer([rows[i] for i in np.r_[cal,policy]],[chosen],True)
 z=extra[chosen];f=flips[chosen];truth=y[np.r_[cal,policy]];nc=len(cal);detected=np.isfinite(z).all(1)&np.isfinite(f).all(1)
 mean_logits=(z+f)/2
 def nll(T):
  ix=detected[:nc];prob=softmax(mean_logits[:nc][ix],T);return -float(np.log(np.maximum(prob[np.arange(ix.sum()),truth[:nc][ix]],1e-12)).mean())
 T=float(minimize_scalar(nll,bounds=(.25,8),method='bounded').x)
 probs=softmax(np.nan_to_num(mean_logits,nan=-1e9),T);pred=probs.argmax(1);ordered=np.sort(probs,axis=1);conf=ordered[:,-1];margin=ordered[:,-1]-ordered[:,-2];agree=z.argmax(1)==f.argmax(1)
 search=[]
 for threshold in [.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.98]:
  for minimum in [0.,.1,.2]:
   accepted=detected&agree&np.isin(pred,[CLASSES.index(c) for c in FER])&(conf>=threshold)&(margin>=minimum);mask=accepted[nc:];n=int(mask.sum());correct=int((pred[nc:][mask]==truth[nc:][mask]).sum())
   search.append({'threshold':threshold,'margin':minimum,'accepted':n,'accuracy':correct/n if n else 0,'wilson_lower':wilson(correct,n)})
 feasible=[r for r in search if r['accepted']>=100 and r['wilson_lower']>=.85]
 if feasible:best=max(feasible,key=lambda r:(r['accepted'],r['wilson_lower']))
 else:best=max([r for r in search if r['accepted']>=100],key=lambda r:(r['wilson_lower'],r['accepted']))
 decision={'version':1,'chosen':chosen,'classes':CLASSES,'supported_classes':FER,'crop':candidates[chosen][1],'temperature':T,'calibration_nll_before':nll(1.),'calibration_nll_after':nll(T),'min_score':best['threshold'],'min_margin':best['margin'],'require_flip_agreement':True,
 'target_precision':.9,'calibration_policy':best,'selection_scores':scores,'n_selection':len(sel),'n_calibration':len(cal),'n_policy':len(policy),'public_unique_images':len(y),
 'selection_seed':4242,'calibration_seed':4243,'selection_rule':'highest seven-class macro-F1 on selection subset, missing detections counted as no prediction',
 'policy_rule':'largest coverage with >=100 accepted and Wilson lower bound >=.85; otherwise max lower bound with >=100 accepted. Target is not a guarantee.',
 'model_sha256':hashlib.sha256(paths[candidates[chosen][0]].read_bytes()).hexdigest(),'test_used':False}
 (P/'frozen_policy.json').write_text(json.dumps(decision,indent=2));np.savez_compressed(P/'public_calibration_logits.npz',logits=z,flip_logits=f,truth=truth,n_cal=nc)
 print(json.dumps(decision,indent=2),flush=True)
else:
 decision=json.loads((P/'frozen_policy.json').read_text());freeze=hashlib.sha256((P/'frozen_policy.json').read_bytes()).hexdigest();chosen=decision['chosen'];assert hashlib.sha256(paths[candidates[chosen][0]].read_bytes()).hexdigest()==decision['model_sha256']
 rows,y,hashes=read('privateTest');public=np.load(P/'public_logits.npz')['hashes'];seen=set(public)
 training_file=P/'head_candidate/training.json'
 if not training_file.exists():training_file=ROOT/'models/emotion_head_candidate/training.json'
 training=json.loads(training_file.read_text())
 seen.update(training.get('selected_sha256',training['train_sha256']));keep=[]
 for i,h in enumerate(hashes):
  if h in seen:continue
  seen.add(h);keep.append(i)
 dropped=len(y)-len(keep);rows=[rows[i] for i in keep];y=y[keep];hashes=hashes[keep]
 keys=list(dict.fromkeys([chosen,'pretrained_aligned','finetuned_bbox','finetuned_aligned']));out,flips=infer(rows,keys,chosen)
 np.savez_compressed(P/'private_logits.npz',y=y,hashes=hashes,**out,**{k+'_flip':v for k,v in flips.items()})
 z=out[chosen];f=flips[chosen];detected=np.isfinite(z).all(1)&np.isfinite(f).all(1);probs=softmax(np.nan_to_num((z+f)/2,nan=-1e9),decision['temperature']);pred=probs.argmax(1);ordered=np.sort(probs,axis=1)
 accepted=detected&np.isin(pred,[CLASSES.index(c) for c in decision['supported_classes']])&(z.argmax(1)==f.argmax(1))&(ordered[:,-1]>=decision['min_score'])&((ordered[:,-1]-ordered[:,-2])>=decision['min_margin'])
 report={'frozen_policy_sha256':freeze,'n_test':len(y),'exact_duplicates_removed':dropped,'selected_model_all_detected':metrics(y,pred,detected),'selective':metrics(y,pred,accepted),'baseline':{},'per_class':{}}
 for k,z0 in out.items():report['baseline'][k]=metrics(y,np.nan_to_num(z0,nan=-1e9).argmax(1),np.isfinite(z0).all(1))
 for i,c in enumerate(CLASSES):
  mask=y==i
  if mask.any():
   item=metrics(y[mask],pred[mask],accepted[mask]);item.pop('macro_f1_answered',None);report['per_class'][c]=item
 report['confusion_matrix_detected']=confusion_matrix(y[detected],pred[detected],labels=list(range(8))).tolist()
 report['classes']=CLASSES
 n=report['selective']['answered'];k=report['selective']['correct_all'];report['selective']['wilson_lower']=wilson(k,n)
 report['selective']['wilson_upper']=1-wilson(n-k,n)
 # Paired independent image bootstrap; subject IDs unavailable, so no subject-independence claim.
 rng=np.random.default_rng(4244);correct=accepted&(pred==y);base=out['finetuned_bbox'];baseok=np.isfinite(base).all(1)&(base.argmax(1)==y);deltas=[]
 for _ in range(2000):
  ix=rng.integers(0,len(y),len(y));m=accepted[ix]
  if m.any():deltas.append(float(correct[ix][m].mean()-baseok[ix][m].mean()))
 report['accepted_same_images_accuracy_delta_vs_old_bbox']=float((pred[accepted]==y[accepted]).mean()-baseok[accepted].mean())
 report['accepted_same_images_delta_95ci']=np.quantile(deltas,[.025,.975]).tolist()
 reference=out['pretrained_aligned'];current=out[chosen];both=np.isfinite(reference).all(1)&np.isfinite(current).all(1)
 delta=(current.argmax(1)[both]==y[both]).astype(float)-(reference.argmax(1)[both]==y[both]).astype(float)
 rng=np.random.default_rng(4246);boot=[float(rng.choice(delta,len(delta),replace=True).mean()) for _ in range(2000)]
 report['head_vs_pretrained_raw_same_detected']={'n':int(both.sum()),'accuracy_delta':float(delta.mean()),'delta_95ci':np.quantile(boot,[.025,.975]).tolist()}
 report['warning']='FER2013 mirror labels may be noisy. Exact pixel duplicates removed; identities unavailable. Unknown overlap with HSEmotion pretraining. Scores do not establish mood. No eight-class evidence: no contempt ground truth.'
 (P/'private_report.json').write_text(json.dumps(report,indent=2));assert hashlib.sha256((P/'frozen_policy.json').read_bytes()).hexdigest()==freeze
 print(json.dumps(report,indent=2),flush=True)
