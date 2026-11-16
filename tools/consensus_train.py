import os
os.environ['ORT_DISABLE_TELEMETRY']='1'
from pathlib import Path
import json,copy,hashlib
import numpy as np
import torch
from sklearn.metrics import f1_score,accuracy_score
from scipy.optimize import minimize_scalar
import onnx
ROOT=Path(__file__).resolve().parents[1]
P=Path(os.environ.get('EMO_CONSENSUS_DIR',ROOT/'datasets/ferplus_consensus')).resolve()
OLD=Path(os.environ.get('EMO_BENCHMARK_DIR',ROOT/'datasets/fer2013_benchmark')).resolve()
C=['anger','contempt','disgust','fear','happiness','neutral','sadness','surprise']
torch.set_num_threads(2);torch.manual_seed(5511)
protocol=json.loads((P/'protocol.json').read_text());data=np.load(P/'features.npz');lookup={str(h):i for i,h in enumerate(data['hashes'])};features=data['features']
def role(name):
 rows=protocol[name];valid=np.array([r['hash'] in lookup for r in rows]);kept=[r for r in rows if r['hash'] in lookup];x=torch.tensor(np.stack([features[lookup[r['hash']]] for r in kept]));y=np.array([r['majority'] for r in rows]);votes=np.array([r['votes'] for r in kept],float);votes/=votes.sum(1,keepdims=True)
 return rows,x,y,valid,torch.tensor(votes,dtype=torch.float32)
train,x,yt,train_valid,targets=role('train');selection,v,y,valid,_=role('selection')
state=torch.load(ROOT/'datasets/weights/enet_b0_8_best_afew.state_dict.pth',map_location='cpu',weights_only=True)
def head(state):
 h=torch.nn.Linear(1280,8);h.load_state_dict({'weight':state['classifier.weight'],'bias':state['classifier.bias']});return h
original=head(state);prior_onnx=onnx.load(ROOT/'models/emotion_head_candidate/model.onnx');prior_state={v.name:torch.tensor(onnx.numpy_helper.to_array(v).copy()) for v in prior_onnx.graph.initializer if v.name.startswith('classifier.')};prior=head(prior_state)
supported=[i for i in range(8) if int((y==i).sum())>=20]
def logits(h,values):
 with torch.no_grad():return h(values).numpy()
def score(h):
 z=logits(h,v);pred=np.full(len(y),-1);pred[valid]=z[:,0].argmax(1)
 return {'macro_f1_supported':float(f1_score(y,pred,labels=supported,average='macro',zero_division=0)),'accuracy_all':float((pred==y).mean())}
baselines={name:score(h) for name,h in [('original',original),('v1',prior)]};chosen=max(baselines,key=lambda n:baselines[n]['macro_f1_supported']);best={'score':baselines[chosen]['macro_f1_supported'],'source':chosen,'state':copy.deepcopy((original if chosen=='original' else prior).state_dict()),'step':0,'strength':None}
flat=x.flatten(0,1);soft=targets.repeat_interleave(2,dim=0);majority=soft.argmax(1);freq=torch.bincount(majority,minlength=8).float();sampling=1/torch.clamp(freq[majority],min=1).sqrt()
with torch.no_grad():teacher_prob=torch.softmax(original(flat)/2,dim=1)
history=[]
for strength in [0.,.25,1.,4.]:
 h=copy.deepcopy(original);optim=torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.01);gen=torch.Generator().manual_seed(5511)
 for step in range(1,401):
  ids=torch.multinomial(sampling,256,replacement=True,generator=gen);z=h(flat[ids]);ce=-(soft[ids]*torch.log_softmax(z,dim=1)).sum(1).mean();kl=torch.nn.functional.kl_div(torch.log_softmax(z/2,dim=1),teacher_prob[ids],reduction='batchmean')*4
  loss=ce+strength*kl;optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1.);optim.step()
  if step in [100,200,400]:
   metrics=score(h);entry={'strength':strength,'step':step,**metrics};history.append(entry);print(entry,flush=True)
   if metrics['macro_f1_supported']>best['score']:best={'score':metrics['macro_f1_supported'],'source':'ferplus_head','state':copy.deepcopy(h.state_dict()),'step':step,'strength':strength}
selected=copy.deepcopy(original);selected.load_state_dict(best['state']);selected.eval()
# Class-wise policies fitted without reading holdout labels or predictions.
_,cal,cal_y,cal_valid,_=role('calibration');cal_z=logits(selected,cal).mean(1);cal_truth=cal_y[cal_valid]
def sm(z,T):
 z=np.asarray(z,dtype=np.float64)/T;z-=z.max(-1,keepdims=True);e=np.exp(z);return e/e.sum(-1,keepdims=True)
def nll(T):return -float(np.log(np.maximum(sm(cal_z,T)[np.arange(len(cal_truth)),cal_truth],1e-12)).mean())
T=float(minimize_scalar(nll,bounds=(.25,8),method='bounded').x)
_,pol,pol_y,pol_valid,_=role('policy');pol_z=logits(selected,pol);probs=sm(pol_z.mean(1),T);pred=probs.argmax(1);agreement=pol_z[:,0].argmax(1)==pol_z[:,1].argmax(1);truth=pol_y[pol_valid]
def wilson(k,n):
 if not n:return 0.
 z=1.96;p=k/n;return float((p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n))
thresholds={};evidence={}
for i,c in enumerate(C):
 candidates=[]
 for cut in [.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.98]:
  mask=agreement&(pred==i)&(probs[:,i]>=cut);n=int(mask.sum());k=int((truth[mask]==i).sum());precision=k/n if n else 0.;lower=wilson(k,n)
  if n>=20 and precision>=.9 and lower>=.75:candidates.append({'threshold':cut,'accepted':n,'correct':k,'precision':precision,'wilson_lower':lower})
 if candidates:
  win=max(candidates,key=lambda r:(r['accepted'],r['wilson_lower']));thresholds[c]=win['threshold'];evidence[c]=win
 else:thresholds[c]=None;evidence[c]={'reason':'insufficient_class_reliability_evidence','predicted_after_agreement':int((agreement&(pred==i)).sum())}
out=P/'candidate';out.mkdir(exist_ok=True)
m=onnx.load(ROOT/'models/emotion_reference/model.onnx');weights=selected.weight.detach().numpy().astype('float32');bias=selected.bias.detach().numpy().astype('float32')
for i,initializer in enumerate(m.graph.initializer):
 if initializer.name=='classifier.weight':m.graph.initializer[i].CopyFrom(onnx.numpy_helper.from_array(weights,'classifier.weight'))
 if initializer.name=='classifier.bias':m.graph.initializer[i].CopyFrom(onnx.numpy_helper.from_array(bias,'classifier.bias'))
onnx.checker.check_model(m);onnx.save(m,out/'model.onnx')
meta=json.loads((ROOT/'models/emotion_head_candidate/model.onnx.json').read_text());meta['adaptation']='FER+ soft-vote head, frozen original backbone';(out/'model.onnx.json').write_text(json.dumps(meta,indent=2));torch.save(selected.state_dict(),out/'head_state.pth')
policy={'version':2,'classes':C,'crop':'yunet5_similarity_v1','model_sha256':hashlib.sha256((out/'model.onnx').read_bytes()).hexdigest(),'supported_classes':[c for c in C if thresholds[c] is not None],'temperature':T,'min_score':0.,'min_margin':0.,'require_flip_agreement':True,'class_thresholds':thresholds,'class_evidence':evidence,'class_target_precision':.9,'class_min_examples':20,'class_min_wilson_lower':.75,'test_used':False,'rule':'Per-class largest acceptance with >=20 policy-fit predictions, >=.90 empirical precision and Wilson lower>=.75; otherwise abstain. Not a guarantee.'}
(out/'policy.json').write_text(json.dumps(policy,indent=2));(P/'frozen_policy_sha256.txt').write_text(hashlib.sha256((out/'policy.json').read_bytes()).hexdigest())
report={'baselines_selection':baselines,'best':{k:v for k,v in best.items() if k!='state'},'selection_scored_classes':[C[i] for i in supported],'history':history,'train_selected':len(train),'train_detected':int(train_valid.sum()),'temperature_nll_before':nll(1),'temperature_nll_after':nll(T),'holdout_used':False,'policy_sha256':hashlib.sha256((out/'policy.json').read_bytes()).hexdigest()}
(P/'training_report.json').write_text(json.dumps(report,indent=2));print('FROZEN',json.dumps(report,indent=2),flush=True);print(json.dumps(policy,indent=2),flush=True)
