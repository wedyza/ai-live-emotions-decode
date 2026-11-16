import os
os.environ['ORT_DISABLE_TELEMETRY']='1'
import sys,json,copy,hashlib
from pathlib import Path
import numpy as np
import torch,onnx
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parents[1]
P=Path(os.environ.get('EMO_VARIANT2_DIR',ROOT/'datasets/variant2_experiment')).resolve();P.mkdir(parents=True,exist_ok=True)
OLD=Path(os.environ.get('EMO_BENCHMARK_DIR',ROOT/'datasets/fer2013_benchmark')).resolve()
V1=Path(os.environ.get('EMO_CONSENSUS_DIR',ROOT/'datasets/ferplus_consensus')).resolve()
sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'libs/common/src'))
from variant2_arch import LandmarkFusionHead,PhotoAndLandmarks
from training_model import load_hsemotion
from common.landmark_features import GEOMETRY_VERSION
C=['anger','contempt','disgust','fear','happiness','neutral','sadness','surprise'];torch.set_num_threads(2)
roles=json.loads((P/'protocol.json').read_text());data=np.load(P/'cache.npz');features=data['features'];geometry=data['geometry'];lookup={str(h):i for i,h in enumerate(data['hashes'])}
def role(name):
 rows=roles[name];mask=np.array([r['hash'] in lookup for r in rows]);kept=[r for r in rows if r['hash'] in lookup];ids=[lookup[r['hash']] for r in kept];votes=np.array([r['votes'] for r in kept],float);votes/=votes.sum(1,keepdims=True)
 return torch.tensor(features[ids]),torch.tensor(geometry[ids]),np.array([r['majority'] for r in rows]),mask,torch.tensor(votes,dtype=torch.float32)
x,g,y,valid,targets=role('train');v,vg,vy,vmask,_=role('selection');image=onnx.load(ROOT/'models/emotion_consensus/model.onnx');params={a.name:torch.tensor(onnx.numpy_helper.to_array(a).copy()) for a in image.graph.initializer if a.name.startswith('classifier.')};W=params['classifier.weight'];B=params['classifier.bias'];mean=g.flatten(0,1).mean(0);std=g.flatten(0,1).std(0).clamp_min(.05)
flat=x.flatten(0,1);geo=g.flatten(0,1);soft=targets.repeat_interleave(2,0);majority=soft.argmax(1);freq=torch.bincount(majority,minlength=8).float();sampling=1/torch.clamp(freq[majority],min=1).sqrt();supported=[i for i in range(8) if (vy==i).sum()>=20]
def score(logits):
 pred=np.full(len(vy),-1);pred[vmask]=logits[:,0].argmax(1)
 return float(f1_score(vy,pred,labels=supported,average='macro',zero_division=0))
with torch.no_grad():baseline=score(torch.nn.functional.linear(v,W,B).numpy())
history=[];winners={}
for kind in ['actual','constant_control']:
 best=None
 for reg in [.0,.1,1.]:
  torch.manual_seed(7701);head=LandmarkFusionHead(W,B,mean,std);optim=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01);generator=torch.Generator().manual_seed(7702)
  input_geo=geo if kind=='actual' else mean.expand_as(geo)
  for step in range(1,501):
   ids=torch.multinomial(sampling,256,replacement=True,generator=generator);z=head(flat[ids],input_geo[ids]);ce=-(soft[ids]*torch.log_softmax(z,dim=1)).sum(1).mean();delta=head.geometry_delta(input_geo[ids]);loss=ce+reg*delta.square().mean();optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optim.step()
   if step in [200,500]:
    head.eval()
    with torch.no_grad():z=head(v,vg if kind=='actual' else mean.expand_as(vg)).numpy()
    value=score(z);record={'kind':kind,'regularization':reg,'step':step,'selection_macro_f1':value};history.append(record);print(record,flush=True)
    if best is None or value>best['score']:best={'score':value,'reg':reg,'step':step,'state':copy.deepcopy(head.state_dict())}
    head.train()
 winners[kind]=best
head=LandmarkFusionHead(W,B,mean,std);head.load_state_dict(winners['actual']['state']);head.eval();control=LandmarkFusionHead(W,B,mean,std);control.load_state_dict(winners['constant_control']['state']);control.eval()
assert torch.equal(head.image_weight,W) and torch.equal(head.image_bias,B);assert head.geometry_weight.abs().max()>1e-8
out=P/'candidate';out.mkdir(exist_ok=True);torch.save(head.state_dict(),out/'fusion_state.pth');torch.save(control.state_dict(),out/'constant_control.pth')
# Calibrate only on the same pre-existing development roles, never on the fresh test.
cx,cg,cy,cmask,_=role('calibration');px,pg,py,pmask,_=role('policy')
with torch.no_grad():cz=head(cx,cg).numpy().mean(1);pz=head(px,pg).numpy()
def sm(z,T):
 z=np.asarray(z,dtype=np.float64)/T;z-=z.max(-1,keepdims=True);e=np.exp(z);return e/e.sum(-1,keepdims=True)
cal_truth=cy[cmask]
def nll(T):return -float(np.log(np.maximum(sm(cz,T)[np.arange(len(cal_truth)),cal_truth],1e-12)).mean())
T=float(minimize_scalar(nll,bounds=(.25,8),method='bounded').x);probs=sm(pz.mean(1),T);pred=probs.argmax(1);agree=pz[:,0].argmax(1)==pz[:,1].argmax(1);truth=py[pmask]
def wilson(k,n):
 if not n:return 0.
 z=1.96;p=k/n;return float((p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n))
thresholds={};evidence={}
for i,c in enumerate(C):
 options=[]
 for cut in [.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.98]:
  mask=agree&(pred==i)&(probs[:,i]>=cut);n=int(mask.sum());k=int((truth[mask]==i).sum())
  if n>=20 and k/n>=.9 and wilson(k,n)>=.75:options.append({'threshold':cut,'n':n,'correct':k,'precision':k/n,'wilson_lower':wilson(k,n)})
 if options:
  best=max(options,key=lambda a:(a['n'],a['wilson_lower']));thresholds[c]=best['threshold'];evidence[c]=best
 else:thresholds[c]=None;evidence[c]={'reason':'insufficient_class_evidence'}
backbone,size,classes=load_hsemotion('enet_b0_8_best_afew',ROOT/'datasets/weights');backbone.eval();full=PhotoAndLandmarks(backbone,head).eval()
torch.onnx.export(full,(torch.zeros(1,3,224,224),mean.unsqueeze(0)),str(out/'model.onnx'),input_names=['image','landmarks'],output_names=['logits'],opset_version=18,dynamo=True,external_data=False)
meta=json.loads((ROOT/'models/emotion_consensus/model.onnx.json').read_text());meta.update({'model_variant':'variant2','geometry_norm':GEOMETRY_VERSION,'geometry_size':10,'input_names':['image','landmarks'],'geometry_branch':[10,32,16],'fusion_features':1296,'image_head_frozen':True});(out/'model.onnx.json').write_text(json.dumps(meta,indent=2))
policy={'version':2,'classes':C,'crop':meta['crop'],'model_variant':'variant2','geometry_norm':GEOMETRY_VERSION,'model_sha256':hashlib.sha256((out/'model.onnx').read_bytes()).hexdigest(),'supported_classes':[c for c in C if thresholds[c] is not None],'temperature':T,'min_score':0.,'min_margin':0.,'require_flip_agreement':True,'class_thresholds':thresholds,'class_evidence':evidence,'test_used':False};(out/'policy.json').write_text(json.dumps(policy,indent=2));(P/'frozen_policy_sha256.txt').write_text(hashlib.sha256((out/'policy.json').read_bytes()).hexdigest())
report={'variant1_selection_macro_f1':baseline,'variant2_selected':{k:v for k,v in winners['actual'].items() if k!='state'},'constant_control_selected':{k:v for k,v in winners['constant_control'].items() if k!='state'},'history':history,'train_images':len(x),'geometry_weight_max_abs':float(head.geometry_weight.detach().abs().max()),'image_head_unchanged':True,'backbone_frozen':True,'geometry_norm':GEOMETRY_VERSION,'test_used':False,'nll_before':nll(1),'nll_after':nll(T),'policy_sha256':hashlib.sha256((out/'policy.json').read_bytes()).hexdigest()};(P/'training_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
