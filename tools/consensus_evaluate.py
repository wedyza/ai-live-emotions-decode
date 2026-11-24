import os
os.environ['ORT_DISABLE_TELEMETRY']='1'
from pathlib import Path
import json,hashlib,onnx
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parents[1]
P=Path(os.environ.get('EMO_CONSENSUS_DIR',ROOT/'datasets/ferplus_consensus')).resolve()
OLD=Path(os.environ.get('EMO_BENCHMARK_DIR',ROOT/'datasets/fer2013_benchmark')).resolve()
C=['anger','contempt','disgust','fear','happiness','neutral','sadness','surprise'];torch.set_num_threads(2)
policy=json.loads((P/'candidate/policy.json').read_text());freeze=hashlib.sha256((P/'candidate/policy.json').read_bytes()).hexdigest();assert freeze==(P/'frozen_policy_sha256.txt').read_text();assert (P/'training_report.json').exists()
roles=json.loads((P/'protocol.json').read_text());rows=roles['holdout'];d=np.load(P/'features.npz');lookup={str(h):i for i,h in enumerate(d['hashes'])};feature_matrix=d['features'];valid=np.array([r['hash'] in lookup for r in rows]);features=torch.tensor(np.stack([feature_matrix[lookup[r['hash']]] for r in rows if r['hash'] in lookup]));y=np.array([r['majority'] for r in rows])
newstate=torch.load(P/'candidate/head_state.pth',map_location='cpu',weights_only=True);prior_onnx=onnx.load(ROOT/'models/emotion_head_candidate/model.onnx');oldstate={v.name:torch.tensor(onnx.numpy_helper.to_array(v).copy()) for v in prior_onnx.graph.initializer if v.name.startswith('classifier.')};reference=torch.load(ROOT/'datasets/weights/enet_b0_8_best_afew.state_dict.pth',map_location='cpu',weights_only=True)
for k,v in oldstate.items():
 if not k.startswith('classifier.'):assert torch.equal(v,reference[k]),'Backbone changed: '+k
def head_logits(state,prefix=''):
 with torch.no_grad():return torch.nn.functional.linear(features,state[prefix+'weight'],state[prefix+'bias']).numpy()
new=head_logits(newstate);old=head_logits(oldstate,'classifier.');base=head_logits(reference,'classifier.')
def probs(z,T):
 a=np.asarray(z,dtype=np.float64)/T;a-=a.max(-1,keepdims=True);p=np.exp(a);return p/p.sum(-1,keepdims=True)
def decision(z,p):
 prob=probs(z.mean(1),p['temperature']);pred=prob.argmax(1);same=z[:,0].argmax(1)==z[:,1].argmax(1);v=np.sort(prob,axis=1)
 supported=np.array([C[int(i)] in p['supported_classes'] for i in pred]);threshold=np.array([p['class_thresholds'][C[int(i)]] if p.get('version')==2 else p['min_score'] for i in pred],dtype=object)
 cut=np.array([float(i) if i is not None else np.inf for i in threshold]);accepted=same&supported&(v[:,-1]>=cut)&((v[:,-1]-v[:,-2])>=p['min_margin'])
 return pred,accepted
newpred,newaccept=decision(new,policy);oldpolicy=json.loads((ROOT/'models/emotion_reliable/policy.json').read_text());oldpred,oldaccept=decision(old,oldpolicy)
def wilson(k,n):
 if not n:return [None,None]
 z=1.96;p=k/n;a=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n));center=p+z*z/(2*n);den=1+z*z/n;return [float((center-a)/den),float((center+a)/den)]
def metrics(pred,accept):
 count=int(accept.sum());truth=y[valid];correct=int(((pred==truth)&accept).sum());return {'n_all':len(y),'detected':int(valid.sum()),'answered':count,'correct':correct,'wrong':count-correct,'coverage':count/len(y),'accuracy_answered':correct/count if count else None,'accuracy_wilson95':wilson(correct,count),'correct_fraction_all':correct/len(y)}
report={'protocol_sha256':hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest(),'frozen_policy_sha256':freeze,'v1':metrics(oldpred,oldaccept),'v2':metrics(newpred,newaccept),'raw':{},'predicted_class_precision':{},'true_class_coverage':{},'release_gate':json.loads((P/'release_gate.json').read_text())}
for name,z in [('original',base),('v1',old),('v2',new)]:report['raw'][name]=metrics(z[:,0].argmax(1),np.ones(len(z),bool))
truth=y[valid]
for i,c in enumerate(C):
 mask=newaccept&(newpred==i);n=int(mask.sum());k=int((truth[mask]==i).sum());report['predicted_class_precision'][c]={'predicted':n,'correct':k,'precision':k/n if n else None,'wilson95':wilson(k,n)}
 mask=truth==i;report['true_class_coverage'][c]={'n_total':int((y==i).sum()),'detected':int(mask.sum()),'answered':int(newaccept[mask].sum()),'correct':int(((newpred[mask]==i)&newaccept[mask]).sum())}
g=report['release_gate'];newm=report['v2'];oldm=report['v1'];report['release_passed']=bool(newm['accuracy_answered'] is not None and newm['accuracy_answered']>=g['minimum_selective_accuracy'] and newm['coverage']>=g['minimum_coverage'] and newm['accuracy_answered']>=oldm['accuracy_answered'] and report['raw']['v2']['accuracy_answered']>=report['raw']['v1']['accuracy_answered']-g['require_raw_accuracy_at_least_previous_minus'])
rng=np.random.default_rng(5512);delta=(new[:,0].argmax(1)==truth).astype(float)-(old[:,0].argmax(1)==truth).astype(float);boot=[float(rng.choice(delta,len(delta),replace=True).mean()) for _ in range(2000)];report['raw_accuracy_delta_vs_v1']={'delta':float(delta.mean()),'bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
report['warning']='New holdout is curated strong-consensus FER+ subset, not the earlier FER2013 test. No subject IDs; pretraining overlap unknown. disgust n=1, contempt n=8 do not support rare-class accuracy claims. Thresholds not changed after evaluation.'
(P/'holdout_report.json').write_text(json.dumps(report,indent=2));assert hashlib.sha256((P/'candidate/policy.json').read_bytes()).hexdigest()==freeze
print(json.dumps(report,indent=2))
