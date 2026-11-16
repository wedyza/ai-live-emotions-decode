import os
os.environ['ORT_DISABLE_TELEMETRY']='1'
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parents[1]
P=Path(os.environ.get('EMO_VARIANT2_DIR',ROOT/'datasets/variant2_experiment')).resolve();P.mkdir(parents=True,exist_ok=True)
OLD=Path(os.environ.get('EMO_BENCHMARK_DIR',ROOT/'datasets/fer2013_benchmark')).resolve()
V1=Path(os.environ.get('EMO_CONSENSUS_DIR',ROOT/'datasets/ferplus_consensus')).resolve()
sys.path.insert(0,str(ROOT/'tools'))
from variant2_arch import LandmarkFusionHead
C=['anger','contempt','disgust','fear','happiness','neutral','sadness','surprise'];torch.set_num_threads(2)
assert (P/'training_report.json').exists();policy=json.loads((P/'candidate/policy.json').read_text());freeze=hashlib.sha256((P/'candidate/policy.json').read_bytes()).hexdigest();assert freeze==(P/'frozen_policy_sha256.txt').read_text()
d=np.load(P/'cache.npz');all_features=d['features'];all_geometry=d['geometry'];lookup={str(h):i for i,h in enumerate(d['hashes'])};rows=json.loads((P/'protocol.json').read_text())['test'];valid=np.array([r['hash'] in lookup for r in rows]);ids=[lookup[r['hash']] for r in rows if r['hash'] in lookup];x=torch.tensor(all_features[ids]);g=torch.tensor(all_geometry[ids]);y=np.array([r['majority'] for r in rows])[valid]
state=torch.load(P/'candidate/fusion_state.pth',map_location='cpu',weights_only=True)
def load(s):
 h=LandmarkFusionHead(s['image_weight'],s['image_bias'],s['geometry_mean'],s['geometry_std']);h.load_state_dict(s);return h.eval()
h=load(state);control=load(torch.load(P/'candidate/constant_control.pth',map_location='cpu',weights_only=True))
with torch.no_grad():
 z1=torch.nn.functional.linear(x,h.image_weight,h.image_bias).numpy();z2=h(x,g).numpy();zm=h(x,h.geometry_mean.expand_as(g)).numpy();zc=control(x,control.geometry_mean.expand_as(g)).numpy()
def soft(z,T):
 a=np.asarray(z,dtype=np.float64)/T;a-=a.max(-1,keepdims=True);p=np.exp(a);return p/p.sum(-1,keepdims=True)
def selective(z,p):
 prob=soft(z.mean(1),p['temperature']);pred=prob.argmax(1);cuts=np.array([p['class_thresholds'][C[int(i)]] for i in pred],dtype=object);cuts=np.array([float(v) if v is not None else np.inf for v in cuts]);accepted=(z[:,0].argmax(1)==z[:,1].argmax(1))&(prob.max(1)>=cuts);return pred,accepted
p1=json.loads((ROOT/'models/emotion_consensus/policy.json').read_text());pred1,a1=selective(z1,p1);pred2,a2=selective(z2,policy)
def wilson(k,n):
 if not n:return [None,None]
 z=1.96;p=k/n;v=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n));c=p+z*z/(2*n);q=1+z*z/n;return [float((c-v)/q),float((c+v)/q)]
def metrics(pred,accepted):
 n=int(accepted.sum());k=int(((pred==y)&accepted).sum());return {'n_total':len(rows),'matched_detected':len(y),'answered':n,'coverage_all':n/len(rows),'correct':k,'wrong':n-k,'accuracy_answered':k/n if n else None,'wilson95':wilson(k,n),'correct_fraction_all':k/len(rows)}
raw1=z1[:,0].argmax(1);raw2=z2[:,0].argmax(1);rawc=zc[:,0].argmax(1);rawm=zm[:,0].argmax(1);rng=np.random.default_rng(77005)
def delta_ci(a,b):
 delta=(a==y).astype(float)-(b==y).astype(float);values=[float(rng.choice(delta,len(delta),replace=True).mean()) for _ in range(5000)];return {'delta':float(delta.mean()),'bootstrap95':np.quantile(values,[.025,.975]).tolist()}
shuffled=[]
for _ in range(20):
 ix=rng.permutation(len(g))
 with torch.no_grad():zs=h(x,g[ix]).numpy()
 shuffled.append(float((zs[:,0].argmax(1)==y).mean()))
report={'test_protocol_sha256':hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest(),'variant2_frozen_policy_sha256':freeze,'variant1_raw':metrics(raw1,np.ones(len(y),bool)),'variant2_raw':metrics(raw2,np.ones(len(y),bool)),'variant1_selective':metrics(pred1,a1),'variant2_selective':metrics(pred2,a2),'raw_macro_f1_present_classes':{'variant1':float(f1_score(y,raw1,labels=sorted(set(y)),average='macro',zero_division=0)),'variant2':float(f1_score(y,raw2,labels=sorted(set(y)),average='macro',zero_division=0))},'paired_variant2_vs_variant1':delta_ci(raw2,raw1),'constant_control_raw':metrics(rawc,np.ones(len(y),bool)),'paired_variant2_vs_constant_control':delta_ci(raw2,rawc),'variant2_mean_geometry_raw':metrics(rawm,np.ones(len(y),bool)),'variant2_shuffled_geometry_accuracy':{'mean':float(np.mean(shuffled)),'min':float(np.min(shuffled)),'max':float(np.max(shuffled)),'replicates':20},'geometry_logits_actual_vs_mean_max_abs':float(np.abs(z2-zm).max()),'per_class':{}}
for i,c in enumerate(C):
 mask=y==i
 if mask.any():report['per_class'][c]={'n':int(mask.sum()),'variant1_raw_accuracy':float((raw1[mask]==y[mask]).mean()),'variant2_raw_accuracy':float((raw2[mask]==y[mask]).mean())}
report['evidence_of_geometry_benefit']=bool(report['paired_variant2_vs_variant1']['bootstrap95'][0]>0 and report['paired_variant2_vs_constant_control']['bootstrap95'][0]>0)
report['default_changed']=False;report['warning']='Fresh 750-image subset covers only five common consensus labels. No evidence for rare classes, subject independence, pretraining exclusion, or arbitrary video. No tuning after test.'
(P/'comparison.json').write_text(json.dumps(report,indent=2));np.savez_compressed(P/'test_predictions.npz',truth=y,variant1_logits=z1,variant2_logits=z2,geometry=g.numpy(),hashes=[r['hash'] for r in rows if r['hash'] in lookup]);assert hashlib.sha256((P/'candidate/policy.json').read_bytes()).hexdigest()==freeze;print(json.dumps(report,indent=2))
