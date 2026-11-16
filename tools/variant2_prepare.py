import os
os.environ['ORT_DISABLE_TELEMETRY']='1'
import json,csv,hashlib,io
from pathlib import Path
import numpy as np,pyarrow.parquet as pq
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
P=Path(os.environ.get('EMO_VARIANT2_DIR',ROOT/'datasets/variant2_experiment')).resolve();P.mkdir(parents=True,exist_ok=True)
OLD=Path(os.environ.get('EMO_BENCHMARK_DIR',ROOT/'datasets/fer2013_benchmark')).resolve()
V1=Path(os.environ.get('EMO_CONSENSUS_DIR',ROOT/'datasets/ferplus_consensus')).resolve()
C=['anger','contempt','disgust','fear','happiness','neutral','sadness','surprise']
protocol=json.loads((ROOT/'reports/consensus/protocol.json').read_text());prior=json.loads((ROOT/'models/emotion_head_candidate/training.json').read_text());used=set(prior['selected_sha256'])
for v in protocol.values():used.update(r['hash'] for r in v)
# Exclude all non-training source images, including exact cross-split duplicates.
for split in ['publicTest','privateTest']:
 for r in pq.read_table(next(OLD.glob(split+'-*.parquet'))).to_pylist():
  with Image.open(io.BytesIO(r['image']['bytes'])) as im:used.add(hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest())
votes=[r for r in csv.DictReader((V1/'ferplus.csv').open()) if r['Usage']=='Training'];rows=pq.read_table(next(OLD.glob('train-*.parquet'))).to_pylist();assert len(rows)==len(votes)==28709
available={c:[] for c in range(8)};seen=set(used)
for i,(r,v) in enumerate(zip(rows,votes)):
 counts=np.array([int(v[c]) for c in C]);unknown=int(v['unknown']);nf=int(v['NF'])
 if int(counts.sum()+unknown+nf)!=10 or counts.max()<6 or unknown>1 or nf:continue
 with Image.open(io.BytesIO(r['image']['bytes'])) as im:h=hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest()
 if h in seen:continue
 seen.add(h);available[int(counts.argmax())].append({'index':i,'split':'train','row':i,'hash':h,'votes':counts.tolist(),'majority':int(counts.argmax()),'strong':True})
rng=np.random.default_rng(77001);test=[]
for c in range(8):
 ids=rng.permutation(len(available[c]));test.extend(available[c][int(i)] for i in ids[:150])
roles={k:protocol[k] for k in ['train','selection','calibration','policy']};roles['test']=test
for k,a in roles.items():
 for j,b in roles.items():
  if k!=j:assert not({r['hash'] for r in a}&{r['hash'] for r in b})
assert not({r['hash'] for r in test}&used)
(P/'protocol.json').write_text(json.dumps(roles,indent=2));summary={'seed':77001,'previous_test_reused':False,'new_test':len(test),'test_by_class':{C[c]:sum(r['majority']==c for r in test) for c in range(8)},'unused_pool_by_class':{C[c]:len(v) for c,v in available.items()},'roles':{k:len(v) for k,v in roles.items()},'warning':'Fresh relative to project adaptations/evaluations, not guaranteed unseen subjects or HSEmotion pretraining. No fresh rare-class evaluation.'};(P/'protocol_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
