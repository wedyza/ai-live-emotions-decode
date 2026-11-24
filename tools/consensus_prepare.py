import os
os.environ['ORT_DISABLE_TELEMETRY']='1'
import csv,json,hashlib,io
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[1]
P=Path(os.environ.get('EMO_CONSENSUS_DIR',ROOT/'datasets/ferplus_consensus')).resolve()
OLD=Path(os.environ.get('EMO_BENCHMARK_DIR',ROOT/'datasets/fer2013_benchmark')).resolve()
CLASSES=['anger','contempt','disgust','fear','happiness','neutral','sadness','surprise']
plus=list(csv.DictReader((P/'ferplus.csv').open()));positions=defaultdict(int);mapped=[];source={}
for name in ['train','publicTest','privateTest']:
 rows=pq.read_table(next(OLD.glob(name+'-*.parquet'))).to_pylist();source[name]=rows
# Verify EVERY original row against mirror pixels and original label, not guessed row ordering.
usage_map={'Training':'train','PublicTest':'publicTest','PrivateTest':'privateTest'}
with (P/'original_fer.csv').open(newline='') as f:
 for idx,raw in enumerate(csv.DictReader(f)):
  row={k.strip():v.strip() for k,v in raw.items()};usage=row['Usage'];split=usage_map[usage];j=positions[split];mirror=source[split][j];positions[split]+=1
  a=np.fromstring(row['pixels'],sep=' ',dtype=np.int16);assert a.size==2304 and a.min()>=0 and a.max()<=255
  gray=a.astype('uint8').reshape(48,48);rgb=np.repeat(gray[...,None],3,2)
  with Image.open(io.BytesIO(mirror['image']['bytes'])) as im:actual=np.asarray(im.convert('RGB'))
  assert np.array_equal(rgb,actual),(idx,split,j,'PIXEL_MISMATCH');assert int(row['emotion'])==mirror['label']
  votes=plus[idx];assert votes['Usage']==usage
  counts=np.array([int(votes[c]) for c in CLASSES]);unknown=int(votes['unknown']);nf=int(votes['NF']);total=int(counts.sum()+unknown+nf)
  strong=total==10 and counts.max()>=6 and unknown<=1 and nf==0
  mapped.append({'index':idx,'split':split,'row':j,'hash':hashlib.sha256(rgb.tobytes()).hexdigest(),'votes':counts.tolist(),'majority':int(counts.argmax()),'strong':bool(strong)})
assert len(mapped)==len(plus)==35887
by_hash=defaultdict(set)
for r in mapped:
 if r['strong']:by_hash[r['hash']].add(r['majority'])
conflicting={h for h,v in by_hash.items() if len(v)>1}
prior=json.loads((ROOT/'models/emotion_head_candidate/training.json').read_text());prior_hashes=set(prior['selected_sha256'])
seen_nontrain={r['hash'] for r in mapped if r['split']!='train'}
clean_train=[];seen=set(seen_nontrain)
for r in mapped:
 if r['split']=='train' and r['strong'] and r['hash'] not in seen and r['hash'] not in conflicting:clean_train.append(r);seen.add(r['hash'])
rng=np.random.default_rng(2026111601);holdout=[]
for c in range(8):
 available=[r for r in clean_train if r['majority']==c and r['hash'] not in prior_hashes];order=rng.permutation(len(available));n=min(150,len(available)//5)
 holdout.extend(available[int(i)] for i in order[:n])
held={r['hash'] for r in holdout};train=[]
for c in range(8):
 available=[r for r in clean_train if r['majority']==c and r['hash'] not in held];order=rng.permutation(len(available));train.extend(available[int(i)] for i in order[:600])
clean_public=[];seen=set()
for r in mapped:
 if r['split']=='publicTest' and r['strong'] and r['hash'] not in conflicting and r['hash'] not in seen:clean_public.append(r);seen.add(r['hash'])
# Independently stratify each class; rare classes may have insufficient calibration evidence and will abstain.
selection=[];calibration=[];policy=[]
for c in range(8):
 available=[r for r in clean_public if r['majority']==c];order=rng.permutation(len(available));n=len(available);nsel=max(1,int(.4*n));ncal=int(.3*n)
 selection.extend(available[int(i)] for i in order[:nsel]);calibration.extend(available[int(i)] for i in order[nsel:nsel+ncal]);policy.extend(available[int(i)] for i in order[nsel+ncal:])
roles={'train':train,'selection':selection,'calibration':calibration,'policy':policy,'holdout':holdout}
sets={k:{r['hash'] for r in v} for k,v in roles.items()}
for k,a in sets.items():
 for j,b in sets.items():
  if k!=j:assert not(a&b),(k,j)
assert not(sets['holdout']&prior_hashes)
summary={'verified_original_rows':len(mapped),'pixel_and_original_label_matches':len(mapped),'strong_majority_rows':sum(r['strong'] for r in mapped),'conflicting_majority_hashes':len(conflicting),'roles':{k:{'n':len(v),'classes':dict(Counter(CLASSES[r['majority']] for r in v))} for k,v in roles.items()},'new_holdout_used_in_previous_training':False,'private_test_used_for_v2_selection':False,'seed':2026111601,'strong_rule':'total10, majority>=6, unknown<=1, not-a-face=0; conflicting-majority hashes removed'}
(P/'protocol.json').write_text(json.dumps(roles,indent=2));(P/'protocol_summary.json').write_text(json.dumps(summary,indent=2));(P/'source_hashes.json').write_text(json.dumps({n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['original_fer.csv','ferplus.csv','ferplus_readme.md']},indent=2))
print(json.dumps(summary,indent=2))
