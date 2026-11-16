"""Frozen EfficientNet backbone, balanced FER2013 head adaptation with distillation."""
import os
os.environ['ORT_DISABLE_TELEMETRY']='1'
from quality_benchmark_lib import P,ROOT,read,image_face,prepare_face,ALIGNED,CLASSES,FER
import sys,json,copy,time,hashlib
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import v2 as T
from sklearn.metrics import f1_score
sys.path.insert(0,str(ROOT/'tools'))
from training_model import load_hsemotion

torch.set_num_threads(2);torch.manual_seed(4245);rng=np.random.default_rng(4245)
rows,y,hashes=read('train')
groups={}
for h,c in zip(hashes,y):groups.setdefault(str(h),set()).add(int(c))
(P/'conflicting_train_hashes.json').write_text(json.dumps([h for h,c in groups.items() if len(c)>1]))
public=np.load(P/'public_logits.npz');excluded=set(public['hashes']);selected=[];seen=set(excluded)
for c in sorted(set(y)):
 ids=np.where(y==c)[0];ids=rng.permutation(ids)
 count=0
 for i in ids:
  if hashes[i] in seen:continue
  seen.add(hashes[i]);selected.append(int(i));count+=1
  if count>=700:break
trainrows=[rows[i] for i in selected];train_y=y[selected]
valrows,val_y,val_hashes=read('publicTest');unique=np.unique(val_hashes,return_index=True)[1];unique.sort();valrows=[valrows[i] for i in unique];val_y=val_y[unique]
sel=public['selection'];valrows=[valrows[i] for i in sel];val_y=val_y[sel]
model,size,classes=load_hsemotion('enet_b0_8_best_afew',ROOT/'datasets/weights');model.eval()
for param in model.parameters():param.requires_grad=False
transform=T.Compose([T.Resize((224,224)),T.ToImage(),T.ToDtype(torch.float32,scale=True),T.Normalize((.485,.456,.406),(.229,.224,.225))])
def extract(rows,truth,cache):
 if cache.exists():
  d=np.load(cache);return torch.tensor(d['features']),torch.tensor(d['targets']),d['positions']
 start=time.perf_counter();queue=[];features=[];targets=[];positions=[]
 def flush():
  if queue:
   with torch.inference_mode():feat=model.forward_head(model.forward_features(torch.stack(queue)),pre_logits=True)
   features.append(feat.cpu().numpy());queue.clear()
 for i,row in enumerate(rows):
  data=image_face(row,ALIGNED)
  if data:
   try:face=prepare_face(*data,mode=ALIGNED)
   except ValueError:continue
   im=Image.fromarray(face)
   for v in [im,im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)]:queue.append(transform(v));targets.append(int(truth[i]));positions.append(i)
   if len(queue)>=32:flush()
  if i%300==0:print(cache.name,i,'/',len(rows),'s',round(time.perf_counter()-start,1),flush=True)
 flush();x=np.concatenate(features);assert np.isfinite(x).all();np.savez_compressed(cache,features=x,targets=targets,positions=positions)
 return torch.tensor(x),torch.tensor(targets),np.array(positions)
x,y,train_positions=extract(trainrows,train_y,P/'head_train_features.npz');v,vy,positions=extract(valrows,val_y,P/'head_val_features.npz')
conflicts=set(json.loads((P/'conflicting_train_hashes.json').read_text()))
keep=np.array([hashes[selected[int(i)]] not in conflicts for i in train_positions]);excluded_conflicts=int((~keep).sum()//2)
x=x[keep];y=y[keep]
used_train_hashes=[str(hashes[selected[int(i)]]) for i in train_positions[keep][::2]]
print('Used detected training images',len(x)//2,'excluded conflicting-label images',excluded_conflicts,flush=True)
original=copy.deepcopy(model.classifier);best={'f1':-1.,'step':0,'distillation':None,'state':copy.deepcopy(original.state_dict())};history=[]
def score(head):
 with torch.no_grad():logits=head(v).numpy()[::2];pred=np.full(len(val_y),-1);pred[positions[::2]]=logits.argmax(1)
 return float(f1_score(val_y,pred,labels=[CLASSES.index(c) for c in FER],average='macro',zero_division=0))
best['f1']=score(original);baseline=best['f1'];teacher=original(x).detach();teacherprob=torch.softmax(teacher/2,dim=1)
for strength in [.25,1.,4.]:
 head=copy.deepcopy(original)
 for param in head.parameters():param.requires_grad=True
 optim=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
 generator=torch.Generator().manual_seed(4245)
 for step in range(1,201):
  ids=torch.randint(len(x),(256,),generator=generator);logits=head(x[ids]);ce=torch.nn.functional.cross_entropy(logits,y[ids]);kl=torch.nn.functional.kl_div(torch.log_softmax(logits/2,dim=1),teacherprob[ids],reduction='batchmean')*4
  loss=ce+strength*kl;optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optim.step()
  if step in [50,100,200]:
   f1=score(head);history.append(dict(strength=strength,step=step,macro_f1=f1));print(history[-1],flush=True)
   if f1>best['f1']:best=dict(f1=f1,step=step,distillation=strength,state=copy.deepcopy(head.state_dict()))
model.classifier.load_state_dict(best['state']);out=P/'head_candidate';out.mkdir(exist_ok=True)
torch.save(dict(state_dict=model.state_dict(),classes=classes,img_size=size,crop=ALIGNED,arch='tf_efficientnet_b0'),out/'model.pth')
torch.onnx.export(model,(torch.zeros(1,3,224,224),),str(out/'model.onnx'),input_names=['input'],output_names=['logits'],opset_version=18,dynamo=True,external_data=False)
meta=dict(classes=classes,img_size=224,mean=[.485,.456,.406],std=[.229,.224,.225],input='RGB',resize='PIL bilinear',crop=ALIGNED,batch_size=1,output='logits',adaptation='frozen backbone, distilled linear head')
(out/'model.onnx.json').write_text(json.dumps(meta,indent=2))
from ai_service.local_emotion_model import LocalEmotionModel
local=LocalEmotionModel(str(out/'model.onnx'))
# Export verification on real retained selection images; no calibration/private labels used.
checked=0;maxdiff=0.
for row in valrows[:20]:
 data=image_face(row,ALIGNED)
 if data is None:continue
 try:face=prepare_face(*data,mode=ALIGNED)
 except ValueError:continue
 with torch.inference_mode():reference=model(transform(Image.fromarray(face)).unsqueeze(0)).numpy()[0]
 actual=local.predict_emotions(face,logits=True)[1];np.testing.assert_allclose(actual,reference,rtol=1e-3,atol=1e-4);maxdiff=max(maxdiff,float(np.abs(actual-reference).max()));checked+=1
report={'train_selected':len(selected),'train_detected':len(x)//2,'conflicting_labels_excluded':excluded_conflicts,'selection_images':len(valrows),'original_macro_f1':baseline,'best_macro_f1':best['f1'],'step':best['step'],'distillation':best['distillation'],'history':history,'seed':4245,'train_sha256':used_train_hashes,'selected_sha256':[str(hashes[i]) for i in selected],'test_used':False,'calibration_policy_labels_used':False,'export_checked':checked,'onnx_max_difference':maxdiff}
(out/'training.json').write_text(json.dumps(report,indent=2));print('SAVED',json.dumps({k:v for k,v in report.items() if k not in ['train_sha256','selected_sha256','history']},indent=2),flush=True)
