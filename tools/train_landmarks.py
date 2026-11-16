"""Reproducible validation-only bbox/aligned experiment; no test-set tuning."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'libs/common/src'))
sys.path.insert(0,str(ROOT/'ai-service/src'))
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from torchvision.transforms import v2 as T
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from common.face_preprocessing import prepare_face,BBOX,ALIGNED
from training_model import load_hsemotion,restrict_head,HSE_CLASSES_8


def measure(y,p):
    return dict(accuracy=float(accuracy_score(y,p)),balanced_accuracy=float(balanced_accuracy_score(y,p)),
                macro_f1=float(f1_score(y,p,labels=sorted(set(y)),average='macro',zero_division=0)))


class FaceRows(Dataset):
    def __init__(self,rows,data,mode,tf,classes):
        self.rows=rows;self.data=data;self.mode=mode;self.tf=tf
        self.targets=[classes.index(r['label']) for r in rows]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];path=(self.data/r['file']).resolve()
        if not path.is_relative_to(self.data):raise ValueError('Image path outside dataset')
        with Image.open(path) as im:rgb=np.asarray(im.convert('RGB')).copy()
        bbox=tuple(int(r[k]) for k in ['x','y','width','height'])
        points=json.loads(r['landmarks']) if self.mode==ALIGNED else None
        face=prepare_face(rgb,bbox,points,self.mode)
        return self.tf(Image.fromarray(face)),self.targets[i]


@torch.no_grad()
def predict(model,loader,device):
    model.eval();out=[];targets=[]
    for x,y in loader:
        out.extend(model(x.to(device)).argmax(1).cpu().tolist());targets.extend(y.tolist())
    return np.asarray(targets),np.asarray(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'datasets/faces_landmarks')
    p.add_argument('--out',type=Path)
    p.add_argument('--mode',choices=['bbox','aligned'],required=True)
    p.add_argument('--head-epochs',type=int,default=2);p.add_argument('--full-epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto')
    p.add_argument('--smoke',action='store_true',help='Only first 32 train and 16 val rows, one epoch per stage; NOT a quality experiment')
    a=p.parse_args()
    if min(a.head_epochs,a.full_epochs)<0 or a.batch_size<1 or a.threads<1:p.error('Invalid training counts')
    torch.set_num_threads(a.threads);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    data=a.data.resolve();out=a.out or ROOT/'reports'/('landmarks_'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'_'+a.mode)
    out=out.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    classes=json.loads((data/'classes.json').read_text())
    if classes!=HSE_CLASSES_8:raise ValueError('Expected original eight-class order')
    with (data/'labels.csv').open(encoding='utf-8-sig',newline='') as f:all_rows=list(csv.DictReader(f))
    for group in ['subject_id','clip_id']:
        seen={}
        for r in all_rows:
            if not r[group] or r['split'] not in ['train','val','test']:raise ValueError('Missing group or invalid split')
            seen.setdefault(r[group],set()).add(r['split'])
        if any(len(v)>1 for v in seen.values()):raise ValueError('Group leakage: '+group)
    rows=[r for r in all_rows if r['label'] in classes and r['is_dup'].lower()=='false']
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
    if a.smoke:train=train[:32];val=val[:16];a.head_epochs=1;a.full_epochs=1
    if not train or not val:raise ValueError('Empty train/val')
    if a.mode=='aligned' and any(not r.get('landmarks') for r in train+val):raise ValueError('Run build_landmarks first')
    device=a.device
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device=torch.device(device);mode=ALIGNED if a.mode=='aligned' else BBOX
    name='enet_b0_8_best_afew';model,size,source_classes=load_hsemotion(name,ROOT/'datasets/weights')
    model=restrict_head(model,source_classes,classes).to(device)
    tf_eval=T.Compose([T.Resize((size,size)),T.ToImage(),T.ToDtype(torch.float32,scale=True),T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_train=T.Compose([T.RandomResizedCrop(size,scale=(.8,1.),ratio=(.9,1.1)),T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(.3,.3,.2,.02)],p=.8),T.RandomRotation(10),T.RandomGrayscale(.05),T.ToImage(),
        T.ToDtype(torch.float32,scale=True),T.Normalize((.485,.456,.406),(.229,.224,.225)),T.RandomErasing(p=.25,scale=(.02,.12))])
    ds=FaceRows(train,data,mode,tf_train,classes);dv=FaceRows(val,data,mode,tf_eval,classes)
    frequencies=np.bincount(ds.targets,minlength=8)
    sampler=WeightedRandomSampler((1./np.maximum(frequencies,1))[ds.targets],num_samples=len(ds),replacement=True)
    loader=DataLoader(ds,batch_size=a.batch_size,sampler=sampler,num_workers=0,drop_last=len(ds)>a.batch_size)
    valid=DataLoader(dv,batch_size=a.batch_size,num_workers=0,shuffle=False)
    y,pred=predict(model,valid,device);baseline=measure(y,pred)
    best=dict(epoch=0,macro_f1=baseline['macro_f1'],state=copy.deepcopy(model.state_dict()))
    history=[];epoch=0;criterion=torch.nn.CrossEntropyLoss(label_smoothing=.1)
    print('device',device,'mode',a.mode,'train',len(train),'val',len(val),'baseline',baseline,flush=True)
    for head_only,n_epochs in [(True,a.head_epochs),(False,a.full_epochs)]:
        for param in model.parameters():param.requires_grad=not head_only
        for param in model.classifier.parameters():param.requires_grad=True
        optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-3 if head_only else 1e-4,weight_decay=1e-4)
        scheduler=None if head_only or n_epochs==0 else torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=1e-4,total_steps=n_epochs*len(loader),pct_start=.1)
        for _ in range(n_epochs):
            epoch+=1;model.eval() if head_only else model.train();total=0.;count=0
            for x,target in loader:
                x=x.to(device);target=target.to(device);optimizer.zero_grad(set_to_none=True)
                loss=criterion(model(x),target)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                if scheduler is not None:scheduler.step()
                total+=float(loss.detach())*len(target);count+=len(target)
            y,pred=predict(model,valid,device);metrics=measure(y,pred)
            row=dict(epoch=epoch,stage='head' if head_only else 'full',train_loss=total/count,**metrics);history.append(row)
            if metrics['macro_f1']>best['macro_f1']:best=dict(epoch=epoch,macro_f1=metrics['macro_f1'],state=copy.deepcopy(model.state_dict()))
            print(json.dumps(row),flush=True)
    model.load_state_dict(best['state']);model=model.cpu().eval()
    torch.save(dict(state_dict=model.state_dict(),classes=classes,img_size=size,crop=mode,arch='tf_efficientnet_b0'),out/'model.pth')
    import onnx
    import onnxruntime as ort
    torch.onnx.export(model,(torch.zeros(1,3,size,size),),str(out/'model.onnx'),input_names=['input'],output_names=['logits'],opset_version=18,dynamo=True,external_data=False)
    onnx.checker.check_model(str(out/'model.onnx'))
    metadata=dict(classes=classes,img_size=size,mean=[.485,.456,.406],std=[.229,.224,.225],input='RGB',resize='PIL bilinear',crop=mode,batch_size=1,output='logits',best_epoch=best['epoch'])
    (out/'model.onnx.json').write_text(json.dumps(metadata,indent=2))
    session=ort.InferenceSession(str(out/'model.onnx'),providers=['CPUExecutionProvider']);maxdiff=0.
    from ai_service.local_emotion_model import LocalEmotionModel
    local=LocalEmotionModel(str(out/'model.onnx'));pipeline_diff=0.
    # Check one image from every represented validation class, not just first ten anger frames.
    examples=[];seen=set()
    for i,r in enumerate(val):
        if r['label'] not in seen:examples.append(i);seen.add(r['label'])
    for i in examples:
        x,_=dv[i];x=x.unsqueeze(0)
        with torch.no_grad():reference=model(x).numpy()
        actual=session.run(['logits'],{'input':x.numpy()})[0]
        np.testing.assert_allclose(actual,reference,rtol=1e-3,atol=1e-4)
        maxdiff=max(maxdiff,float(np.abs(reference-actual).max()))
        r=val[i]
        with Image.open(data/r['file']) as im:rgb=np.asarray(im.convert('RGB')).copy()
        points=json.loads(r['landmarks']) if mode==ALIGNED else None
        face=prepare_face(rgb,tuple(int(r[k]) for k in ['x','y','width','height']),points,mode)
        _,got=local.predict_emotions(face,logits=True)
        np.testing.assert_allclose(got,reference[0],rtol=1e-3,atol=1e-4)
        pipeline_diff=max(pipeline_diff,float(np.abs(got-reference[0]).max()))
    report=dict(mode=a.mode,preprocessing=mode,smoke=a.smoke,device=str(device),seed=a.seed,train_rows=len(train),val_rows=len(val),
                train_ids=[r['id'] for r in train],val_ids=[r['id'] for r in val],baseline_val=baseline,best_epoch=best['epoch'],best_val_macro_f1=best['macro_f1'],history=history,
                pipeline_maxdiff=pipeline_diff,labels_sha256=hashlib.sha256((data/'labels.csv').read_bytes()).hexdigest(),onnx_maxdiff=maxdiff,
                test_evaluated=False,created=datetime.now(timezone.utc).isoformat(),torch=torch.__version__,
                warning='Weak video labels. Validation experiment, not a proven improvement on independent subjects.')
    (out/'experiment.json').write_text(json.dumps(report,indent=2))
    print('SAVED',out,flush=True)

if __name__=='__main__':main()
