"""Enrich a frame dataset without changing its labels, subjects or split assignments."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from common.face_preprocessing import ALIGNED, prepare_face
from cv_service.detector import detect_captures
from model.yunet import get_yunet


def iou(a,b):
    x,y,w,h=a;u,v,s,t=b
    area=max(0,min(x+w,u+s)-max(x,u))*max(0,min(y+h,v+t)-max(y,v))
    union=w*h+s*t-area
    return area/union if union>0 else 0.


def build(data,output,min_iou=.5):
    data=data.resolve();output=output.resolve()
    if output==data or output.is_relative_to(data):raise ValueError('Output must be separate from input dataset')
    if output.exists() and any(output.iterdir()):raise ValueError('Output must be empty')
    csv_path=data/'labels.csv'
    with csv_path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=list(reader.fieldnames or []);records=list(reader)
    required={'id','file','label','x','y','width','height','clip_id','subject_id','split'}
    if not required.issubset(fields):raise ValueError(f'Missing columns: {required-set(fields)}')
    ids=[r['id'] for r in records]
    if not records or len(set(ids))!=len(ids) or any(not i for i in ids):raise ValueError('Require non-empty unique IDs')
    for field in ['clip_id','subject_id']:
        groups={}
        for r in records:groups.setdefault(r[field],set()).add(r['split'])
        if any(len(v)!=1 for v in groups.values()):raise ValueError(f'Split leakage by {field}')
    output.mkdir(parents=True,exist_ok=True)
    model=get_yunet();accepted=[];rejected=[]
    for index,row in enumerate(records,1):
        image_path=(data/row['file']).resolve()
        if not image_path.is_relative_to(data):raise ValueError('Image path escapes input directory')
        jpeg=image_path.read_bytes();faces=detect_captures(jpeg,model)
        reason='no_face' if not faces else 'multiple_faces' if len(faces)!=1 else ''
        old=tuple(int(row[k]) for k in ['x','y','width','height'])
        if not reason:
            face=faces[0];b=face.bbox;new=(b.x,b.y,b.width,b.height)
            if iou(old,new)<min_iou:reason='bbox_mismatch'
        if reason:
            rejected.append(dict(id=row['id'],file=row['file'],reason=reason));continue
        with Image.open(io.BytesIO(jpeg)) as image:rgb=np.asarray(image.convert('RGB')).copy()
        try:aligned=prepare_face(rgb,new,face.landmarks.points,ALIGNED)
        except ValueError as exc:
            rejected.append(dict(id=row['id'],file=row['file'],reason='invalid_landmarks: '+str(exc)));continue
        item=dict(row)
        for key,value in zip(['x','y','width','height'],new):
            item['source_bbox_'+key]=row[key];item[key]=value
        item['landmarks']=json.dumps(face.landmarks.points,separators=(',',':'))
        item['det_score']=face.det_score
        # Safe generated paths, independent of IDs or source paths with ../.
        token=hashlib.sha256(row['id'].encode()).hexdigest()[:24]
        item['file']='images/'+token+image_path.suffix.lower()
        item['crop_path']='crops/'+token+'.png'
        item['landmark_order']='right_eye,left_eye,nose,right_mouth,left_mouth'
        for name,point in zip(['right_eye','left_eye','nose','right_mouth','left_mouth'],face.landmarks.points):
            item[name+'_x'],item[name+'_y']=point
        dst=output/item['file'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(image_path,dst)
        preview=output/item['crop_path'];preview.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(aligned).save(preview)
        accepted.append(item)
        if index%100==0:print(f'{index}/{len(records)} checked; {len(accepted)} retained',flush=True)
    if not accepted:raise ValueError('No accepted faces; inspect input bbox/landmarks before training')
    extra=['landmarks','landmark_order','det_score','crop_path']+[f'source_bbox_{k}' for k in ['x','y','width','height']]+[n+'_'+axis for n in ['right_eye','left_eye','nose','right_mouth','left_mouth'] for axis in ['x','y']]
    columns=list(dict.fromkeys(fields+extra))
    for filename in ['labels.csv','dataset.csv']:
        with (output/filename).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(accepted)
    for name in ['classes.json']:
        shutil.copy2(data/name,output/name)
    with (output/'landmark_rejected.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','file','reason']);w.writeheader();w.writerows(rejected)
    report=dict(input_rows=len(records),accepted=len(accepted),rejected=len(rejected),
                source_labels_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                alignment=ALIGNED,points='five YuNet points in full-frame pixels',
                note='Labels and split assignments preserved. Compare bbox and alignment on this SAME retained dataset.')
    (output/'landmark_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--min-iou',type=float,default=.5)
    a=p.parse_args()
    if not 0<=a.min_iou<=1:p.error('--min-iou must be 0..1')
    build(a.data,a.out,a.min_iou)

if __name__=='__main__':main()
