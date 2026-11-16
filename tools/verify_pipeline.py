"""Real image -> real YuNet -> Face message -> real ONNX; in-memory transport only."""
import argparse
import asyncio
import csv
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',type=Path,default=ROOT/'datasets/faces_landmarks')
p.add_argument('--model',choices=['reliable','expanded','experimental','aligned'],default='experimental')
a=p.parse_args()
os.environ['AI_EMOTION_MODEL']='reliable' if a.model in ['reliable','expanded'] else 'experimental'
os.environ['AI_EMOTION_MODEL_PATH']='models/emotion_aligned/model.onnx' if a.model=='aligned' else 'models/emotion_experiment/model.onnx'
if a.model=='expanded':os.environ['AI_RELIABILITY_PATH']='models/emotion_consensus'
from common.messages import FrameRef,FaceCaptureMessage
from cv_service.service import FaceCaptureHandler
from ai_service.service import AnalyzeHandler
from notification_service.service import handle_alert
with (a.data/'labels.csv').open(encoding='utf-8',newline='') as f:row=next(csv.DictReader(f))
jpeg=(a.data/row['file']).read_bytes()
class Storage:
    async def get_async(self,key):return jpeg
class Publisher:
    def __init__(self):self.messages=[]
    async def publish(self,message):self.messages.append(message)
async def main():
    first=Publisher();second=Publisher()
    frame=FrameRef(video_id='local-verification',frame_id=row['id'],s3_key=row['file'],timestamp_ms=int(float(row.get('timestamp_ms') or 0)))
    await FaceCaptureHandler(Storage(),first)(FaceCaptureMessage(frame=frame))
    assert len(first.messages)==1,'Expected one AnalyzeMessage'
    message=first.messages[0]
    assert len(message.faces)==1 and message.faces[0].landmarks is not None
    assert len(message.faces[0].landmarks.points)==5
    await AnalyzeHandler(Storage(),second)(message)
    assert len(second.messages)==1,'No alert produced'
    await handle_alert(second.messages[0])
    print('Pipeline handlers: OK')
    print('Landmarks preserved:',len(message.faces[0].landmarks.points))
    print('Model:',a.model)
    print('Real image result:',second.messages[0].text)
    print('Transport: in-memory fixture, NOT a Docker/RabbitMQ/MinIO integration test')
asyncio.run(main())
