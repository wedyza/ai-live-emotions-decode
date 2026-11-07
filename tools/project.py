"""Cross-platform entry point. Run with uv run --no-sync python tools/project.py."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SERVICES = ['notification_service', 'ai_service', 'cv_service', 'backend']


def check():
    for name, expected in json.loads((ROOT/'models/checksums.json').read_text()).items():
        p = ROOT/name
        if not p.is_file(): raise SystemExit(f'Missing bundled file: {name}')
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != expected: raise SystemExit(f'Checksum mismatch: {name}')
    from ai_service.analyzer import _recognizer, analyze_emotions
    from common.messages import Face, BoundingBox
    from PIL import Image
    import io
    selected = _recognizer()
    b=io.BytesIO();Image.new('RGB',(96,96),(120,120,120)).save(b,'JPEG')
    result=analyze_emotions(b.getvalue(),[Face(bbox=BoundingBox(x=0,y=0,width=96,height=96))])
    assert result, 'No model output'
    print('Weight checksums: OK')
    print('Selected model:',type(selected).__name__)
    print('Synthetic-image inference: OK (not an accuracy test)')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','start','infra','status'])
    p.add_argument('--model',choices=['experimental','pretrained'],default=None)
    a=p.parse_args()
    os.chdir(ROOT)
    if a.model:os.environ['AI_EMOTION_MODEL']=a.model
    if a.command in ['infra','status']:
        args=['docker','compose']+(['up','-d','--wait'] if a.command=='infra' else ['ps'])
        try:subprocess.run(args,cwd=ROOT,check=True)
        except FileNotFoundError:raise SystemExit('Docker not found. Install/start Docker Desktop with Compose.')
        return
    check()
    if a.command=='check':return
    processes=[]
    try:
        for service in SERVICES:
            kwargs={}
            if os.name!='nt':kwargs['start_new_session']=True
            else:kwargs['creationflags']=subprocess.CREATE_NEW_PROCESS_GROUP
            proc=subprocess.Popen([sys.executable,'-c',f'from {service} import main; main()'],cwd=ROOT,**kwargs)
            processes.append((service,proc))
            print('Started',service,'PID',proc.pid,flush=True)
        print('Open http://localhost:8000/docs after backend startup. Ctrl+C stops these Python services.',flush=True)
        while True:
            for name,proc in processes:
                code=proc.poll()
                if code is not None:raise RuntimeError(f'{name} exited with {code}; stopping other children')
            time.sleep(.3)
    except KeyboardInterrupt:
        print('\nStopping Python services; Docker containers remain running.',flush=True)
    finally:
        for name,proc in processes:
            if proc.poll() is None:
                try:
                    if os.name=='nt':proc.send_signal(signal.CTRL_BREAK_EVENT)
                    else:os.killpg(proc.pid,signal.SIGINT)
                except (ProcessLookupError,OSError):proc.terminate()
        deadline=time.monotonic()+15
        for name,proc in processes:
            try:proc.wait(timeout=max(.1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:proc.kill();proc.wait()

if __name__=='__main__':main()
