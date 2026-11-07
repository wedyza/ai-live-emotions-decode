import concurrent.futures
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from common.paths import project_root, project_path
from common.messages import Face, BoundingBox
from ai_service.local_emotion_model import LocalEmotionModel
from ai_service.analyzer import analyze_emotions, _recognizer
from cv_service.detector import detect_faces

ROOT=project_root()

class PortabilityTests(unittest.TestCase):
    def test_paths_independent_of_cwd(self):
        previous=Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                self.assertEqual(project_path('models/emotion_experiment/model.onnx'),ROOT/'models/emotion_experiment/model.onnx')
                self.assertTrue(project_path('models/emotion_experiment/model.onnx').is_file())
            finally:os.chdir(previous)

    def test_missing_model_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError,'AI_EMOTION_MODEL'):
            LocalEmotionModel('/does-not-exist/model.onnx')

    def test_real_model_batch_one_rgb(self):
        model=LocalEmotionModel(str(ROOT/'models/emotion_experiment/model.onnx'))
        rgb=np.full((96,80,3),128,np.uint8)
        label,scores=model.predict_emotions(rgb)
        self.assertIn(label,model.classes)
        self.assertEqual(scores.shape,(8,))
        self.assertTrue(np.isfinite(scores).all())
        self.assertAlmostEqual(float(scores.sum()),1.,places=5)
        with self.assertRaises(ValueError):model.predict_emotions(np.zeros((0,0,3),np.uint8))

    def test_service_inference_without_dataset(self):
        image=Image.new('RGB',(128,96),(250,10,10));b=io.BytesIO();image.save(b,'JPEG')
        output=analyze_emotions(b.getvalue(),[Face(bbox=BoundingBox(x=10,y=10,width=50,height=60))])
        self.assertIn('face #1:',output)
        self.assertIsNone(analyze_emotions(b'bad',[]))
        with self.assertRaises(ValueError):analyze_emotions(b'bad',[Face(bbox=BoundingBox(x=0,y=0,width=5,height=5))])

    def test_detector_serializes_input_mutation_and_forward(self):
        # Fake the slow native forward to expose overlapping calls reliably.
        class SharedDetector:
            def __init__(self):self.current=None;self.active=0;self.maximum=0;self.guard=threading.Lock()
            def setInputSize(self,size):self.current=tuple(size)
            def infer(self,image):
                with self.guard:self.active+=1;self.maximum=max(self.maximum,self.active)
                size=(image.shape[1],image.shape[0]);time.sleep(.005)
                assert self.current==size,'Concurrent input size mutation'
                with self.guard:self.active-=1
                return np.empty((0,15),np.float32)
        detector=SharedDetector();pictures=[]
        for size in [(96,96),(160,120),(256,96),(96,160)]*4:
            b=io.BytesIO();Image.new('RGB',size).save(b,'JPEG');pictures.append(b.getvalue())
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda jpeg:detect_faces(jpeg,detector),pictures))
        self.assertEqual(detector.maximum,1)

    def test_real_yunet_concurrent_different_frame_sizes(self):
        from model.yunet import get_yunet
        model=get_yunet()
        pictures=[]
        for size in [(320,320),(640,480),(852,480),(480,640)]*2:
            b=io.BytesIO();Image.new('RGB',size,(128,128,128)).save(b,'JPEG');pictures.append(b.getvalue())
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(lambda jpeg:detect_faces(jpeg,model),pictures))
        self.assertEqual(len(results),len(pictures))

    def test_no_target_coordinate_collision_in_notebook(self):
        nb=json.loads((ROOT/'notebooks/02_train_eval.ipynb').read_text())
        source='\n'.join(''.join(c['source']) for c in nb['cells'])
        self.assertNotIn('df["y"] =',source)
        self.assertIn('frame["target"]',source)
        self.assertIn('row["y"]',source)
        self.assertNotIn('/Users/',source)
        for cell in nb['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'<notebook>','exec')

if __name__=='__main__':unittest.main()
