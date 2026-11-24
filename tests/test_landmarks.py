import concurrent.futures
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from common.face_preprocessing import prepare_face, similarity_matrix, TEMPLATE, ALIGNED, BBOX
from common.messages import Face, FaceLandmarks, BoundingBox
from cv_service.detector import detect_captures, detect_faces
from ai_service.analyzer import analyze_emotions


def frame():
    y,x=np.indices((300,300));return np.stack([x%256,y%256,(x+y)%256],axis=-1).astype(np.uint8)

def encoded(rgb):
    b=io.BytesIO();Image.fromarray(rgb).save(b,'JPEG');return b.getvalue()

class LandmarkTests(unittest.TestCase):
    def test_similarity_recovers_known_transform(self):
        angle=.25;scale=1.3
        rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        points=(TEMPLATE@rot.T)*scale+np.array([35,20])
        matrix=similarity_matrix(points)
        got=np.column_stack([points,np.ones(5)])@matrix.T
        np.testing.assert_allclose(got,TEMPLATE,atol=1e-8)
        # Rotation is not a free affine warp/reflection.
        linear=matrix[:,:2];self.assertGreater(np.linalg.det(linear),0)
        np.testing.assert_allclose(linear.T@linear,np.eye(2)/(scale**2),atol=1e-8)

    def test_raw_mode_is_unchanged(self):
        image=frame()
        np.testing.assert_array_equal(prepare_face(image,(10,20,60,70)),image[20:90,10:70])

    def test_invalid_points_never_fall_back_to_bbox(self):
        for points in [None,[[1,2]]*5,[[float('nan'),0]]*5,[[1,2]]*4]:
            with self.assertRaises(ValueError):prepare_face(frame(),(10,10,200,200),points,ALIGNED)

    def test_backwards_message_compatibility_and_point_order(self):
        face=Face.model_validate({'bbox':{'x':0,'y':0,'width':100,'height':100}})
        self.assertIsNone(face.landmarks)
        points=TEMPLATE.tolist();face=Face(bbox=face.bbox,landmarks=FaceLandmarks(points=points),det_score=.9)
        self.assertEqual(Face.model_validate_json(face.model_dump_json()),face)

    def test_detector_exports_the_five_full_frame_points(self):
        points=np.array([[35,40],[85,40],[60,65],[42,88],[78,88]])
        detection=np.r_[20,20,80,100,points.flatten(),.95].reshape(1,15)
        class Fake:
            def setInputSize(self,size):pass
            def infer(self,image):return detection
        result=detect_captures(encoded(frame()),Fake())
        self.assertEqual(len(result),1)
        np.testing.assert_allclose(result[0].landmarks.points,points)
        self.assertAlmostEqual(result[0].det_score,.95)
        self.assertEqual(detect_faces(encoded(frame()),Fake()),[result[0].bbox])

    def test_inference_and_training_use_identical_alignment(self):
        jpeg=encoded(frame());points=(TEMPLATE+np.array([25,35])).tolist()
        face=Face(bbox=BoundingBox(x=20,y=20,width=230,height=260),landmarks=FaceLandmarks(points=points))
        with Image.open(io.BytesIO(jpeg)) as image:rgb=np.asarray(image.convert('RGB')).copy()
        expected=prepare_face(rgb,(20,20,230,260),points,ALIGNED)
        class Recognizer:
            crop_mode=ALIGNED
            def predict_emotions(self,crop,logits=False):
                np.testing.assert_array_equal(crop,expected)
                return 'neutral',np.array([0,0,0,0,0,1,0,0],dtype=float)
        with patch('ai_service.analyzer._recognizer',return_value=Recognizer()):
            self.assertIn('neutral',analyze_emotions(jpeg,[face]))
            self.assertIsNone(analyze_emotions(jpeg,[Face(bbox=face.bbox)]))

    def test_enrichment_preserves_split_and_labels_and_input(self):
        import importlib.util
        from common.paths import project_root
        spec=importlib.util.spec_from_file_location('build_landmarks',project_root()/'tools/build_landmarks.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        face=Face(bbox=BoundingBox(x=10,y=10,width=220,height=250),landmarks=FaceLandmarks(points=TEMPLATE.tolist()),det_score=.9)
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'input';out=Path(tmp)/'output';data.mkdir()
            (data/'one.jpg').write_bytes(encoded(frame()));(data/'classes.json').write_text('["anger"]')
            row=dict(id='row1',file='one.jpg',label='anger',emotion='anger',x=10,y=10,width=220,height=250,
                     clip_id='clip1',subject_id='person1',split='val',is_dup=False)
            with (data/'labels.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
            original=(data/'labels.csv').read_bytes()
            with patch.object(module,'get_yunet',return_value=object()),patch.object(module,'detect_captures',return_value=[face]):
                report=module.build(data,out)
            with (out/'labels.csv').open() as f:
                records=list(csv.DictReader(f))
            self.assertEqual(report['accepted'],1)
            for key in ['id','label','split','subject_id','clip_id']:self.assertEqual(records[0][key],str(row[key]))
            self.assertEqual((data/'labels.csv').read_bytes(),original)
            self.assertTrue((out/records[0]['file']).is_file());self.assertTrue((out/records[0]['crop_path']).is_file())
            self.assertEqual(json.loads(records[0]['landmarks']),TEMPLATE.tolist())

if __name__=='__main__':unittest.main()
