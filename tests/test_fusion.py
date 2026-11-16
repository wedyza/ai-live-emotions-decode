import io,unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from common.landmark_features import landmark_features,flip_geometry,validate_geometry
from common.messages import Face,FaceLandmarks,BoundingBox
from common.face_preprocessing import ALIGNED,prepare_face
from ai_service.analyzer import analyze_emotions
from ai_service.reliable_emotion_model import ExpressionDecision

POINTS=np.array([[20,30],[60,30],[43,47],[25,64],[57,65]],dtype=float)

class LandmarkFeatureTests(unittest.TestCase):
    def test_translation_rotation_scale_invariance(self):
        a=.4;R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        np.testing.assert_allclose(landmark_features(POINTS),landmark_features(POINTS@R.T*2+[350,-50]),atol=1e-6)
    def test_reflection_swaps_eye_and_mouth_semantics(self):
        p=POINTS.copy();p[:,0]=99-p[:,0];p=p[[1,0,2,4,3]]
        np.testing.assert_allclose(flip_geometry(landmark_features(POINTS)),landmark_features(p),atol=1e-6)
    def test_double_reflection_is_identity(self):
        g=landmark_features(POINTS);np.testing.assert_allclose(flip_geometry(flip_geometry(g)),g,atol=1e-7)
    def test_different_actual_points_produce_different_features(self):
        p=POINTS.copy();p[3:,1]+=5
        self.assertGreater(float(np.abs(landmark_features(p)-landmark_features(POINTS)).max()),.1)
    def test_missing_and_raw_coordinates_rejected(self):
        with self.assertRaises(ValueError):landmark_features(None)
        with self.assertRaises(ValueError):validate_geometry(POINTS.reshape(-1))
        with self.assertRaises(ValueError):landmark_features(np.full((5,2),10.))
    def test_degenerate_normalized_geometry_rejected(self):
        with self.assertRaises(ValueError):validate_geometry(np.array([-.5,0,.5,0,0,0,-.2,0,.2,0],np.float32))
    def test_analyzer_passes_actual_normalized_points_with_photo(self):
        b=io.BytesIO();Image.new('RGB',(100,100),(100,80,60)).save(b,'JPEG');rgb=np.asarray(Image.open(io.BytesIO(b.getvalue())).convert('RGB')).copy();face=Face(bbox=BoundingBox(x=10,y=15,width=70,height=65),landmarks=FaceLandmarks(points=POINTS.tolist()))
        expected_image=prepare_face(rgb,(10,15,70,65),POINTS,ALIGNED)
        class Recognizer:
            requires_geometry=True;crop_mode=ALIGNED
            def predict_decision(self,image,g):
                np.testing.assert_array_equal(image,expected_image);np.testing.assert_allclose(g,landmark_features(POINTS));return ExpressionDecision('neutral',True,np.array([0,0,0,0,0,1,0,0]),None)
        with patch('ai_service.analyzer._recognizer',return_value=Recognizer()):self.assertIn('neutral',analyze_emotions(b.getvalue(),[face]))

class ActualFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from common.paths import project_root
        from ai_service.fusion_emotion_model import ReliableFusionModel
        cls.model=ReliableFusionModel(project_root()/'models/emotion_variant2')
    def test_two_actual_onnx_inputs_and_nonzero_geometry_effect(self):
        self.assertEqual({i.name for i in self.model.model.session.get_inputs()},{'image','landmarks'})
        image=np.full((224,224,3),128,np.uint8);g=landmark_features(POINTS);changed=g.copy();changed[5:]+=np.array([.1,0,.1,0,.1],dtype=np.float32)
        a=self.model.model.predict_emotions(image,g,logits=True)[1];b=self.model.model.predict_emotions(image,changed,logits=True)[1]
        self.assertGreater(float(np.abs(a-b).max()),1e-6)
    def test_no_geometry_fallback(self):
        with self.assertRaises(ValueError):self.model.predict_decision(np.zeros((224,224,3),np.uint8))
    def test_probability_and_decision_output(self):
        d=self.model.predict_decision(np.full((224,224,3),128,np.uint8),landmark_features(POINTS));self.assertTrue(np.isfinite(d.scores).all());self.assertAlmostEqual(float(d.scores.sum()),1.,places=6)

if __name__=='__main__':unittest.main()
