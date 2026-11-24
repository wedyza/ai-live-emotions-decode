import unittest
from unittest.mock import patch
import numpy as np
from ai_service.reliable_emotion_model import ReliableEmotionModel

class ReliabilityTests(unittest.TestCase):
    def classifier(self):
        m=ReliableEmotionModel.__new__(ReliableEmotionModel)
        m.classes=['anger','contempt','disgust','fear','happiness','neutral','sadness','surprise']
        m.temperature=2.;m.min_score=.7;m.min_margin=.2;m.require_flip_agreement=True
        return m
    def test_disagreement_rejected_even_with_extreme_logits(self):
        m=self.classifier();a=np.zeros(8);b=np.zeros(8);a[4]=100;b[6]=100
        with patch.object(m,'_logits',return_value=(a,b)):
            d=m.predict_decision(np.zeros((32,32,3),np.uint8))
        self.assertFalse(d.accepted);self.assertEqual(d.reason,'view_disagreement')
    def test_uncertain_is_explicit_not_a_forced_expression(self):
        m=self.classifier()
        with patch.object(m,'_logits',return_value=(np.zeros(8),np.zeros(8))):
            label,scores=m.predict_emotions(np.zeros((32,32,3),np.uint8))
        self.assertEqual(label,'uncertain');self.assertAlmostEqual(float(scores.sum()),1.)
    def test_temperature_and_acceptance(self):
        m=self.classifier();a=np.zeros(8);a[4]=10
        with patch.object(m,'_logits',return_value=(a,a)):
            d=m.predict_decision(np.zeros((32,32,3),np.uint8))
        expected=np.exp(5)/(np.exp(5)+7)
        self.assertTrue(d.accepted);self.assertEqual(d.label,'happiness')
        self.assertAlmostEqual(float(d.scores.max()),expected)
    def test_ambiguous_two_classes_rejected_by_margin(self):
        m=self.classifier();m.min_score=.3;a=np.zeros(8);a[4]=10;a[5]=9.9
        with patch.object(m,'_logits',return_value=(a,a)):
            d=m.predict_decision(np.zeros((32,32,3),np.uint8))
        self.assertFalse(d.accepted);self.assertEqual(d.reason,'small_margin')


class PolicyIntegrityTests(unittest.TestCase):
    def test_policy_cannot_silently_use_other_weights(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'model.onnx').write_bytes(b'fixture')
            policy=dict(version=1,model_sha256=hashlib.sha256(b'fixture').hexdigest(),classes=['neutral'],crop='fixture',temperature=2.,min_score=.7,min_margin=.1,require_flip_agreement=True)
            (p/'policy.json').write_text(json.dumps(policy))
            with patch('ai_service.reliable_emotion_model.LocalEmotionModel',return_value=SimpleNamespace(classes=['neutral'],crop_mode='fixture')):
                ReliableEmotionModel(p)
                (p/'model.onnx').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'different weights'):ReliableEmotionModel(p)
    def test_policy_preprocessing_must_match_model(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'model.onnx').write_bytes(b'fixture')
            policy=dict(version=1,model_sha256=hashlib.sha256(b'fixture').hexdigest(),classes=['neutral'],crop='aligned',temperature=2.,min_score=.7,min_margin=.1,require_flip_agreement=True)
            (p/'policy.json').write_text(json.dumps(policy))
            with patch('ai_service.reliable_emotion_model.LocalEmotionModel',return_value=SimpleNamespace(classes=['neutral'],crop_mode='bbox')):
                with self.assertRaisesRegex(ValueError,'metadata mismatch'):ReliableEmotionModel(p)



class AnalyzerDecisionTests(unittest.TestCase):
    def test_rejected_expression_is_not_printed_as_a_label(self):
        import io
        from PIL import Image
        from types import SimpleNamespace
        from ai_service.analyzer import analyze_emotions
        from ai_service.reliable_emotion_model import ExpressionDecision
        from common.messages import Face,BoundingBox
        from common.face_preprocessing import BBOX
        b=io.BytesIO();Image.new('RGB',(96,96)).save(b,'JPEG')
        r=SimpleNamespace(crop_mode=BBOX,predict_decision=lambda crop:ExpressionDecision('sadness',False,np.ones(8)/8,'low_score'))
        with patch('ai_service.analyzer._recognizer',return_value=r):
            answer=analyze_emotions(b.getvalue(),[Face(bbox=BoundingBox(x=0,y=0,width=96,height=96))])
        self.assertIn('uncertain',answer);self.assertNotIn('sadness',answer)
    def test_missing_landmarks_is_explicit_uncertain_for_reliable_model(self):
        import io
        from PIL import Image
        from types import SimpleNamespace
        from ai_service.analyzer import analyze_emotions
        from common.messages import Face,BoundingBox
        from common.face_preprocessing import ALIGNED
        b=io.BytesIO();Image.new('RGB',(96,96)).save(b,'JPEG')
        r=SimpleNamespace(crop_mode=ALIGNED,predict_decision=lambda crop:None)
        with patch('ai_service.analyzer._recognizer',return_value=r):
            answer=analyze_emotions(b.getvalue(),[Face(bbox=BoundingBox(x=0,y=0,width=96,height=96))])
        self.assertIn('invalid_face_geometry',answer)



class ClassThresholdTests(unittest.TestCase):
    classifier=ReliabilityTests.classifier
    def test_class_specific_threshold_overrides_global_cutoff(self):
        m=self.classifier();m.supported_classes=['happiness'];m.min_score=0.
        m.class_thresholds={c:None for c in m.classes};m.class_thresholds['happiness']=.99
        a=np.zeros(8);a[4]=10
        with patch.object(m,'_logits',return_value=(a,a)):
            self.assertFalse(m.predict_decision(np.zeros((8,8,3),np.uint8)).accepted)
            m.class_thresholds['happiness']=.9
            self.assertTrue(m.predict_decision(np.zeros((8,8,3),np.uint8)).accepted)
    def test_unvalidated_class_is_rejected_even_at_extreme_score(self):
        m=self.classifier();m.supported_classes=['happiness'];m.class_thresholds={c:None for c in m.classes};m.class_thresholds['happiness']=.5
        a=np.zeros(8);a[2]=10000
        with patch.object(m,'_logits',return_value=(a,a)):
            d=m.predict_decision(np.zeros((8,8,3),np.uint8))
        self.assertFalse(d.accepted);self.assertEqual(d.reason,'unvalidated_class');self.assertTrue(np.isfinite(d.scores).all())

class Version2IntegrityTests(unittest.TestCase):
    def policy(self,directory,thresholds):
        import hashlib,json
        (directory/'model.onnx').write_bytes(b'fixture')
        value=dict(version=2,model_sha256=hashlib.sha256(b'fixture').hexdigest(),classes=['neutral'],supported_classes=['neutral'],crop='fixture',temperature=1.,min_score=0.,min_margin=0.,require_flip_agreement=True,class_thresholds=thresholds)
        (directory/'policy.json').write_text(json.dumps(value))
    def test_missing_threshold_key_fails(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);self.policy(p,{})
            with patch('ai_service.reliable_emotion_model.LocalEmotionModel',return_value=SimpleNamespace(classes=['neutral'],crop_mode='fixture')):
                with self.assertRaisesRegex(ValueError,'cover every'):ReliableEmotionModel(p)
    def test_supported_classes_must_match_active_thresholds(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);self.policy(p,{'neutral':None})
            with patch('ai_service.reliable_emotion_model.LocalEmotionModel',return_value=SimpleNamespace(classes=['neutral'],crop_mode='fixture')):
                with self.assertRaisesRegex(ValueError,'differ from class'):ReliableEmotionModel(p)

if __name__=="__main__":unittest.main()
