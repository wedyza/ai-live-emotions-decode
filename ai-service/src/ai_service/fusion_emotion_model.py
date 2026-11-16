"""Variant 2: ONNX genuinely consumes an image AND ten landmark coordinates."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from ai_service.local_emotion_model import LocalEmotionModel
from ai_service.reliable_emotion_model import ReliableEmotionModel
from common.landmark_features import GEOMETRY_VERSION,validate_geometry,flip_geometry

class LocalFusionModel(LocalEmotionModel):
    def __init__(self,path):
        super().__init__(path)
        meta=json.loads(Path(str(Path(path).resolve())+'.json').read_text(encoding='utf-8'))
        if meta.get('model_variant')!='variant2' or meta.get('geometry_norm')!=GEOMETRY_VERSION or meta.get('geometry_size')!=10:
            raise ValueError('Variant 2 requires declared landmark feature preprocessing')
        inputs={v.name:v for v in self.session.get_inputs()}
        if set(inputs)!= {'image','landmarks'} or inputs['image'].shape!=[1,3,self.size,self.size] or inputs['landmarks'].shape!=[1,10]:
            raise ValueError('Expected image [1,3,H,W] and landmarks [1,10] ONNX inputs')
        if any(v.type!='tensor(float)' for v in inputs.values()):raise ValueError('Expected float32 fusion inputs')
    def predict_emotions(self,face_img,geometry=None,logits=False):
        if geometry is None:raise ValueError('Variant 2 requires landmark coordinates; no fallback')
        g=validate_geometry(geometry)
        if not isinstance(face_img,np.ndarray) or face_img.dtype!=np.uint8 or face_img.ndim!=3 or face_img.shape[2]!=3 or not face_img.size:
            raise ValueError('Expected non-empty uint8 RGB face')
        image=Image.fromarray(face_img).resize((self.size,self.size),Image.Resampling.BILINEAR)
        x=(np.asarray(image,np.float32)/255-self.mean)/self.std;x=np.ascontiguousarray(x.transpose(2,0,1)[None])
        scores=self.session.run(['logits'],{'image':x,'landmarks':g[None]})[0][0]
        if scores.shape!=(8,) or not np.isfinite(scores).all():raise RuntimeError('Invalid fusion logits')
        label=self.classes[int(scores.argmax())]
        if not logits:
            scores=np.exp(scores-scores.max());scores/=scores.sum()
        return label,scores

class ReliableFusionModel(ReliableEmotionModel):
    requires_geometry=True
    def __init__(self,directory):
        super().__init__(directory,model_class=LocalFusionModel)
        policy=json.loads((Path(directory)/'policy.json').read_text(encoding='utf-8'))
        if policy.get('geometry_norm')!=GEOMETRY_VERSION or policy.get('model_variant')!='variant2':
            raise ValueError('Fusion policy geometry metadata mismatch')
    def _fusion_logits(self,face_img,geometry):
        g=validate_geometry(geometry)
        _,a=self.model.predict_emotions(face_img,g,logits=True)
        _,b=self.model.predict_emotions(np.ascontiguousarray(face_img[:,::-1]),flip_geometry(g),logits=True)
        return np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    def predict_decision(self,face_img,geometry=None):
        if geometry is None:raise ValueError('Variant 2 requires normalized landmarks')
        return self._decision_from_logits(*self._fusion_logits(face_img,geometry))
    def predict_emotions(self,face_img,geometry=None,logits=False):
        if geometry is None:raise ValueError('Variant 2 requires normalized landmarks')
        if logits:
            a,b=self._fusion_logits(face_img,geometry);values=(a+b)/2
            return self.classes[int(values.argmax())],values
        result=self.predict_decision(face_img,geometry)
        return result.label if result.accepted else 'uncertain',result.scores
