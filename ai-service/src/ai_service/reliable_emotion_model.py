"""Calibrated selective classifier. Abstention is a result, never a hidden failure."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from ai_service.local_emotion_model import LocalEmotionModel


@dataclass(frozen=True)
class ExpressionDecision:
    label: str
    accepted: bool
    scores: np.ndarray
    reason: str | None


class ReliableEmotionModel:
    def __init__(self, directory: str | Path):
        directory = Path(directory).resolve()
        policy = json.loads((directory / 'policy.json').read_text(encoding='utf-8'))
        path = directory / 'model.onnx'
        if policy.get('version') not in [1,2]:
            raise ValueError('Unsupported reliability policy version')
        if hashlib.sha256(path.read_bytes()).hexdigest() != policy['model_sha256']:
            raise ValueError('Reliability policy belongs to different weights')
        self.model = LocalEmotionModel(str(path))
        self.classes = self.model.classes
        self.crop_mode = self.model.crop_mode
        if self.classes != policy['classes'] or self.crop_mode != policy['crop']:
            raise ValueError('Reliability policy/model metadata mismatch')
        self.supported_classes = policy.get('supported_classes',self.classes)
        if not self.supported_classes or not set(self.supported_classes).issubset(self.classes):
            raise ValueError('Invalid supported classes')
        self.class_thresholds = policy.get('class_thresholds') if policy['version']==2 else None
        if policy['version']==2:
            if not isinstance(self.class_thresholds,dict) or set(self.class_thresholds)!=set(self.classes):
                raise ValueError('Class thresholds must cover every output class')
            for value in self.class_thresholds.values():
                if value is not None and (not isinstance(value,(int,float)) or not np.isfinite(value) or not 0<=value<=1):
                    raise ValueError('Invalid class-specific threshold')
            active={c for c,v in self.class_thresholds.items() if v is not None}
            if active!=set(self.supported_classes):
                raise ValueError('Supported classes differ from class thresholds')
        self.temperature = float(policy['temperature'])
        self.min_score = float(policy['min_score'])
        self.min_margin = float(policy['min_margin'])
        self.require_flip_agreement = policy['require_flip_agreement']
        if not np.isfinite([self.temperature,self.min_score,self.min_margin]).all():
            raise ValueError('Non-finite policy parameter')
        if self.temperature <= 0 or not 0 <= self.min_score <= 1 or not 0 <= self.min_margin <= 1:
            raise ValueError('Invalid policy parameter')
        if self.require_flip_agreement is not True:
            raise ValueError('This policy requires horizontal-flip consistency')

    def _logits(self, face_img):
        _, original = self.model.predict_emotions(face_img, logits=True)
        _, flipped = self.model.predict_emotions(np.ascontiguousarray(face_img[:, ::-1]), logits=True)
        return np.asarray(original,dtype=np.float64),np.asarray(flipped,dtype=np.float64)

    def predict_decision(self, face_img) -> ExpressionDecision:
        original, flipped = self._logits(face_img)
        logits = (original + flipped) / (2 * self.temperature)
        scores = np.exp(logits - logits.max())
        scores /= scores.sum()
        ordered = np.sort(scores)
        label = self.classes[int(scores.argmax())]
        reason = None
        thresholds=getattr(self,'class_thresholds',None)
        required=self.min_score if thresholds is None else thresholds.get(label)
        if label not in getattr(self,'supported_classes',self.classes) or required is None:
            reason = 'unvalidated_class'
        elif original.argmax() != flipped.argmax():
            reason = 'view_disagreement'
        elif ordered[-1] < max(self.min_score,required):
            reason = 'low_score'
        elif ordered[-1] - ordered[-2] < self.min_margin:
            reason = 'small_margin'
        return ExpressionDecision(label,reason is None,scores,reason)

    def predict_emotions(self, face_img, logits=False):
        if logits:
            original, flipped = self._logits(face_img)
            values = (original + flipped) / 2
            return self.classes[int(values.argmax())],values
        decision = self.predict_decision(face_img)
        return decision.label if decision.accepted else 'uncertain',decision.scores
