import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


class LocalEmotionModel:
    def __init__(self, path: str):
        model_path = Path(path).expanduser().resolve()
        metadata_path = Path(str(model_path) + ".json")

        if not model_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Model and metadata required: {model_path} and {metadata_path}. "
                "Keep models/emotion_experiment in the checkout, or set "
                "AI_EMOTION_MODEL=pretrained to use the original model."
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.classes = metadata["classes"]
        self.size = int(metadata["img_size"])
        self.mean = np.asarray(metadata["mean"], dtype=np.float32)
        self.std = np.asarray(metadata["std"], dtype=np.float32)

        if self.size <= 0 or len(set(self.classes)) != 8:
            raise ValueError("Invalid image size or duplicate classes")
        if metadata.get("output") != "logits" or metadata.get("batch_size") != 1:
            raise ValueError("Expected batch-one logits model")
        if metadata.get("crop") != "bbox without margin or alignment":
            raise ValueError("Unsupported crop preprocessing")
        if metadata["input"] != "RGB":
            raise ValueError("Expected RGB input")
        if metadata["resize"] != "PIL bilinear":
            raise ValueError("Unsupported resize method")
        if len(self.classes) != 8:
            raise ValueError("Expected 8 expression classes")
        if self.mean.shape != (3,) or self.std.shape != (3,):
            raise ValueError("Invalid normalization metadata")
        if not np.isfinite(self.mean).all():
            raise ValueError("Invalid mean")
        if not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Invalid standard deviation")

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

    def predict_emotions(self, face_img, logits=False):
        if face_img.dtype != np.uint8 or face_img.ndim != 3 or face_img.shape[2] != 3 or not face_img.size:
            raise ValueError("Expected non-empty uint8 RGB face")
        # face_img — RGB-лицо, уже вырезанное по рамке.
        image = Image.fromarray(face_img).convert("RGB")
        image = image.resize(
            (self.size, self.size),
            resample=Image.Resampling.BILINEAR,
        )

        x = np.asarray(image, dtype=np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])

        scores = self.session.run(["logits"], {"input": x})[0][0]

        if scores.shape != (8,) or not np.isfinite(scores).all():
            raise RuntimeError("Invalid model output")

        emotion = self.classes[int(scores.argmax())]

        if not logits:
            scores = np.exp(scores - scores.max())
            scores = scores / scores.sum()

        return emotion, scores
