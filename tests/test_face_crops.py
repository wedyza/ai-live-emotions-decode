"""Кропы лиц в cv-service: те же пиксели, что ai-service получил бы из кадра, и корректный обмен через S3."""
import io
import typing
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from common.face_preprocessing import ALIGNED, BBOX, TEMPLATE, ALIGN_SIZE, prepare_face
from common.messages import AnalyzeMessage, BoundingBox, Face, FaceCaptureMessage, FaceLandmarks, FrameRef
from ai_service.analyzer import analyze_emotions
from cv_service.config import Settings as CvSettings
from cv_service.detector import capture_faces


def photo(width=320, height=240, seed=0):
    """Неоднородная картинка: на гладкой заливке расхождения декодеров были бы не видны."""
    rng = np.random.default_rng(seed)
    y, x = np.indices((height, width))
    base = np.stack([x % 256, (y * 2) % 256, (x + y) % 256], axis=-1).astype(np.float64)
    return (base + rng.normal(0, 12, base.shape)).clip(0, 255).astype(np.uint8)


def jpeg_of(rgb, quality=85):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


def pil_rgb(data):
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB")).copy()


# Лицо 120x140 в (100, 50) и пять точек, слегка повёрнутых относительно шаблона.
POINTS = (TEMPLATE / (ALIGN_SIZE - 1) * 110 + np.array([105, 60])) @ np.array([[0.99, 0.1], [-0.1, 0.99]])
DETECTION = np.r_[100, 50, 120, 140, POINTS.flatten(), 0.93].reshape(1, 15)


class FakeYuNet:
    def __init__(self, detections=DETECTION):
        self.detections = detections

    def setInputSize(self, size):
        pass

    def infer(self, image):
        return self.detections


class CaptureFacesTests(unittest.TestCase):
    def test_crop_is_pixel_identical_to_ai_service_cropping(self):
        # cv-service декодирует JPEG через OpenCV, ai-service — через PIL. Если декодеры
        # разойдутся (обновление библиотек), модель начнёт получать другие пиксели.
        jpeg = jpeg_of(photo())
        frame = pil_rgb(jpeg)
        for mode in [ALIGNED, BBOX]:
            with self.subTest(mode=mode):
                (capture,) = capture_faces(jpeg, FakeYuNet(), mode)
                box = capture.face.bbox
                expected = prepare_face(
                    frame, (box.x, box.y, box.width, box.height), capture.face.landmarks.points, mode
                )
                self.assertTrue(capture.crop_png.startswith(b"\x89PNG"))
                np.testing.assert_array_equal(pil_rgb(capture.crop_png), expected)

    def test_aligned_crop_has_model_input_size(self):
        (capture,) = capture_faces(jpeg_of(photo()), FakeYuNet(), ALIGNED)
        self.assertEqual(pil_rgb(capture.crop_png).shape, (ALIGN_SIZE, ALIGN_SIZE, 3))

    def test_face_without_valid_geometry_keeps_detection_without_crop(self):
        degenerate = DETECTION.copy()
        degenerate[0, 4:14] = 150  # все пять точек в одной точке
        (capture,) = capture_faces(jpeg_of(photo()), FakeYuNet(degenerate), ALIGNED)
        self.assertIsNone(capture.crop_png)
        self.assertAlmostEqual(capture.face.det_score, 0.93, places=5)

    def test_crops_disabled(self):
        (capture,) = capture_faces(jpeg_of(photo()), FakeYuNet(), None)
        self.assertIsNone(capture.crop_png)

    def test_undecodable_frame(self):
        self.assertEqual(capture_faces(b"not a jpeg", FakeYuNet(), ALIGNED), [])

    def test_config_modes_are_the_preprocessing_modes(self):
        annotation = CvSettings.model_fields["face_crop"].annotation
        self.assertEqual(set(typing.get_args(annotation)), {ALIGNED, BBOX})
        self.assertEqual(CvSettings.model_fields["face_crop"].default, ALIGNED)


class FakeStorage:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.events = []

    async def get_async(self, key):
        self.events.append(("get", key))
        return self.objects[key]

    async def put_async(self, key, data, content_type="image/jpeg"):
        self.events.append(("put", key, content_type))
        self.objects[key] = data


class FakePublisher:
    def __init__(self, storage):
        self.storage = storage
        self.messages = []

    async def publish(self, message):
        self.storage.events.append(("publish", type(message).__name__))
        self.messages.append(message)


FRAME = FrameRef(video_id="cam1", frame_id="cam1/000007", s3_key="cam1/000007.jpg", timestamp_ms=700)


class FaceCaptureHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def run_handler(self, crop_mode):
        from cv_service.service import FaceCaptureHandler

        storage = FakeStorage({FRAME.s3_key: jpeg_of(photo())})
        publisher = FakePublisher(storage)
        with patch("cv_service.service.get_yunet", return_value=FakeYuNet()):
            handler = FaceCaptureHandler(storage, publisher, crop_mode)
        await handler(FaceCaptureMessage(frame=FRAME))
        return storage, publisher

    async def test_crops_are_stored_before_message_is_published(self):
        storage, publisher = await self.run_handler(ALIGNED)
        self.assertEqual(
            storage.events,
            [("get", FRAME.s3_key), ("put", "cam1/000007_f0.png", "image/png"), ("publish", "AnalyzeMessage")],
        )
        (face,) = publisher.messages[0].faces
        self.assertEqual((face.crop_s3_key, face.crop_mode), ("cam1/000007_f0.png", ALIGNED))
        self.assertIsNotNone(face.landmarks)
        # Сообщение проходит через RabbitMQ как JSON.
        self.assertEqual(AnalyzeMessage.model_validate_json(publisher.messages[0].model_dump_json()), publisher.messages[0])

    async def test_without_crops_only_detections_are_published(self):
        storage, publisher = await self.run_handler(None)
        self.assertEqual(storage.events, [("get", FRAME.s3_key), ("publish", "AnalyzeMessage")])
        self.assertIsNone(publisher.messages[0].faces[0].crop_s3_key)


class Recognizer:
    """Запоминает, какие кропы получила модель."""

    crop_mode = ALIGNED

    def __init__(self):
        self.seen = []

    def predict_emotions(self, crop, logits=False):
        self.seen.append(crop)
        return "neutral", np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=float)


def detected_face():
    return Face(
        bbox=BoundingBox(x=100, y=50, width=120, height=140),
        landmarks=FaceLandmarks(points=POINTS.tolist()),
        det_score=0.9,
    )


class AnalyzerWithCropsTests(unittest.TestCase):
    def test_crop_and_frame_paths_feed_the_model_identical_pixels(self):
        jpeg = jpeg_of(photo())
        (capture,) = capture_faces(jpeg, FakeYuNet(), ALIGNED)
        recognizer = Recognizer()
        with patch("ai_service.analyzer._recognizer", return_value=recognizer):
            from_frame = analyze_emotions(jpeg, [capture.face])
            from_crop = analyze_emotions(None, [capture.face], [capture.crop_png])
        self.assertEqual(from_frame, from_crop)
        np.testing.assert_array_equal(recognizer.seen[0], recognizer.seen[1])

    def test_frame_required_only_for_faces_without_crop(self):
        (capture,) = capture_faces(jpeg_of(photo()), FakeYuNet(), ALIGNED)
        with patch("ai_service.analyzer._recognizer", return_value=Recognizer()):
            with self.assertRaisesRegex(ValueError, "Frame image is required"):
                analyze_emotions(None, [capture.face, detected_face()], [capture.crop_png, None])
            with self.assertRaisesRegex(ValueError, "one-to-one"):
                analyze_emotions(None, [capture.face], [])


class AnalyzeHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def run_handler(self, faces, objects):
        from ai_service.service import AnalyzeHandler

        storage = FakeStorage(objects)
        publisher = FakePublisher(storage)
        recognizer = Recognizer()
        with patch("ai_service.analyzer._recognizer", return_value=recognizer):
            await AnalyzeHandler(storage, publisher)(AnalyzeMessage(frame=FRAME, faces=faces))
        gets = [event[1] for event in storage.events if event[0] == "get"]
        return gets, publisher, recognizer

    def crop(self, mode):
        (capture,) = capture_faces(jpeg_of(photo()), FakeYuNet(), mode)
        face = capture.face.model_copy(update={"crop_s3_key": "cam1/000007_f0.png", "crop_mode": mode})
        return face, {"cam1/000007_f0.png": capture.crop_png, FRAME.s3_key: jpeg_of(photo())}

    async def test_matching_crop_skips_frame_download(self):
        face, objects = self.crop(ALIGNED)
        gets, publisher, recognizer = await self.run_handler([face], objects)
        self.assertEqual(gets, ["cam1/000007_f0.png"])
        self.assertIn("neutral", publisher.messages[0].text)
        self.assertEqual(recognizer.seen[0].shape, (ALIGN_SIZE, ALIGN_SIZE, 3))

    async def test_crop_in_other_mode_falls_back_to_frame(self):
        face, objects = self.crop(BBOX)
        gets, _, recognizer = await self.run_handler([face], objects)
        self.assertEqual(gets, [FRAME.s3_key])
        # Модель получила выровненное лицо из кадра, а не bbox-кроп от cv-service.
        self.assertEqual(recognizer.seen[0].shape, (ALIGN_SIZE, ALIGN_SIZE, 3))

    async def test_old_messages_without_crops_still_work(self):
        gets, publisher, _ = await self.run_handler([detected_face()], {FRAME.s3_key: jpeg_of(photo())})
        self.assertEqual(gets, [FRAME.s3_key])
        self.assertEqual(len(publisher.messages), 1)


if __name__ == "__main__":
    unittest.main()
