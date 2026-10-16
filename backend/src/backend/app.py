import asyncio
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, status
from pydantic import BaseModel

from backend.broker import Broker
from backend.config import settings
from backend.storage import FrameStorage
from backend.video import InvalidVideoError, extract_frames

broker = Broker()
storage = FrameStorage()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await broker.connect()
    yield
    await broker.close()


app = FastAPI(title="AI Live Emotions Decode", lifespan=lifespan)


class StoredFrame(BaseModel):
    frame_id: str
    s3_key: str
    timestamp_ms: int


class VideoUploadResponse(BaseModel):
    video_id: str
    fps: float
    frames_count: int


def _process_video(path: Path, video_id: str) -> list[StoredFrame]:
    """CPU/IO-bound part: decode, downscale and upload frames. Runs in a worker thread."""
    stored = []
    for frame in extract_frames(
        path,
        target_fps=settings.target_fps,
        max_height=settings.max_frame_height,
        jpeg_quality=settings.jpeg_quality,
    ):
        frame_id = f"{video_id}/{frame.index:06d}"
        s3_key = f"{frame_id}.jpg"
        storage.put_jpeg(s3_key, frame.jpeg)
        stored.append(
            StoredFrame(frame_id=frame_id, s3_key=s3_key, timestamp_ms=frame.timestamp_ms)
        )
    return stored


@app.post(
    "/videos",
    response_model=VideoUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_video(file: UploadFile) -> VideoUploadResponse:
    video_id = uuid.uuid4().hex
    suffix = Path(file.filename or "").suffix

    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        await asyncio.to_thread(shutil.copyfileobj, file.file, tmp)
        tmp.flush()
        try:
            frames = await asyncio.to_thread(_process_video, Path(tmp.name), video_id)
        except InvalidVideoError as e:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e

    if not frames:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "video has no frames")

    for frame in frames:
        await broker.publish(
            settings.face_capture_queue,
            {"video_id": video_id, **frame.model_dump()},
        )

    return VideoUploadResponse(
        video_id=video_id, fps=settings.target_fps, frames_count=len(frames)
    )
