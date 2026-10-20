import asyncio
import shutil
import tempfile
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, status
from pydantic import BaseModel

from common.messages import FaceCaptureMessage

from backend.broker import publisher, rabbit
from backend.config import settings
from backend.frames import store_frame
from backend.video import InvalidVideoError, extract_frames
from backend.webcam import router as webcam_router


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    await rabbit.connect()
    yield
    await rabbit.close()


app = FastAPI(title="AI Live Emotions Decode", lifespan=lifespan)
app.include_router(webcam_router)


class VideoUploadResponse(BaseModel):
    video_id: str
    fps: float
    frames_count: int


def _process_video(path: Path, video_id: str) -> list[FaceCaptureMessage]:
    """Тяжёлая часть (CPU/IO): декодирование, уменьшение и загрузка кадров. Выполняется в отдельном потоке."""
    return [
        store_frame(video_id, frame.index, frame.timestamp_ms, frame.jpeg)
        for frame in extract_frames(
            path,
            target_fps=settings.target_fps,
            max_height=settings.max_frame_height,
            jpeg_quality=settings.jpeg_quality,
        )
    ]


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
        await publisher.publish(frame)

    return VideoUploadResponse(
        video_id=video_id, fps=settings.target_fps, frames_count=len(frames)
    )
