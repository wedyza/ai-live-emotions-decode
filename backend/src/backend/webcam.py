import asyncio
import time
import uuid
from importlib.resources import files

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from backend.config import settings
from backend.frames import save_and_publish_frame
from backend.video import InvalidImageError, normalize_jpeg

router = APIRouter(tags=["webcam"])

# Таймеры браузера неточны, поэтому кадр принимается, если пришёл не раньше 80% шага после предыдущего.
_MIN_INTERVAL_RATIO = 0.8


@router.get("/webcam", response_class=HTMLResponse)
async def webcam_page() -> str:
    """Страница, которая снимает веб-камеру в браузере и отправляет кадры в /ws/webcam."""
    html = (files("backend") / "static" / "webcam.html").read_text(encoding="utf-8")
    return (
        html.replace("__TARGET_FPS__", str(settings.target_fps))
        .replace("__MAX_FRAME_HEIGHT__", str(settings.max_frame_height))
        .replace("__JPEG_QUALITY__", str(settings.jpeg_quality))
    )


@router.websocket("/ws/webcam")
async def webcam_stream(websocket: WebSocket) -> None:
    """Принимает JPEG-кадры бинарными сообщениями; каждый сохранённый кадр уходит в S3 и брокер.

    Сообщения сервер -> клиент: {"session_id"} при подключении, затем ровно один ответ на кадр:
    {"frame_id", "timestamp_ms"} — кадр сохранён, {"skipped": true} — превышен лимит fps,
    {"error"} — это не картинка.
    """
    await websocket.accept()
    session_id = uuid.uuid4().hex
    await websocket.send_json({"session_id": session_id})

    min_interval = _MIN_INTERVAL_RATIO / settings.target_fps
    started = time.monotonic()
    last_accepted: float | None = None
    index = 0

    try:
        while True:
            data = await websocket.receive_bytes()
            elapsed = time.monotonic() - started
            if last_accepted is not None and elapsed - last_accepted < min_interval:
                # Клиент шлёт чаще, чем target_fps.
                await websocket.send_json({"skipped": True})
                continue

            try:
                jpeg = await asyncio.to_thread(
                    normalize_jpeg, data, settings.max_frame_height, settings.jpeg_quality
                )
            except InvalidImageError as e:
                await websocket.send_json({"error": str(e)})
                continue

            last_accepted = elapsed
            frame = await save_and_publish_frame(
                session_id, index, round(elapsed * 1000), jpeg
            )
            index += 1
            await websocket.send_json(
                {"frame_id": frame.frame_id, "timestamp_ms": frame.timestamp_ms}
            )
    except WebSocketDisconnect:
        pass
