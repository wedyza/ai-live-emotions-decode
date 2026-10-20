import asyncio

from common.messages import FaceCaptureMessage, FrameRef

from backend.broker import publisher
from backend.storage import get_storage


def store_frame(video_id: str, index: int, timestamp_ms: int, jpeg: bytes) -> FaceCaptureMessage:
    frame_id = f"{video_id}/{index:06d}"
    s3_key = f"{frame_id}.jpg"
    get_storage().put(s3_key, jpeg)
    return FaceCaptureMessage(
        frame=FrameRef(
            video_id=video_id, frame_id=frame_id, s3_key=s3_key, timestamp_ms=timestamp_ms
        )
    )


async def save_and_publish_frame(
    video_id: str, index: int, timestamp_ms: int, jpeg: bytes
) -> FrameRef:
    message = await asyncio.to_thread(store_frame, video_id, index, timestamp_ms, jpeg)
    await publisher.publish(message)
    return message.frame
