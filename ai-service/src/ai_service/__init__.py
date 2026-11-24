import os

# Disable ONNX Runtime telemetry before any model runtime is imported.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

def main() -> None:
    import asyncio

    from common.lifecycle import configure_logging

    from ai_service.service import run

    configure_logging()
    asyncio.run(run())
