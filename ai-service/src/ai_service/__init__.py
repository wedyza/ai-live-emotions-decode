def main() -> None:
    import asyncio

    from common.lifecycle import configure_logging

    from ai_service.service import run

    configure_logging()
    asyncio.run(run())
