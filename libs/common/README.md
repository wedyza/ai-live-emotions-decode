# common

Shared code for all services:

- `settings` — `RabbitSettings`, `S3Settings` to nest into a service's own `BaseSettings`.
- `messages` — message contracts; each model knows its queue (`QUEUE`).
- `broker` — `RabbitConnection`, `Publisher`, `Consumer` (prefetch, ack/reject, dead-letter queue).
- `storage` — `S3Storage` (MinIO / S3).
- `lifecycle` — `wait_for_shutdown()` for SIGINT / SIGTERM.

Library code never reads global settings: everything is passed into constructors.
