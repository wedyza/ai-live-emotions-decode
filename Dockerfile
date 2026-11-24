# syntax=docker/dockerfile:1
# Один Dockerfile на все Python-сервисы workspace. Какой сервис собирать — build-arg PACKAGE
# (имя пакета = имя папки = имя консольной команды): backend, cv-service, ai-service, notification-service.
#
#   docker build --build-arg PACKAGE=cv-service --target app -t emotions/cv-service .
#   docker build --build-arg PACKAGE=ai-service --target ai-service -t emotions/ai-service .
#
# Пакеты ставятся editable, с той же структурой папок, что в репозитории: common.paths.project_root()
# ищет корень по pyproject.toml и папке ai-service, а веса лежат по путям относительно корня.

ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:0.12.21 AS uv

# ---------- сборка .venv ----------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
ARG PACKAGE
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# uv читает весь workspace, поэтому нужны pyproject.toml всех участников; README — для метаданных пакетов.
COPY pyproject.toml uv.lock ./
COPY libs/common/pyproject.toml libs/common/README.md libs/common/
COPY backend/pyproject.toml backend/README.md backend/
COPY cv-service/pyproject.toml cv-service/README.md cv-service/
COPY ai-service/pyproject.toml ai-service/README.md ai-service/
COPY notification-service/pyproject.toml notification-service/README.md notification-service/

# Сторонние зависимости отдельным слоем: он пересобирается только при изменении uv.lock.
RUN --mount=type=cache,target=/root/.cache/uv \
    test -n "$PACKAGE" || { echo "build-arg PACKAGE is required" >&2; exit 1; }; \
    uv sync --frozen --no-dev --no-install-workspace --package "$PACKAGE"

COPY libs/common/src libs/common/src
COPY ${PACKAGE}/src ${PACKAGE}/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package "$PACKAGE"

# ---------- рантайм ----------
FROM python:${PYTHON_VERSION}-slim-bookworm AS app
ARG PACKAGE
# opencv-python (не headless, так в uv.lock) при импорте требует libGL и glib, а libGL в Debian
# тянет mesa и LLVM (~250 МБ). cv2 импортируют только cv-service и ai-service (в common он ленивый),
# поэтому остальным образам эти библиотеки не ставим.
RUN case "$PACKAGE" in \
        cv-service|ai-service) \
            apt-get update \
            && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
            && rm -rf /var/lib/apt/lists/* ;; \
    esac \
    && useradd --system --no-create-home --uid 10001 app
WORKDIR /app
COPY --from=builder /app /app
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    APP=${PACKAGE}
USER app
# exec: процесс сервиса становится PID 1 и сам получает SIGTERM от docker stop (корректная остановка consumer'ов).
CMD ["sh", "-c", "exec \"$APP\""]

# ai-service: дополнительно веса моделей эмоций (~100 МБ), остальным образам они не нужны.
FROM app AS ai-service
COPY models models
