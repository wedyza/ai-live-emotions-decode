# Вариант 1 — эталонная архитектура

Ветка `sergej/variant1-reference`. Начните с `BRANCH_SUMMARY_RU.md`.

Точки YuNet используются для выравнивания; классификатор получает только RGB-фотографию. В этой ветке нет второго числового входа, Fusion-модели и MLP координат. Веса runtime включены. Фотографии, датасеты, виртуальные окружения и секреты в Git не загружаются.

```bash
uv sync --all-packages --no-dev --python 3.12 --locked
uv run --no-sync python tools/project.py check --model variant1
uv run --no-sync python tools/project.py infra
uv run --no-sync python tools/project.py start --model variant1
```

Откройте http://localhost:8000/docs → POST /videos. Для более осторожного профиля используйте `--model reliable`; default сохранён reliable. Для контролируемого сравнения архитектур `variant1` является псевдонимом последней image-only модели expanded.

```bash
uv run --no-sync python -m unittest discover -s tests -v
```

## Запуск целиком в Docker

Все сервисы и инфраструктура одной командой, без локального `uv`:

```bash
docker compose up -d --build --wait
```

- Вебкамера: http://localhost:8000/webcam, API: http://localhost:8000/docs
- MinIO: http://localhost:9001 (бакет `frames`: кадры `*.jpg` и кропы лиц `*_fN.png`), RabbitMQ: http://localhost:15672
- Логи: `docker compose logs -f cv-service ai-service notification-service`

Образы собираются из одного `Dockerfile` (build-arg `PACKAGE`), веса моделей есть только в образе ai-service.
Настройки передаются переменными окружения из `docker-compose.yaml`; файл `.env` в образ не попадает,
но compose берёт из него значения для подстановки (`AI_EMOTION_MODEL`, `CV_FACE_CROP`, логины MinIO/RabbitMQ).
Для смены модели задайте, например, `AI_EMOTION_MODEL=experimental` и `CV_FACE_CROP="bbox without margin or alignment"`
и выполните `docker compose up -d`.

`tools/project.py infra` поднимает только инфраструктуру. Не запускайте одновременно `tools/project.py start`
и контейнеры сервисов: они делят очереди и порт 8000 (`docker compose stop backend cv-service ai-service notification-service`).

Не запускайте одновременно несколько копий Python-сервисов на тех же очередях/портах. Ctrl+C останавливает сервисы, Docker остаётся работать. Для приложения PyTorch и тренировочные изображения не требуются. Данные для повторения FER-экспериментов скачиваются отдельной командой из `CONSENSUS_REPORT_RU.md`. Исторические упоминания «комплектного датасета» в отчётах относятся к архивным сборкам, а не к этой Git-ветке.

Результаты описывают категории видимой мимики, не внутреннее состояние человека. Модели ошибаются; precision среди ответов всегда сопровождается coverage. Main/master этой работой не обновлялись.
