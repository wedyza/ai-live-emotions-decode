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

Не запускайте одновременно несколько копий Python-сервисов на тех же очередях/портах. Ctrl+C останавливает сервисы, Docker остаётся работать. Для приложения PyTorch и тренировочные изображения не требуются. Данные для повторения FER-экспериментов скачиваются отдельной командой из `CONSENSUS_REPORT_RU.md`. Исторические упоминания «комплектного датасета» в отчётах относятся к архивным сборкам, а не к этой Git-ветке.

Результаты описывают категории видимой мимики, не внутреннее состояние человека. Модели ошибаются; precision среди ответов всегда сопровождается coverage. Main/master этой работой не обновлялись.
