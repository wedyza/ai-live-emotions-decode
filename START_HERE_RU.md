# Вариант 2 — эксперимент: изображение и точки лица

Ветка `sergej/variant2-experiment` основана на `sergej/variant1-reference`.

```bash
uv sync --all-packages --no-dev --python 3.12 --locked
uv run --no-sync python tools/project.py check --model variant2
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python tools/project.py infra
uv run --no-sync python tools/project.py start --model variant2
```

Откройте http://localhost:8000/docs → POST /videos. Для парного сравнения используйте `--model variant1`. Без параметра сохраняется консервативный reliable; эксперимент включается явно. Веса ONNX включены, PyTorch и тренировочные изображения для запуска не нужны. Требуются Python3.12, uv и Docker для инфраструктуры. Фотографии, виртуальные окружения и секреты исключены из Git.

Читайте `BRANCH_SUMMARY_RU.md` и `VARIANTS_COMPARISON_RU.md`. Преимущество дополнительных координат по имеющемуся независимому контролю не установлено. Основная ветка не изменялась; решение о слиянии принимает команда.
