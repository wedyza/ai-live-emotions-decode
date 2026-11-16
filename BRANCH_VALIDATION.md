# Проверки отдельной экспериментальной Git-ветки

- Чистое окружение Python3.12: `uv sync --all-packages --no-dev --python 3.12 --locked` — успешно.
- `python -m unittest discover -s tests -v` — 36 тестов, OK.
- `python tools/project.py check --model variant2` — checksums OK, ReliableFusionModel, synthetic inference OK. Это проверка выполнения, не оценка точности.
- Историческое сравнение с PyTorch:20 изображений,40 отражённых представлений, максимальное расхождение около5.25e-6; протокол в reports/variants.
- Парный контроль качества и ограничения — VARIANTS_COMPARISON_RU.md. Превосходство геометрии не установлено.
- Текущая проверка не включает полный Docker/HTTP/RabbitMQ запуск, Windows или macOS.
