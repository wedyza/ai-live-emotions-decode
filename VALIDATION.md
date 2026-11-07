# Проверки переносимости

Независимая проверка: Linux, Python 3.12, новая .venv, uv sync --all-packages --no-dev --locked. Реально загружены комплектные ONNX-веса пользователя; проверены инференс и softmax. Для synthetic inputs нельзя оценить правильность эмоций.

7 unittest checks passed: paths from different cwd; actionable missing-weight failure; real model output; service analyzer without dataset; serialization of shared detector with deliberately concurrent calls; real concurrent YuNet inference with different frame sizes; notebook target/y separation and code compilation.

tools/project.py check passed; bundled hashes match. Notebook code cells compile and contain no saved outputs/private absolute paths. Training dependencies resolved and locked, but training notebook was not re-executed on a dataset in this environment (dataset not supplied with this update).

Full Docker pipeline, Windows and macOS launcher behavior are NOT independently verified here. User earlier verified the prior version's complete pipeline and the new classifier on Mac; that does not replace a fresh acceptance check for this package. Docker executable unavailable in this environment.

Known model limits: inherited video labels, missing classes, macro-F1 clip-bootstrap CI includes zero, possible dependency between same-participant clips. See MODEL_CARD.md.
