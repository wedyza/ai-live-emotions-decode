"""Paths relative to a source checkout, never to the caller's working directory."""
from pathlib import Path


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "ai-service").is_dir():
            return parent
    raise RuntimeError("Project checkout not found; install with uv sync from the repository")


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()
