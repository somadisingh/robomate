from __future__ import annotations

import os
from pathlib import Path


DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


def load_dotenv(path: Path | None = None) -> None:
    for env_path in _candidate_paths(path):
        if env_path.exists():
            _load_file(env_path)
            return


def openai_model_from_env() -> str:
    return os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


def _candidate_paths(path: Path | None) -> list[Path]:
    if path is not None:
        return [path.expanduser().resolve()]

    package_root = Path(__file__).resolve().parents[1]
    candidates = [Path.cwd() / ".env", package_root / ".env"]
    candidates.extend(parent / ".env" for parent in Path.cwd().resolve().parents)

    unique: list[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _load_file(path: Path) -> None:
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
