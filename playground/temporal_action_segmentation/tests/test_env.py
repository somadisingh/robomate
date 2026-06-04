from __future__ import annotations

from temporal_action_segmentation.env import DEFAULT_OPENAI_MODEL, load_dotenv, openai_model_from_env


def test_load_dotenv_reads_openai_model_without_overriding_existing_env(monkeypatch, tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_MODEL=gpt-5.4-mini\nOPENAI_API_KEY=example\n")
    monkeypatch.setenv("OPENAI_MODEL", "custom-model")

    load_dotenv(env_path)

    assert openai_model_from_env() == "custom-model"


def test_openai_model_defaults_to_latest_series(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    assert openai_model_from_env() == DEFAULT_OPENAI_MODEL == "gpt-5.4-mini"
