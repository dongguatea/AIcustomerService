import pytest
from pydantic import ValidationError

from app.config import Settings


def test_account_fields_required(monkeypatch):
    for k in ("CHAT_MODEL", "CHAT_BASE_URL", "CHAT_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_env_override(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("TOKEN_BUDGET", "500")
    assert Settings(_env_file=None).token_budget == 500


def test_chat_thinking_defaults_to_disabled(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    assert Settings(_env_file=None).chat_thinking == "disabled"


def test_db_and_tool_settings_have_defaults(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    s = Settings(_env_file=None)
    assert s.database_url.startswith("mysql+pymysql://")
    assert "3307" in s.database_url
    assert "charset=utf8mb4" in s.database_url
    assert s.tool_timeout == 10.0
    assert s.tool_max_retries == 2


def test_tool_settings_env_override(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("TOOL_TIMEOUT", "3.5")
    monkeypatch.setenv("TOOL_MAX_RETRIES", "5")
    s = Settings(_env_file=None)
    assert s.tool_timeout == 3.5
    assert s.tool_max_retries == 5


def test_embed_settings_defaults(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k",
                 "EMBED_API_KEY": "ek"}.items():
        monkeypatch.setenv(k, v)
    s = Settings(_env_file=None)
    assert s.embed_base_url == "https://api.siliconflow.cn/v1"
    assert s.embed_model == "BAAI/bge-m3"
    assert s.milvus_uri == "http://localhost:19530"
    assert s.dense_top_k == 5
    assert s.dense_min_score == 0.40


def test_embed_settings_env_override(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("EMBED_API_KEY", "ek")
    monkeypatch.setenv("EMBED_MODEL", "BAAI/bge-m3-custom")
    monkeypatch.setenv("DENSE_TOP_K", "8")
    s = Settings(_env_file=None)
    assert s.embed_model == "BAAI/bge-m3-custom"
    assert s.dense_top_k == 8


def test_embed_api_key_defaults_empty(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    assert Settings(_env_file=None).embed_api_key == ""
