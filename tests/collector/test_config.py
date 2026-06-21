"""Config: env parsing, secret redaction, derived values."""
from __future__ import annotations

from collector.config import CollectorConfig, load_dotenv


def test_from_env_defaults(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("COLLECTOR_") or k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "DATASET_SLUG"):
            monkeypatch.delenv(k, raising=False)
    cfg = CollectorConfig.from_env(load_env_file=False)
    assert cfg.competition == "pokemon-tcg-ai-battle"
    assert cfg.rps == 0.2
    assert cfg.sink == "local"
    assert cfg.save_value_records is True


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("COLLECTOR_RPS", "1.5")
    monkeypatch.setenv("COLLECTOR_TOP_N", "25")
    monkeypatch.setenv("COLLECTOR_TARGET_TEAMS", "ace, bee ,")
    monkeypatch.setenv("COLLECTOR_KEEP_RAW", "true")
    monkeypatch.setenv("KAGGLE_KEY", "secret")
    cfg = CollectorConfig.from_env(load_env_file=False)
    assert cfg.rps == 1.5
    assert cfg.top_n_leaders == 25
    assert cfg.target_teams == ["ace", "bee"]
    assert cfg.keep_raw is True


def test_min_interval():
    assert CollectorConfig(rps=2.0).min_interval() == 0.5
    assert CollectorConfig(rps=0.0).min_interval() == 0.0


def test_redacted_masks_secrets():
    cfg = CollectorConfig(kaggle_username="alice", kaggle_key="topsecret")
    r = cfg.redacted()
    assert r["kaggle_key"] == "***"
    assert r["kaggle_username"].startswith("al") and "ice" not in r["kaggle_username"]


def test_has_credentials():
    assert not CollectorConfig().has_credentials()
    assert CollectorConfig(kaggle_username="a", kaggle_key="b").has_credentials()


def test_load_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('FOO_BAR="baz"\n# comment\nEMPTY=\nQUUX=1\n')
    monkeypatch.delenv("FOO_BAR", raising=False)
    parsed = load_dotenv(env)
    assert parsed["FOO_BAR"] == "baz"
    assert parsed["QUUX"] == "1"
