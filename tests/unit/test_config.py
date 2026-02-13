"""Tests for configuration loading.

Tests config.yaml loading, dataclass construction, env var overrides,
and error handling for missing required values.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from prism.config import (
    ConfigError,
    ModelConfig,
    get_config,
    reset_config,
)


class TestConfigLoading:
    """Test that config.yaml loads into correct dataclasses."""

    def test_loads_server_config(self) -> None:
        config = get_config()
        assert config.server.port == 8765
        assert config.server.transport == "streamable-http"
        assert config.server.log_level == "INFO"

    def test_loads_search_config(self) -> None:
        config = get_config()
        assert config.search.max_query_length == 10000

    def test_loads_all_levels(self) -> None:
        config = get_config()
        assert set(config.levels.keys()) == {0, 1, 2, 3}

    def test_l0_has_no_agent_allocation(self) -> None:
        config = get_config()
        assert config.levels[0].agent_allocation is None

    def test_l1_agent_allocation(self) -> None:
        config = get_config()
        alloc = config.levels[1].agent_allocation
        assert alloc is not None
        assert alloc == {
            "claude_search": 1,
            "gemini_search": 1,
            "tavily_search": 1,
            "perplexity_search": 1,
        }

    def test_l2_agent_allocation(self) -> None:
        config = get_config()
        alloc = config.levels[2].agent_allocation
        assert alloc is not None
        assert alloc == {
            "claude_search": 2,
            "gemini_search": 2,
            "tavily_search": 1,
            "perplexity_search": 1,
        }

    def test_l3_agent_allocation(self) -> None:
        config = get_config()
        alloc = config.levels[3].agent_allocation
        assert alloc is not None
        assert alloc == {
            "claude_search": 3,
            "gemini_search": 2,
            "tavily_search": 2,
            "perplexity_search": 1,
        }

    def test_no_manager_timeout_in_levels(self) -> None:
        config = get_config()
        for level_config in config.levels.values():
            assert not hasattr(level_config, "manager_timeout_seconds")

    def test_loads_retry_config(self) -> None:
        config = get_config()
        assert config.retry.max_transient_retries == 3
        assert config.retry.base_delay_seconds == 1.0
        assert config.retry.max_delay_seconds == 8.0
        assert config.retry.exponential_base == 2.0
        assert config.retry.max_validation_retries == 5

    def test_loads_process_config(self) -> None:
        config = get_config()
        assert config.process.sigterm_timeout_seconds == 2.0
        assert config.process.sigkill_timeout_seconds == 5.0

    def test_loads_database_config(self) -> None:
        config = get_config()
        assert "postgresql" in config.database.url
        assert config.database.pool_size == 20
        assert config.database.max_overflow == 10

    def test_loads_retention_config(self) -> None:
        config = get_config()
        assert config.retention.session_ttl_days == 30

    def test_no_synthesizer_config(self) -> None:
        config = get_config()
        assert not hasattr(config, "synthesizer")

    def test_no_workers_config(self) -> None:
        config = get_config()
        assert not hasattr(config, "workers")


class TestModelsConfig:
    """Test the new models section."""

    def test_loads_session_manager_models(self) -> None:
        config = get_config()
        sm = config.models.session_manager
        assert set(sm.keys()) == {1, 2, 3}
        assert sm[1] == ModelConfig(model="opus", effort="low")
        assert sm[2] == ModelConfig(model="opus", effort="low")
        assert sm[3] == ModelConfig(model="opus", effort=None)

    def test_loads_claude_workers_models(self) -> None:
        config = get_config()
        cw = config.models.claude_workers
        assert set(cw.keys()) == {0, 1, 2, 3}
        assert cw[0] == ModelConfig(model="haiku", effort=None)
        assert cw[1] == ModelConfig(model="opus", effort="low")
        assert cw[2] == ModelConfig(model="opus", effort="low")
        assert cw[3] == ModelConfig(model="opus", effort=None)

    def test_loads_gemini_workers_models(self) -> None:
        config = get_config()
        gw = config.models.gemini_workers
        assert set(gw.keys()) == {0, 1, 2, 3}
        assert gw[0] == ModelConfig(model="gemini-3-flash-preview", effort=None)
        assert gw[1] == ModelConfig(model="gemini-3-flash-preview", effort=None)
        assert gw[2] == ModelConfig(model="gemini-3-pro-preview", effort=None)
        assert gw[3] == ModelConfig(model="gemini-3-pro-preview", effort=None)


class TestLevel0Config:
    """Test the level0 section."""

    def test_loads_default_providers(self) -> None:
        config = get_config()
        assert config.level0.default_providers == ["claude_search"]


class TestEnvOverrides:
    """Test environment variable overrides."""

    def test_override_server_port(self) -> None:
        os.environ["PRISM_SERVER_PORT"] = "9999"
        try:
            config = get_config()
            assert config.server.port == 9999
        finally:
            del os.environ["PRISM_SERVER_PORT"]

    def test_override_worker_timeout(self) -> None:
        os.environ["PRISM_LEVELS_0_WORKER_TIMEOUT_SECONDS"] = "120"
        try:
            config = get_config()
            assert config.levels[0].worker_timeout_seconds == 120
        finally:
            del os.environ["PRISM_LEVELS_0_WORKER_TIMEOUT_SECONDS"]

    def test_override_model(self) -> None:
        os.environ["PRISM_MODELS_CLAUDE_WORKERS_0_MODEL"] = "sonnet"
        try:
            config = get_config()
            assert config.models.claude_workers[0].model == "sonnet"
        finally:
            del os.environ["PRISM_MODELS_CLAUDE_WORKERS_0_MODEL"]

    def test_override_float_value(self) -> None:
        os.environ["PRISM_RETRY_BASE_DELAY_SECONDS"] = "2.5"
        try:
            config = get_config()
            assert config.retry.base_delay_seconds == 2.5
        finally:
            del os.environ["PRISM_RETRY_BASE_DELAY_SECONDS"]


class TestConfigCaching:
    """Test singleton behavior."""

    def test_returns_same_instance(self) -> None:
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reset_clears_cache(self) -> None:
        config1 = get_config()
        reset_config()
        config2 = get_config()
        assert config1 is not config2


class TestConfigErrors:
    """Test error handling for invalid/missing config."""

    def test_missing_config_file_raises(self, tmp_path: Path) -> None:
        os.environ["PRISM_CONFIG_PATH"] = str(tmp_path / "nonexistent.yaml")
        try:
            reset_config()
            with pytest.raises(FileNotFoundError, match="PRISM_CONFIG_PATH"):
                get_config()
        finally:
            del os.environ["PRISM_CONFIG_PATH"]

    def test_missing_section_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(textwrap.dedent("""\
            server:
              port: 8765
              transport: streamable-http
              log_level: INFO
        """))
        os.environ["PRISM_CONFIG_PATH"] = str(config_file)
        try:
            reset_config()
            with pytest.raises(ConfigError, match="search"):
                get_config()
        finally:
            del os.environ["PRISM_CONFIG_PATH"]

    def test_missing_model_field_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "bad_models.yaml"
        config_file.write_text(textwrap.dedent("""\
            server:
              port: 8765
              transport: streamable-http
              log_level: INFO
            search:
              max_query_length: 10000
            levels:
              0:
                worker_timeout_seconds: 70
                worker_visible_timeout: 30
            models:
              session_manager: {}
              claude_workers:
                0:
                  not_model: haiku
              gemini_workers: {}
            level0:
              default_providers: [claude_search]
            retry:
              max_transient_retries: 3
              base_delay_seconds: 1.0
              max_delay_seconds: 8.0
              exponential_base: 2.0
              max_validation_retries: 5
            process:
              sigterm_timeout_seconds: 2.0
              sigkill_timeout_seconds: 5.0
            database:
              url: sqlite+aiosqlite://
              pool_size: 5
              max_overflow: 0
            retention:
              session_ttl_days: 7
        """))
        os.environ["PRISM_CONFIG_PATH"] = str(config_file)
        try:
            reset_config()
            with pytest.raises(ConfigError, match="model"):
                get_config()
        finally:
            del os.environ["PRISM_CONFIG_PATH"]


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_with_effort(self) -> None:
        mc = ModelConfig(model="opus", effort="low")
        assert mc.model == "opus"
        assert mc.effort == "low"

    def test_without_effort(self) -> None:
        mc = ModelConfig(model="haiku")
        assert mc.model == "haiku"
        assert mc.effort is None

    def test_equality(self) -> None:
        a = ModelConfig(model="opus", effort="low")
        b = ModelConfig(model="opus", effort="low")
        assert a == b
        assert ModelConfig(model="opus") != ModelConfig(model="haiku")


class TestRetryConfigDelay:
    """Test RetryConfig.get_delay calculation."""

    def test_exponential_backoff(self) -> None:
        from prism.config import RetryConfig

        rc = RetryConfig(
            max_transient_retries=3,
            base_delay_seconds=1.0,
            max_delay_seconds=8.0,
            exponential_base=2.0,
            max_validation_retries=5,
        )
        assert rc.get_delay(0) == 1.0
        assert rc.get_delay(1) == 2.0
        assert rc.get_delay(2) == 4.0
        assert rc.get_delay(3) == 8.0

    def test_delay_capped_at_max(self) -> None:
        from prism.config import RetryConfig

        rc = RetryConfig(
            max_transient_retries=3,
            base_delay_seconds=1.0,
            max_delay_seconds=8.0,
            exponential_base=2.0,
            max_validation_retries=5,
        )
        assert rc.get_delay(10) == 8.0
