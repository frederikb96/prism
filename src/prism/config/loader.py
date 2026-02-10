"""
Centralized Configuration for Prism.

Loads configuration from YAML files with environment variable overrides.
Loading order: env vars > prism.yaml (XDG override) > config.yaml (defaults)

All values MUST be defined in config.yaml - no hardcoded defaults here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

DEFAULT_CONFIG_PATH = Path("/app/config/config.yaml")
OVERRIDE_CONFIG_PATH = Path.home() / ".config" / "prism" / "prism.yaml"


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


def _require(cfg: dict[str, Any], key: str, path: str) -> Any:
    """
    Get a required config value, raising clear error if missing.

    Args:
        cfg: Config dict to read from
        key: Key to look up
        path: Full path for error message (e.g., "server.port")

    Raises:
        ConfigError: If key is missing
    """
    if key not in cfg:
        raise ConfigError(
            f"Missing required config: '{path}'. "
            f"Ensure config.yaml contains this value."
        )
    return cfg[key]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """
    Recursively merge override dict into base dict.

    Override values take precedence. Modifies base in-place.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _get_env_override(prefix: str, config: dict[str, Any]) -> None:
    """
    Apply environment variable overrides to config dict.

    Env var naming: PRISM_<SECTION>_<KEY> in uppercase with underscores.
    Examples:
        server.port -> PRISM_SERVER_PORT
        retry.max_transient_retries -> PRISM_RETRY_MAX_TRANSIENT_RETRIES
        levels.1.worker_timeout_seconds -> PRISM_LEVELS_1_WORKER_TIMEOUT_SECONDS
    """
    for key, value in config.items():
        env_key = f"{prefix}_{key}".upper()

        if isinstance(value, dict):
            _get_env_override(env_key, value)
        else:
            env_value = os.environ.get(env_key)
            if env_value is not None:
                if isinstance(value, bool):
                    config[key] = env_value.lower() in ("true", "1", "yes", "on")
                elif isinstance(value, int):
                    try:
                        config[key] = int(env_value)
                    except ValueError:
                        pass
                elif isinstance(value, float):
                    try:
                        config[key] = float(env_value)
                    except ValueError:
                        pass
                else:
                    config[key] = env_value


def _load_config() -> dict[str, Any]:
    """
    Load and merge configuration from files and environment.

    Loading order:
        1. Load default config (config.yaml)
        2. Merge override config (prism.yaml) if exists
        3. Apply environment variable overrides
    """
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH) as f:
            config: dict[str, Any] = yaml.safe_load(f) or {}
    else:
        local_config = Path(__file__).parent.parent.parent.parent / "config" / "config.yaml"
        if local_config.exists():
            with open(local_config) as f:
                config = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(
                f"Config not found at {DEFAULT_CONFIG_PATH} or {local_config}"
            )

    if OVERRIDE_CONFIG_PATH.exists():
        with open(OVERRIDE_CONFIG_PATH) as f:
            override: dict[str, Any] = yaml.safe_load(f) or {}
        _deep_merge(config, override)

    _get_env_override("PRISM", config)

    return config


_CONFIG: dict[str, Any] = {}


def _ensure_config() -> dict[str, Any]:
    """Ensure config is loaded, return the config dict."""
    global _CONFIG
    if not _CONFIG:
        _CONFIG = _load_config()
    return _CONFIG


TransportType = Literal["stdio", "http", "sse", "streamable-http"]


@dataclass
class ServerConfig:
    """MCP Server configuration."""

    port: int
    transport: TransportType
    log_level: str


@dataclass
class SearchConfig:
    """Search parameters configuration."""

    max_query_length: int


@dataclass
class LevelConfig:
    """Timeout configuration for a search level."""

    manager_timeout_seconds: int
    worker_timeout_seconds: int
    worker_visible_timeout: int


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_transient_retries: int
    base_delay_seconds: float
    max_delay_seconds: float
    exponential_base: float
    max_validation_retries: int

    def get_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay for a retry attempt.

        Args:
            attempt: Zero-indexed attempt number (0 = first retry)
        """
        delay = self.base_delay_seconds * (self.exponential_base**attempt)
        return min(delay, self.max_delay_seconds)


@dataclass
class WorkerConfig:
    """Configuration for a worker agent."""

    model: str
    default_timeout_seconds: int


@dataclass
class DatabaseConfig:
    """Database connection configuration."""

    url: str
    pool_size: int
    max_overflow: int


@dataclass
class RetentionConfig:
    """Session retention configuration."""

    session_ttl_days: int


@dataclass
class ProcessConfig:
    """Process termination timeouts."""

    sigterm_timeout_seconds: float
    sigkill_timeout_seconds: float


@dataclass
class SynthesizerConfig:
    """Result synthesizer configuration."""

    model: str
    default_timeout_seconds: int


@dataclass
class WorkersConfig:
    """All worker configurations."""

    perplexity: WorkerConfig
    tavily: WorkerConfig
    researcher: WorkerConfig
    manager: WorkerConfig


@dataclass
class PrismConfig:
    """
    Main configuration container.

    Usage:
        config = get_config()
        print(config.server.port)
        print(config.database.url)
    """

    server: ServerConfig
    search: SearchConfig
    levels: dict[int, LevelConfig]
    retry: RetryConfig
    process: ProcessConfig
    synthesizer: SynthesizerConfig
    workers: WorkersConfig
    database: DatabaseConfig
    retention: RetentionConfig


_config_instance: PrismConfig | None = None


def get_config() -> PrismConfig:
    """
    Get the global configuration instance.

    Creates a new instance on first call, reuses on subsequent calls.
    All config values must be present in config.yaml - no hardcoded defaults.

    Raises:
        ConfigError: If required config values are missing
    """
    global _config_instance
    if _config_instance is None:
        cfg = _ensure_config()

        server_cfg = _require(cfg, "server", "server")
        search_cfg = _require(cfg, "search", "search")
        levels_cfg = _require(cfg, "levels", "levels")
        retry_cfg = _require(cfg, "retry", "retry")
        process_cfg = _require(cfg, "process", "process")
        synthesizer_cfg = _require(cfg, "synthesizer", "synthesizer")
        workers_cfg = _require(cfg, "workers", "workers")
        database_cfg = _require(cfg, "database", "database")
        retention_cfg = _require(cfg, "retention", "retention")

        levels: dict[int, LevelConfig] = {}
        for level_key, level_data in levels_cfg.items():
            level_num = int(level_key)
            mgr_path = f"levels.{level_key}.manager_timeout_seconds"
            wrk_path = f"levels.{level_key}.worker_timeout_seconds"
            vis_path = f"levels.{level_key}.worker_visible_timeout"
            levels[level_num] = LevelConfig(
                manager_timeout_seconds=_require(level_data, "manager_timeout_seconds", mgr_path),
                worker_timeout_seconds=_require(level_data, "worker_timeout_seconds", wrk_path),
                worker_visible_timeout=_require(level_data, "worker_visible_timeout", vis_path),
            )

        perplexity_cfg = _require(workers_cfg, "perplexity", "workers.perplexity")
        tavily_cfg = _require(workers_cfg, "tavily", "workers.tavily")
        researcher_cfg = _require(workers_cfg, "researcher", "workers.researcher")
        manager_cfg = _require(workers_cfg, "manager", "workers.manager")

        _config_instance = PrismConfig(
            server=ServerConfig(
                port=_require(server_cfg, "port", "server.port"),
                transport=_require(server_cfg, "transport", "server.transport"),
                log_level=_require(server_cfg, "log_level", "server.log_level"),
            ),
            search=SearchConfig(
                max_query_length=_require(
                    search_cfg, "max_query_length", "search.max_query_length"
                ),
            ),
            levels=levels,
            retry=RetryConfig(
                max_transient_retries=_require(
                    retry_cfg, "max_transient_retries", "retry.max_transient_retries"
                ),
                base_delay_seconds=_require(
                    retry_cfg, "base_delay_seconds", "retry.base_delay_seconds"
                ),
                max_delay_seconds=_require(
                    retry_cfg, "max_delay_seconds", "retry.max_delay_seconds"
                ),
                exponential_base=_require(
                    retry_cfg, "exponential_base", "retry.exponential_base"
                ),
                max_validation_retries=_require(
                    retry_cfg, "max_validation_retries", "retry.max_validation_retries"
                ),
            ),
            process=ProcessConfig(
                sigterm_timeout_seconds=_require(
                    process_cfg, "sigterm_timeout_seconds", "process.sigterm_timeout_seconds"
                ),
                sigkill_timeout_seconds=_require(
                    process_cfg, "sigkill_timeout_seconds", "process.sigkill_timeout_seconds"
                ),
            ),
            synthesizer=SynthesizerConfig(
                model=_require(synthesizer_cfg, "model", "synthesizer.model"),
                default_timeout_seconds=_require(
                    synthesizer_cfg, "default_timeout_seconds",
                    "synthesizer.default_timeout_seconds",
                ),
            ),
            workers=WorkersConfig(
                perplexity=WorkerConfig(
                    model=_require(perplexity_cfg, "model", "workers.perplexity.model"),
                    default_timeout_seconds=_require(
                        perplexity_cfg, "default_timeout_seconds",
                        "workers.perplexity.default_timeout_seconds",
                    ),
                ),
                tavily=WorkerConfig(
                    model=_require(tavily_cfg, "model", "workers.tavily.model"),
                    default_timeout_seconds=_require(
                        tavily_cfg, "default_timeout_seconds",
                        "workers.tavily.default_timeout_seconds",
                    ),
                ),
                researcher=WorkerConfig(
                    model=_require(researcher_cfg, "model", "workers.researcher.model"),
                    default_timeout_seconds=_require(
                        researcher_cfg, "default_timeout_seconds",
                        "workers.researcher.default_timeout_seconds",
                    ),
                ),
                manager=WorkerConfig(
                    model=_require(manager_cfg, "model", "workers.manager.model"),
                    default_timeout_seconds=_require(
                        manager_cfg, "default_timeout_seconds",
                        "workers.manager.default_timeout_seconds",
                    ),
                ),
            ),
            database=DatabaseConfig(
                url=_require(database_cfg, "url", "database.url"),
                pool_size=_require(database_cfg, "pool_size", "database.pool_size"),
                max_overflow=_require(database_cfg, "max_overflow", "database.max_overflow"),
            ),
            retention=RetentionConfig(
                session_ttl_days=_require(
                    retention_cfg, "session_ttl_days", "retention.session_ttl_days"
                ),
            ),
        )
    return _config_instance


def reset_config() -> None:
    """Reset the global configuration. Useful for testing."""
    global _config_instance, _CONFIG
    _config_instance = None
    _CONFIG = {}
