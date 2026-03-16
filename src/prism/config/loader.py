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


def _resolve_config_path() -> Path:
    """Resolve the default config path, checking env override and local fallback."""
    env_path = os.environ.get("PRISM_CONFIG_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Config not found at PRISM_CONFIG_PATH={env_path}")

    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH

    local_config = Path(__file__).parent.parent.parent.parent / "config" / "config.yaml"
    if local_config.exists():
        return local_config

    raise FileNotFoundError(
        f"Config not found at {DEFAULT_CONFIG_PATH} or {local_config}"
    )


def _load_config() -> dict[str, Any]:
    """
    Load and merge configuration from files and environment.

    Loading order:
        1. Load default config (config.yaml)
        2. Merge override config (prism.yaml) if exists
        3. Apply environment variable overrides
    """
    config_path = _resolve_config_path()
    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}

    if OVERRIDE_CONFIG_PATH.is_file():
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
    """Timeout and allocation configuration for a search level."""

    worker_timeout_seconds: int
    worker_visible_timeout: int
    agent_allocation: dict[str, int] | None = None


@dataclass
class ModelConfig:
    """Model configuration for a specific level."""

    model: str
    effort: str | None = None


@dataclass
class ModelsConfig:
    """Per-level model configuration for all agent types."""

    session_manager: dict[int, ModelConfig]
    claude_workers: dict[int, ModelConfig]
    gemini_workers: dict[int, ModelConfig]


@dataclass
class Level0Config:
    """Level 0 specific configuration."""

    default_providers: list[str]


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
    models: ModelsConfig
    level0: Level0Config
    retry: RetryConfig
    process: ProcessConfig
    database: DatabaseConfig
    retention: RetentionConfig


_config_instance: PrismConfig | None = None


def _parse_model_configs(
    models_cfg: dict[str, Any], section: str
) -> dict[int, ModelConfig]:
    """Parse a models sub-section into a dict of level -> ModelConfig."""
    result: dict[int, ModelConfig] = {}
    for level_key, model_data in models_cfg.items():
        level_num = int(level_key)
        model_path = f"models.{section}.{level_key}.model"
        result[level_num] = ModelConfig(
            model=_require(model_data, "model", model_path),
            effort=model_data.get("effort"),
        )
    return result


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
        models_cfg = _require(cfg, "models", "models")
        level0_cfg = _require(cfg, "level0", "level0")
        retry_cfg = _require(cfg, "retry", "retry")
        process_cfg = _require(cfg, "process", "process")
        database_cfg = _require(cfg, "database", "database")
        retention_cfg = _require(cfg, "retention", "retention")

        levels: dict[int, LevelConfig] = {}
        for level_key, level_data in levels_cfg.items():
            level_num = int(level_key)
            wrk_path = f"levels.{level_key}.worker_timeout_seconds"
            vis_path = f"levels.{level_key}.worker_visible_timeout"
            raw_alloc = level_data.get("agent_allocation")
            agent_allocation: dict[str, int] | None = None
            if isinstance(raw_alloc, dict):
                agent_allocation = {str(k): int(v) for k, v in raw_alloc.items()}
            levels[level_num] = LevelConfig(
                worker_timeout_seconds=_require(level_data, "worker_timeout_seconds", wrk_path),
                worker_visible_timeout=_require(level_data, "worker_visible_timeout", vis_path),
                agent_allocation=agent_allocation,
            )

        sm_cfg = _require(models_cfg, "session_manager", "models.session_manager")
        cw_cfg = _require(models_cfg, "claude_workers", "models.claude_workers")
        gw_cfg = _require(models_cfg, "gemini_workers", "models.gemini_workers")

        db_url = _require(database_cfg, "url", "database.url")
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

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
            models=ModelsConfig(
                session_manager=_parse_model_configs(sm_cfg, "session_manager"),
                claude_workers=_parse_model_configs(cw_cfg, "claude_workers"),
                gemini_workers=_parse_model_configs(gw_cfg, "gemini_workers"),
            ),
            level0=Level0Config(
                default_providers=_require(
                    level0_cfg, "default_providers", "level0.default_providers"
                ),
            ),
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
            database=DatabaseConfig(
                url=db_url,
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
