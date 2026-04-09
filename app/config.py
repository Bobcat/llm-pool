from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import os
from pathlib import Path


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.json"
DEFAULT_LOCAL_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "local.json"


@dataclass(frozen=True)
class ServiceSettings:
    host: str = "127.0.0.1"
    port: int = 8010
    log_level: str = "info"


@dataclass(frozen=True)
class ModelSettings:
    model_path: str
    device: str = "cuda"
    compute_type: str = "int8"


@dataclass(frozen=True)
class EngineSettings:
    backend: str = "stub"
    default_model: str = "eurollm-9b-ct2-int8"
    models: dict[str, ModelSettings] = field(default_factory=dict)


@dataclass(frozen=True)
class AppSettings:
    service: ServiceSettings = field(default_factory=ServiceSettings)
    engine: EngineSettings = field(default_factory=EngineSettings)


def load_settings(path: str | Path | None = None) -> AppSettings:
    settings_path = _resolve_settings_path(path)
    payload: dict[str, object] = {}
    if settings_path.exists():
        loaded = json.loads(settings_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload = loaded

    local_settings_path = _resolve_local_settings_path(settings_path)
    if local_settings_path.exists():
        loaded = json.loads(local_settings_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload = _merge_dicts(payload, loaded)

    service_payload = payload.get("service", {}) if isinstance(payload, dict) else {}
    engine_payload = payload.get("engine", {}) if isinstance(payload, dict) else {}
    models_payload = engine_payload.get("models", {}) if isinstance(engine_payload, dict) else {}
    if not isinstance(service_payload, dict):
        service_payload = {}
    if not isinstance(engine_payload, dict):
        engine_payload = {}
    if not isinstance(models_payload, dict):
        models_payload = {}

    models: dict[str, ModelSettings] = {}
    for model_name, model_payload in models_payload.items():
        if not isinstance(model_payload, dict):
            continue
        model_path = str(model_payload.get("model_path", "")).strip()
        if not model_path:
            continue
        models[str(model_name)] = ModelSettings(
            model_path=model_path,
            device=str(model_payload.get("device", "cuda")),
            compute_type=str(model_payload.get("compute_type", "int8")),
        )

    return AppSettings(
        service=ServiceSettings(
            host=str(service_payload.get("host", "127.0.0.1")),
            port=int(service_payload.get("port", 8010)),
            log_level=str(service_payload.get("log_level", "info")),
        ),
        engine=EngineSettings(
            backend=str(engine_payload.get("backend", "stub")),
            default_model=str(engine_payload.get("default_model", "eurollm-9b-ct2-int8")),
            models=models,
        ),
    )


def _resolve_settings_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    env_value = os.environ.get("LLM_POOL_SETTINGS_PATH", "").strip()
    if env_value:
        return Path(env_value)
    env_value = os.environ.get("LLM_RESPONSES_API_SETTINGS_PATH", "").strip()
    if env_value:
        return Path(env_value)
    return DEFAULT_SETTINGS_PATH


def _resolve_local_settings_path(settings_path: Path) -> Path:
    env_value = os.environ.get("LLM_POOL_LOCAL_SETTINGS_PATH", "").strip()
    if env_value:
        return Path(env_value)
    if settings_path == DEFAULT_SETTINGS_PATH:
        return DEFAULT_LOCAL_SETTINGS_PATH
    return settings_path.with_name("local.json")


def _merge_dicts(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged
