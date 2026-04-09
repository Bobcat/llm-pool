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
    prompt_format: str = "generic"
    enabled: bool = True


@dataclass(frozen=True)
class DecodingDefaults:
    beam_size: int = 1
    top_k: int = 1
    top_p: float = 1.0
    temperature: float = 0.1
    repetition_penalty: float = 1.0
    max_tokens: int = 256
    stop: list[str] = field(default_factory=lambda: ["<|im_end|>"])


@dataclass(frozen=True)
class EngineSettings:
    backend: str = "stub"
    default_model: str = "eurollm-9b-ct2-int8"
    models: dict[str, ModelSettings] = field(default_factory=dict)
    decoding: DecodingDefaults = field(default_factory=DecodingDefaults)


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
    decoding_payload = engine_payload.get("decoding", {}) if isinstance(engine_payload, dict) else {}
    if not isinstance(service_payload, dict):
        service_payload = {}
    if not isinstance(engine_payload, dict):
        engine_payload = {}
    if not isinstance(models_payload, dict):
        models_payload = {}
    if not isinstance(decoding_payload, dict):
        decoding_payload = {}

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
            prompt_format=str(model_payload.get("prompt_format", "generic")),
            enabled=bool(model_payload.get("enabled", True)),
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
            decoding=DecodingDefaults(
                beam_size=int(decoding_payload.get("beam_size", 1)),
                top_k=int(decoding_payload.get("top_k", 1)),
                top_p=float(decoding_payload.get("top_p", 1.0)),
                temperature=float(decoding_payload.get("temperature", 0.1)),
                repetition_penalty=float(decoding_payload.get("repetition_penalty", 1.0)),
                max_tokens=int(decoding_payload.get("max_tokens", 256)),
                stop=_coerce_stop_tokens(decoding_payload.get("stop"), default=["<|im_end|>"]),
            ),
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


def _coerce_stop_tokens(value: object, *, default: list[str]) -> list[str]:
    if isinstance(value, list):
        tokens = [str(item) for item in value if str(item) != ""]
        if tokens:
            return tokens
    return list(default)
