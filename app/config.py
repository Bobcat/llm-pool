from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import os
from pathlib import Path


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.json"
DEFAULT_LOCAL_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "local.json"
_GGUF_FLASH_ATTN_ALLOWED_VALUES = ("on", "off", "auto")


@dataclass(frozen=True)
class ServiceSettings:
    host: str = "127.0.0.1"
    port: int = 8010
    log_level: str = "info"


_ALLOWED_MODALITIES = ("text", "image")
_REMOTE_FILE_MODES = ("chat_completions_inline", "files_extract")


@dataclass(frozen=True)
class ModelSettings:
    model_path: str | None
    backend: str | None = None
    device: str = "cuda"
    compute_type: str = "int8"
    prompt_format: str = "generic"
    enable_thinking: bool | None = None
    enabled: bool = True
    replicas: int = 1
    replica_max: int = 1
    target_inflight: int = 1
    modalities: tuple[str, ...] = ("text",)
    exllama_cache_size: int = 8192
    exllama_cache_quant: str | None = None
    exllama_gpu_split: str | None = None
    exllama_tensor_parallel: bool = False
    exllama_tp_backend: str = "native"
    exllama_max_batch_size: int = 64
    exllama_max_chunk_size: int = 2048
    exllama_max_q_size: int = 8
    exllama_max_rq_tokens: int | None = None
    gguf_n_gpu_layers: int = -1
    gguf_n_ctx: int = 4096
    gguf_flash_attn: str = "auto"
    gguf_type_k: str | None = None
    gguf_type_v: str | None = None
    remote_api_kind: str | None = None
    remote_base_url: str | None = None
    remote_api_key_env: str | None = None
    remote_model: str | None = None
    remote_timeout_s: float = 60.0
    remote_health_check: str = "config_only"
    remote_max_retries: int = 0
    remote_thinking: str | None = None
    remote_file_mode: str | None = None
    remote_file_purpose: str | None = None
    llama_server_binary: str = "llama-server"
    llama_server_host: str = "127.0.0.1"
    llama_server_port: int | None = None
    llama_server_model_alias: str | None = None
    llama_server_timeout_s: float = 120.0
    llama_server_start_timeout_s: float = 180.0
    llama_server_stop_timeout_s: float = 10.0
    llama_server_library_path: tuple[str, ...] = ()
    llama_server_api_key: str | None = None
    llama_server_n_ctx: int | None = None
    llama_server_n_gpu_layers: str | None = None
    llama_server_flash_attn: str = "auto"
    llama_server_mmproj_path: str | None = None
    llama_server_image_max_tokens: int | None = None
    llama_server_draft_model_path: str | None = None
    llama_server_spec_type: str | None = None
    llama_server_spec_draft_n_max: int | None = None
    llama_server_spec_draft_p_min: float | None = None
    llama_server_spec_draft_ngl: str | None = None
    llama_server_reasoning: str | None = None
    llama_server_extra_args: tuple[str, ...] = ()
    vllm_model: str | None = None
    vllm_dtype: str = "auto"
    vllm_gpu_memory_utilization: float | None = None
    vllm_kv_cache_memory_bytes: int | None = None
    vllm_kv_cache_dtype: str = "auto"
    vllm_max_model_len: int | None = None
    vllm_tensor_parallel_size: int = 1
    vllm_trust_remote_code: bool = False
    vllm_enforce_eager: bool = False
    vllm_limit_mm_per_prompt: tuple[tuple[str, int], ...] = ()
    vllm_mm_processor_kwargs: tuple[tuple[str, int], ...] = ()
    vllm_speculative_method: str | None = None
    vllm_speculative_model: str | None = None
    vllm_speculative_moe_backend: str | None = None
    vllm_speculative_attention_backend: str | None = None
    vllm_num_speculative_tokens: int = 1
    vllm_serve_binary: str = "vllm"
    vllm_serve_host: str = "127.0.0.1"
    vllm_serve_port: int | None = None
    vllm_serve_model_alias: str | None = None
    vllm_serve_timeout_s: float = 120.0
    vllm_serve_start_timeout_s: float = 300.0
    vllm_serve_stop_timeout_s: float = 10.0
    vllm_serve_library_path: tuple[str, ...] = ()
    vllm_serve_env: tuple[tuple[str, str], ...] = ()
    vllm_serve_api_key: str | None = None
    vllm_serve_extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecodingDefaults:
    beam_size: int = 1
    top_k: int = 1
    top_p: float = 1.0
    temperature: float = 0.1
    repetition_penalty: float = 1.0
    max_tokens: int = 256
    stop: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EngineSettings:
    backend: str = "stub"
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
    default_backend = str(engine_payload.get("backend", "stub")).strip().lower()
    for model_name, model_payload in models_payload.items():
        if not isinstance(model_payload, dict):
            continue
        backend_value = model_payload.get("backend")
        backend = None
        if backend_value is not None:
            parsed_backend = str(backend_value).strip().lower()
            if parsed_backend:
                backend = parsed_backend
        resolved_backend = backend or default_backend
        model_path = _coerce_optional_str(model_payload.get("model_path"))
        if model_path is None and resolved_backend not in {"openai_remote", "vllm", "vllm_serve"}:
            continue
        cache_quant_value = model_payload.get("exllama_cache_quant")
        cache_quant = None
        if cache_quant_value is not None:
            parsed_cache_quant = str(cache_quant_value).strip()
            if parsed_cache_quant:
                cache_quant = parsed_cache_quant
        gpu_split_value = model_payload.get("exllama_gpu_split")
        gpu_split = None
        if gpu_split_value is not None:
            parsed_gpu_split = str(gpu_split_value).strip()
            if parsed_gpu_split:
                gpu_split = parsed_gpu_split
        max_rq_tokens_value = model_payload.get("exllama_max_rq_tokens")
        max_rq_tokens = None
        if max_rq_tokens_value not in (None, ""):
            max_rq_tokens = int(max_rq_tokens_value)
        gguf_type_k_value = model_payload.get("gguf_type_k")
        gguf_type_k = None
        if gguf_type_k_value is not None:
            parsed_gguf_type_k = str(gguf_type_k_value).strip()
            if parsed_gguf_type_k:
                gguf_type_k = parsed_gguf_type_k
        gguf_type_v_value = model_payload.get("gguf_type_v")
        gguf_type_v = None
        if gguf_type_v_value is not None:
            parsed_gguf_type_v = str(gguf_type_v_value).strip()
            if parsed_gguf_type_v:
                gguf_type_v = parsed_gguf_type_v
        remote_api_kind = _coerce_optional_str(model_payload.get("remote_api_kind"))
        if remote_api_kind is not None:
            remote_api_kind = remote_api_kind.lower()
        remote_health_check = str(model_payload.get("remote_health_check", "config_only")).strip().lower()
        if remote_health_check == "":
            remote_health_check = "config_only"
        remote_thinking = _coerce_optional_str(model_payload.get("remote_thinking"))
        if remote_thinking is not None:
            remote_thinking = remote_thinking.lower()
        remote_file_mode = _coerce_optional_str(model_payload.get("remote_file_mode"))
        if remote_file_mode is not None:
            remote_file_mode = remote_file_mode.lower()
            if remote_file_mode not in _REMOTE_FILE_MODES:
                raise ValueError(
                    "remote_file_mode must be one of "
                    f"{', '.join(repr(value) for value in _REMOTE_FILE_MODES)}"
                )
        models[str(model_name)] = ModelSettings(
            model_path=model_path,
            backend=backend,
            device=str(model_payload.get("device", "cuda")),
            compute_type=str(model_payload.get("compute_type", "int8")),
            prompt_format=str(model_payload.get("prompt_format", "generic")),
            enable_thinking=(
                bool(model_payload["enable_thinking"])
                if model_payload.get("enable_thinking") is not None
                else None
            ),
            enabled=bool(model_payload.get("enabled", True)),
            replicas=max(1, int(model_payload.get("replicas", 1))),
            replica_max=max(1, int(model_payload.get("replica_max", 1))),
            target_inflight=max(1, int(model_payload.get("target_inflight", 1))),
            modalities=_coerce_modalities(model_payload.get("modalities")),
            exllama_cache_size=int(model_payload.get("exllama_cache_size", 8192)),
            exllama_cache_quant=cache_quant,
            exllama_gpu_split=gpu_split,
            exllama_tensor_parallel=bool(model_payload.get("exllama_tensor_parallel", False)),
            exllama_tp_backend=str(model_payload.get("exllama_tp_backend", "native")),
            exllama_max_batch_size=int(model_payload.get("exllama_max_batch_size", 64)),
            exllama_max_chunk_size=int(model_payload.get("exllama_max_chunk_size", 2048)),
            exllama_max_q_size=int(model_payload.get("exllama_max_q_size", 8)),
            exllama_max_rq_tokens=max_rq_tokens,
            gguf_n_gpu_layers=int(model_payload.get("gguf_n_gpu_layers", -1)),
            gguf_n_ctx=int(model_payload.get("gguf_n_ctx", 4096)),
            gguf_flash_attn=_coerce_gguf_flash_attn_mode(model_payload.get("gguf_flash_attn", "auto")),
            gguf_type_k=gguf_type_k,
            gguf_type_v=gguf_type_v,
            remote_api_kind=remote_api_kind,
            remote_base_url=_coerce_optional_str(model_payload.get("remote_base_url")),
            remote_api_key_env=_coerce_optional_str(model_payload.get("remote_api_key_env")),
            remote_model=_coerce_optional_str(model_payload.get("remote_model")),
            remote_timeout_s=float(model_payload.get("remote_timeout_s", 60.0)),
            remote_health_check=remote_health_check,
            remote_max_retries=int(model_payload.get("remote_max_retries", 0)),
            remote_thinking=remote_thinking,
            remote_file_mode=remote_file_mode,
            remote_file_purpose=_coerce_optional_str(
                model_payload.get("remote_file_purpose")
            ),
            llama_server_binary=(
                str(model_payload.get("llama_server_binary", "llama-server") or "llama-server").strip()
                or "llama-server"
            ),
            llama_server_host=(
                str(model_payload.get("llama_server_host", "127.0.0.1") or "127.0.0.1").strip()
                or "127.0.0.1"
            ),
            llama_server_port=_coerce_optional_positive_int(model_payload.get("llama_server_port")),
            llama_server_model_alias=_coerce_optional_str(model_payload.get("llama_server_model_alias")),
            llama_server_timeout_s=float(model_payload.get("llama_server_timeout_s", 120.0)),
            llama_server_start_timeout_s=float(model_payload.get("llama_server_start_timeout_s", 180.0)),
            llama_server_stop_timeout_s=float(model_payload.get("llama_server_stop_timeout_s", 10.0)),
            llama_server_library_path=_coerce_path_tuple(
                model_payload.get("llama_server_library_path"),
                "llama_server_library_path",
            ),
            llama_server_api_key=_coerce_optional_str(model_payload.get("llama_server_api_key")),
            llama_server_n_ctx=_coerce_optional_positive_int(model_payload.get("llama_server_n_ctx")),
            llama_server_n_gpu_layers=_coerce_optional_str(model_payload.get("llama_server_n_gpu_layers")),
            llama_server_flash_attn=_coerce_gguf_flash_attn_mode(
                model_payload.get("llama_server_flash_attn", "auto")
            ),
            llama_server_mmproj_path=_coerce_optional_str(model_payload.get("llama_server_mmproj_path")),
            llama_server_image_max_tokens=_coerce_optional_positive_int(
                model_payload.get("llama_server_image_max_tokens")
            ),
            llama_server_draft_model_path=_coerce_optional_str(model_payload.get("llama_server_draft_model_path")),
            llama_server_spec_type=_coerce_optional_str(model_payload.get("llama_server_spec_type")),
            llama_server_spec_draft_n_max=_coerce_optional_positive_int(
                model_payload.get("llama_server_spec_draft_n_max")
            ),
            llama_server_spec_draft_p_min=_coerce_optional_float(model_payload.get("llama_server_spec_draft_p_min")),
            llama_server_spec_draft_ngl=_coerce_optional_str(model_payload.get("llama_server_spec_draft_ngl")),
            llama_server_reasoning=_coerce_optional_str(model_payload.get("llama_server_reasoning")),
            llama_server_extra_args=_coerce_str_tuple(
                model_payload.get("llama_server_extra_args"),
                "llama_server_extra_args",
            ),
            vllm_model=_coerce_optional_str(model_payload.get("vllm_model")),
            vllm_dtype=str(model_payload.get("vllm_dtype", "auto") or "auto").strip() or "auto",
            vllm_gpu_memory_utilization=_coerce_optional_float(model_payload.get("vllm_gpu_memory_utilization")),
            vllm_kv_cache_memory_bytes=_coerce_optional_positive_int(model_payload.get("vllm_kv_cache_memory_bytes")),
            vllm_kv_cache_dtype=str(model_payload.get("vllm_kv_cache_dtype", "auto") or "auto").strip() or "auto",
            vllm_max_model_len=_coerce_optional_positive_int(model_payload.get("vllm_max_model_len")),
            vllm_tensor_parallel_size=max(1, int(model_payload.get("vllm_tensor_parallel_size", 1))),
            vllm_trust_remote_code=bool(model_payload.get("vllm_trust_remote_code", False)),
            vllm_enforce_eager=bool(model_payload.get("vllm_enforce_eager", False)),
            vllm_limit_mm_per_prompt=_coerce_mm_limit(model_payload.get("vllm_limit_mm_per_prompt")),
            vllm_mm_processor_kwargs=_coerce_int_str_map(model_payload.get("vllm_mm_processor_kwargs")),
            vllm_speculative_method=_coerce_optional_str(model_payload.get("vllm_speculative_method")),
            vllm_speculative_model=_coerce_optional_str(model_payload.get("vllm_speculative_model")),
            vllm_speculative_moe_backend=_coerce_optional_str(
                model_payload.get("vllm_speculative_moe_backend")
            ),
            vllm_speculative_attention_backend=_coerce_optional_str(
                model_payload.get("vllm_speculative_attention_backend")
            ),
            vllm_num_speculative_tokens=max(1, int(model_payload.get("vllm_num_speculative_tokens", 1))),
            vllm_serve_binary=(
                str(model_payload.get("vllm_serve_binary", "vllm") or "vllm").strip()
                or "vllm"
            ),
            vllm_serve_host=(
                str(model_payload.get("vllm_serve_host", "127.0.0.1") or "127.0.0.1").strip()
                or "127.0.0.1"
            ),
            vllm_serve_port=_coerce_optional_positive_int(model_payload.get("vllm_serve_port")),
            vllm_serve_model_alias=_coerce_optional_str(model_payload.get("vllm_serve_model_alias")),
            vllm_serve_timeout_s=float(model_payload.get("vllm_serve_timeout_s", 120.0)),
            vllm_serve_start_timeout_s=float(model_payload.get("vllm_serve_start_timeout_s", 300.0)),
            vllm_serve_stop_timeout_s=float(model_payload.get("vllm_serve_stop_timeout_s", 10.0)),
            vllm_serve_library_path=_coerce_path_tuple(
                model_payload.get("vllm_serve_library_path"),
                "vllm_serve_library_path",
            ),
            vllm_serve_env=_coerce_str_str_map(
                model_payload.get("vllm_serve_env"),
                "vllm_serve_env",
            ),
            vllm_serve_api_key=_coerce_optional_str(model_payload.get("vllm_serve_api_key")),
            vllm_serve_extra_args=_coerce_str_tuple(
                model_payload.get("vllm_serve_extra_args"),
                "vllm_serve_extra_args",
            ),
        )
        if models[str(model_name)].replicas > models[str(model_name)].replica_max:
            raise ValueError(
                f"model '{model_name}' has replicas={models[str(model_name)].replicas} "
                f"greater than replica_max={models[str(model_name)].replica_max}"
            )

    return AppSettings(
        service=ServiceSettings(
            host=str(service_payload.get("host", "127.0.0.1")),
            port=int(service_payload.get("port", 8010)),
            log_level=str(service_payload.get("log_level", "info")),
        ),
        engine=EngineSettings(
            backend=str(engine_payload.get("backend", "stub")),
            models=models,
            decoding=DecodingDefaults(
                beam_size=int(decoding_payload.get("beam_size", 1)),
                top_k=int(decoding_payload.get("top_k", 1)),
                top_p=float(decoding_payload.get("top_p", 1.0)),
                temperature=float(decoding_payload.get("temperature", 0.1)),
                repetition_penalty=float(decoding_payload.get("repetition_penalty", 1.0)),
                max_tokens=int(decoding_payload.get("max_tokens", 256)),
                stop=_coerce_stop_tokens(decoding_payload.get("stop"), default=[]),
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


def _coerce_optional_str(value: object) -> str | None:
    if value is None:
        return None
    parsed = str(value).strip()
    if parsed == "":
        return None
    return parsed


def _coerce_optional_positive_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    if parsed <= 0:
        return None
    return parsed


def _coerce_optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _coerce_str_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of strings")
    return tuple(str(item) for item in value)


def _coerce_path_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        raw_items = value.split(os.pathsep)
    elif isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raise ValueError(f"{field_name} must be a list of paths or a path-separated string")
    return tuple(item for item in (str(raw_item).strip() for raw_item in raw_items) if item)


def _coerce_str_str_map(value: object, field_name: str) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object of {{key: string}}")
    parsed: list[tuple[str, str]] = []
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        if key == "":
            continue
        parsed.append((key, str(raw_value)))
    return tuple(parsed)


def _coerce_mm_limit(value: object) -> tuple[tuple[str, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError("vllm_limit_mm_per_prompt must be an object of {modality: int}")
    parsed: list[tuple[str, int]] = []
    for key, raw in value.items():
        modality = str(key).strip().lower()
        if modality == "":
            continue
        try:
            count = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"vllm_limit_mm_per_prompt['{modality}'] must be an integer"
            ) from exc
        if count < 1:
            raise ValueError(
                f"vllm_limit_mm_per_prompt['{modality}'] must be >= 1"
            )
        parsed.append((modality, count))
    return tuple(parsed)


def _coerce_int_str_map(value: object) -> tuple[tuple[str, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError("vllm_mm_processor_kwargs must be an object of {key: int}")
    parsed: list[tuple[str, int]] = []
    for key, raw in value.items():
        name = str(key).strip()
        if name == "":
            continue
        try:
            parsed.append((name, int(raw)))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"vllm_mm_processor_kwargs['{name}'] must be an integer"
            ) from exc
    return tuple(parsed)


def _coerce_modalities(value: object) -> tuple[str, ...]:
    if value is None:
        return ("text",)
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple)):
        candidates = [str(item) for item in value]
    else:
        raise ValueError("modalities must be a list of strings")
    seen: set[str] = set()
    parsed: list[str] = []
    for raw in candidates:
        normalized = str(raw).strip().lower()
        if normalized == "" or normalized in seen:
            continue
        if normalized not in _ALLOWED_MODALITIES:
            allowed = ", ".join(_ALLOWED_MODALITIES)
            raise ValueError(f"modality must be one of: {allowed}")
        seen.add(normalized)
        parsed.append(normalized)
    if not parsed:
        return ("text",)
    if "text" not in seen:
        parsed.insert(0, "text")
    return tuple(parsed)


def _coerce_gguf_flash_attn_mode(value: object, *, default: str = "auto") -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "on" if value else "off"
    normalized = str(value).strip().lower()
    if normalized == "":
        return default
    if normalized not in _GGUF_FLASH_ATTN_ALLOWED_VALUES:
        allowed_values = ", ".join(_GGUF_FLASH_ATTN_ALLOWED_VALUES)
        raise ValueError(f"gguf_flash_attn must be one of: {allowed_values}")
    return normalized
