from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import logging
from pathlib import Path
import subprocess


LOGGER = logging.getLogger("llm_pool.engine")

_GGUF_CACHE_TYPE_ALLOWED_VALUES = (
    "f32",
    "f16",
    "bf16",
    "q8_0",
    "q4_0",
    "q4_1",
    "iq4_nl",
    "q5_0",
    "q5_1",
)
_GGUF_FLASH_ATTN_ALLOWED_VALUES = ("on", "off", "auto")

_EXLLAMA_CACHE_BIT_ALLOWED_VALUES = (2, 3, 4, 5, 6, 7, 8)

_VLLM_KV_CACHE_DTYPE_ALLOWED_VALUES = ("auto", "fp8", "fp8_e4m3", "fp8_e5m2")

_COMMON_MODEL_DEFINITION_FIELDS = (
    "model_path",
    "backend",
    "prompt_format",
    "enable_thinking",
    "enabled",
    "replicas",
    "replica_max",
    "target_inflight",
)

_BACKEND_MODEL_DEFINITION_FIELDS = {
    "ct2": (
        "device",
        "compute_type",
    ),
    "exllamav3": (
        "device",
        "exllama_cache_size",
        "exllama_cache_quant",
        "exllama_gpu_split",
        "exllama_tensor_parallel",
        "exllama_tp_backend",
        "exllama_max_batch_size",
        "exllama_max_chunk_size",
        "exllama_max_q_size",
        "exllama_max_rq_tokens",
    ),
    "llama_cpp": (
        "gguf_n_gpu_layers",
        "gguf_n_ctx",
        "gguf_flash_attn",
        "gguf_type_k",
        "gguf_type_v",
    ),
    "llama_server": (
        "llama_server_binary",
        "llama_server_host",
        "llama_server_port",
        "llama_server_model_alias",
        "llama_server_timeout_s",
        "llama_server_start_timeout_s",
        "llama_server_stop_timeout_s",
        "llama_server_library_path",
        "llama_server_api_key",
        "llama_server_n_ctx",
        "llama_server_n_gpu_layers",
        "llama_server_flash_attn",
        "llama_server_mmproj_path",
        "llama_server_image_max_tokens",
        "llama_server_draft_model_path",
        "llama_server_spec_type",
        "llama_server_spec_draft_n_max",
        "llama_server_spec_draft_p_min",
        "llama_server_spec_draft_ngl",
        "llama_server_reasoning",
        "llama_server_extra_args",
    ),
    "openai_remote": (
        "remote_api_kind",
        "remote_base_url",
        "remote_api_key_env",
        "remote_model",
        "remote_timeout_s",
        "remote_health_check",
        "remote_max_retries",
        "remote_thinking",
        "remote_prompt_cache_key_enabled",
        "remote_file_mode",
        "remote_file_purpose",
    ),
    "vllm": (
        "vllm_model",
        "vllm_dtype",
        "vllm_gpu_memory_utilization",
        "vllm_kv_cache_memory_bytes",
        "vllm_kv_cache_dtype",
        "vllm_max_model_len",
        "vllm_tensor_parallel_size",
        "vllm_trust_remote_code",
        "vllm_enforce_eager",
        "vllm_limit_mm_per_prompt",
        "vllm_mm_processor_kwargs",
        "vllm_speculative_method",
        "vllm_speculative_model",
        "vllm_speculative_moe_backend",
        "vllm_speculative_attention_backend",
        "vllm_num_speculative_tokens",
    ),
    "vllm_serve": (
        "vllm_model",
        "vllm_dtype",
        "vllm_gpu_memory_utilization",
        "vllm_kv_cache_memory_bytes",
        "vllm_kv_cache_dtype",
        "vllm_max_model_len",
        "vllm_tensor_parallel_size",
        "vllm_trust_remote_code",
        "vllm_enforce_eager",
        "vllm_limit_mm_per_prompt",
        "vllm_mm_processor_kwargs",
        "vllm_speculative_method",
        "vllm_speculative_model",
        "vllm_speculative_moe_backend",
        "vllm_speculative_attention_backend",
        "vllm_num_speculative_tokens",
        "vllm_serve_binary",
        "vllm_serve_host",
        "vllm_serve_port",
        "vllm_serve_model_alias",
        "vllm_serve_timeout_s",
        "vllm_serve_start_timeout_s",
        "vllm_serve_stop_timeout_s",
        "vllm_serve_library_path",
        "vllm_serve_env",
        "vllm_serve_api_key",
        "vllm_serve_extra_args",
    ),
}

# llama_cpp multi-turn is text-only and depends on llama.cpp's chat-completion path.
# TranslateGemma remains a structured translation flow rather than assistant chat.
_GGUF_MULTI_TURN_PROMPT_FORMATS = frozenset(
    {
        "gemma4_template",
        "generic",
        "mistral_template",
        "qwen3_template",
    }
)
_THINKING_CONTROL_PROMPT_FORMATS = {
    "ct2": frozenset({"qwen3_template"}),
    "exllamav3": frozenset({"gemma4_template", "qwen3_template"}),
    "llama_cpp": frozenset({"gemma4_template"}),
    "vllm": frozenset({"gemma4_template", "qwen3_template"}),
}
_DEFAULT_THINKING_MODES = ("default",)
_OVERRIDE_THINKING_MODES = ("default", "enabled", "disabled")
_DEFAULT_RESPONSE_FORMATS = ("text",)
_VLLM_SERVE_RESPONSE_FORMATS = ("text", "json_schema")


def _model_supports_multi_turn(backend: str, prompt_format: str | None) -> bool:
    normalized_backend = backend.strip().lower()
    if normalized_backend in {"llama_server", "openai_remote", "vllm", "vllm_serve"}:
        return True
    if normalized_backend == "llama_cpp":
        normalized_prompt_format = (prompt_format or "").strip().lower()
        return normalized_prompt_format in _GGUF_MULTI_TURN_PROMPT_FORMATS
    return False


def _model_supports_file_inputs(backend: str, remote_file_mode: str | None) -> bool:
    return backend.strip().lower() == "openai_remote" and remote_file_mode is not None


def _model_thinking_modes(
    backend: str,
    prompt_format: str | None,
    remote_thinking: str | None = None,
) -> list[str]:
    normalized_backend = backend.strip().lower()
    if normalized_backend == "openai_remote":
        if remote_thinking is not None:
            return list(_OVERRIDE_THINKING_MODES)
        return list(_DEFAULT_THINKING_MODES)
    prompt_formats = _THINKING_CONTROL_PROMPT_FORMATS.get(normalized_backend)
    if prompt_formats is None:
        return list(_DEFAULT_THINKING_MODES)
    normalized_prompt_format = (prompt_format or "").strip().lower()
    if normalized_prompt_format in prompt_formats:
        return list(_OVERRIDE_THINKING_MODES)
    return list(_DEFAULT_THINKING_MODES)


def _model_response_formats(backend: str) -> list[str]:
    if backend.strip().lower() == "vllm_serve":
        return list(_VLLM_SERVE_RESPONSE_FORMATS)
    return list(_DEFAULT_RESPONSE_FORMATS)


def _resolve_request_enable_thinking(request, default: bool | None) -> bool | None:
    if request.thinking == "enabled":
        return True
    if request.thinking == "disabled":
        return False
    return default


def _resolve_request_remote_thinking(request, default: str | None) -> str | None:
    if request.thinking == "enabled":
        return "enabled"
    if request.thinking == "disabled":
        return "disabled"
    return default


def _native_stop_strings(prompt_format: str) -> list[str]:
    if prompt_format in {"generic", "qwen3_template"}:
        return ["<|im_end|>"]
    if prompt_format == "gemma4_template":
        return ["<turn|>"]
    if prompt_format == "translategemma_template":
        return ["<end_of_turn>"]
    return []


def _merge_stop_strings(prompt_format: str, extra_stop_tokens: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for token in [*_native_stop_strings(prompt_format), *extra_stop_tokens]:
        if token == "" or token in seen:
            continue
        seen.add(token)
        merged.append(token)
    return merged


def _exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    if message != "":
        return message
    return exc.__class__.__name__


def _query_gpu_memory() -> tuple[list[dict[str, object]], str | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return [], "nvidia-smi not found"
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() if isinstance(exc.stderr, str) else ""
        if message == "":
            message = "nvidia-smi failed"
        return [], message

    gpus: list[dict[str, object]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if line == "":
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpu_index = int(parts[0])
            used_mib = int(parts[2])
            total_mib = int(parts[3])
        except ValueError:
            continue
        gpus.append(
            {
                "index": gpu_index,
                "name": parts[1],
                "used_mib": used_mib,
                "total_mib": total_mib,
                "used_over_total": f"{used_mib}MiB / {total_mib}MiB",
            }
        )
    return gpus, None


def _query_primary_gpu_used_mib() -> int | None:
    gpus, _ = _query_gpu_memory()
    if not gpus:
        return None
    first = gpus[0]
    used = first.get("used_mib")
    if isinstance(used, int):
        return used
    return None


def _estimate_model_artifact_size_mib(model_path: str | None) -> int | None:
    if model_path is None:
        return None
    path = Path(model_path)
    try:
        if path.is_file():
            total_bytes = path.stat().st_size
        elif path.is_dir():
            total_bytes = 0
            for candidate in path.rglob("*"):
                if candidate.is_file():
                    total_bytes += candidate.stat().st_size
        else:
            return None
    except OSError:
        return None
    if total_bytes <= 0:
        return None
    mib = int(total_bytes / (1024 * 1024))
    if mib <= 0:
        return 1
    return mib


def _empty_cuda_allocator_cache() -> None:
    try:
        import torch  # type: ignore
    except Exception:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        LOGGER.warning("Failed to empty CUDA allocator cache during runtime cleanup.", exc_info=True)


def _load_constraints_for_backend(backend: str) -> dict[str, object]:
    normalized_backend = backend.strip().lower()
    if normalized_backend == "llama_cpp":
        return {
            "gguf_n_ctx": {
                "kind": "integer",
                "minimum": 1,
                "step": 1,
            },
            "gguf_flash_attn": {
                "kind": "enum",
                "default": "auto",
                "allowed_values": list(_GGUF_FLASH_ATTN_ALLOWED_VALUES),
                "examples": ["auto", "on", "off"],
            },
            "gguf_type_k": {
                "kind": "string_or_null",
                "format": "ggml_type_name",
                "default": "f16",
                "allowed_values": list(_GGUF_CACHE_TYPE_ALLOWED_VALUES),
                "examples": ["f16", "q8_0", "q4_0"],
            },
            "gguf_type_v": {
                "kind": "string_or_null",
                "format": "ggml_type_name",
                "default": "f16",
                "allowed_values": list(_GGUF_CACHE_TYPE_ALLOWED_VALUES),
                "examples": ["f16", "q8_0", "q4_0"],
            },
        }
    if normalized_backend == "llama_server":
        return {
            "llama_server_n_ctx": {
                "kind": "integer",
                "minimum": 1,
                "step": 1,
            },
            "llama_server_image_max_tokens": {
                "kind": "integer",
                "minimum": 1,
                "step": 1,
            },
            "llama_server_spec_type": {
                "kind": "enum",
                "default": "draft-mtp",
                "allowed_values": ["draft-mtp"],
                "examples": ["draft-mtp"],
            },
            "llama_server_spec_draft_n_max": {
                "kind": "integer",
                "minimum": 1,
                "maximum": 6,
                "step": 1,
                "default": 2,
            },
            "llama_server_spec_draft_p_min": {
                "kind": "float",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.0,
            },
        }
    if normalized_backend == "exllamav3":
        return {
            "exllama_cache_size": {
                "kind": "integer",
                "minimum": 256,
                "step": 256,
            },
            "exllama_max_rq_tokens": {
                "kind": "integer",
                "minimum": 1,
                "step": 1,
            },
            "exllama_cache_k_bits": {
                "kind": "integer_or_null",
                "minimum": 2,
                "maximum": 8,
                "default": None,
                "null_means": "fp16",
                "allowed_values": list(_EXLLAMA_CACHE_BIT_ALLOWED_VALUES),
            },
            "exllama_cache_v_bits": {
                "kind": "integer_or_null",
                "minimum": 2,
                "maximum": 8,
                "default": None,
                "null_means": "fp16",
                "allowed_values": list(_EXLLAMA_CACHE_BIT_ALLOWED_VALUES),
            },
            "exllama_cache_quant": {
                "kind": "string_or_null",
                "format": "<bits>|<k_bits>,<v_bits>",
            },
        }
    if normalized_backend in {"vllm", "vllm_serve"}:
        return {
            "vllm_max_model_len": {
                "kind": "integer",
                "minimum": 256,
                "step": 256,
            },
            "vllm_kv_cache_dtype": {
                "kind": "enum",
                "default": "auto",
                "allowed_values": list(_VLLM_KV_CACHE_DTYPE_ALLOWED_VALUES),
                "examples": ["auto", "fp8"],
            },
            "vllm_kv_cache_memory_bytes": {
                "kind": "integer",
                "minimum": 268435456,
                "step": 268435456,
                "unit": "bytes",
                "display_unit": "mib",
            },
            "vllm_max_pixels": {
                "kind": "integer",
                "minimum": 200704,
                "step": 200704,
                "unit": "pixels",
            },
            "vllm_speculative_method": {
                "kind": "string_or_null",
                "format": "vllm_speculative_method",
                "default": None,
                "examples": ["mtp", "draft_model", "mlp_speculator"],
            },
            "vllm_speculative_model": {
                "kind": "string_or_null",
                "format": "hf_id_or_local_path",
                "default": None,
                "examples": ["google/gemma-4-26B-A4B-it-assistant"],
            },
            "vllm_speculative_moe_backend": {
                "kind": "string_or_null",
                "format": "vllm_moe_backend",
                "default": None,
                "examples": ["triton", "marlin"],
            },
            "vllm_speculative_attention_backend": {
                "kind": "string_or_null",
                "format": "vllm_attention_backend",
                "default": None,
                "examples": ["triton_attn", "flashinfer"],
            },
            "vllm_num_speculative_tokens": {
                "kind": "integer",
                "minimum": 1,
                "step": 1,
                "default": 1,
            },
        }
    return {}


def _load_recommendations_for_backend(backend: str) -> dict[str, object]:
    normalized_backend = backend.strip().lower()
    if normalized_backend == "llama_cpp":
        return {
            "gguf_cache_type_pairs": {
                "kind": "pair_presets",
                "fields": ["gguf_type_k", "gguf_type_v"],
                "recommended_pairs": [
                    {
                        "label": "f16/f16",
                        "gguf_type_k": "f16",
                        "gguf_type_v": "f16",
                    },
                    {
                        "label": "q8_0/q8_0",
                        "gguf_type_k": "q8_0",
                        "gguf_type_v": "q8_0",
                    },
                    {
                        "label": "q4_0/q4_0",
                        "gguf_type_k": "q4_0",
                        "gguf_type_v": "q4_0",
                    },
                ],
                "notes": [
                    "Service-curated presets for GGUF cache types.",
                    "Prefer symmetric GGUF K/V pairs by default; asymmetric pairs may reduce or disable GPU offload in upstream llama.cpp.",
                ],
            }
        }
    if normalized_backend == "exllamav3":
        return {
            "exllama_cache_bit_pairs": {
                "kind": "pair_presets",
                "fields": ["exllama_cache_k_bits", "exllama_cache_v_bits"],
                "recommended_pairs": [
                    {
                        "label": "fp16",
                        "exllama_cache_k_bits": None,
                        "exllama_cache_v_bits": None,
                    },
                    {
                        "label": "8/8",
                        "exllama_cache_k_bits": 8,
                        "exllama_cache_v_bits": 8,
                    },
                    {
                        "label": "8/4",
                        "exllama_cache_k_bits": 8,
                        "exllama_cache_v_bits": 4,
                    },
                ],
                "notes": [
                    "Service-curated presets for ExLlamaV3 cache bits.",
                    "These presets are not an exhaustive list of valid ExLlamaV3 K/V bit pairs.",
                ],
            }
        }
    return {}


def _normalize_gguf_cache_type_name(cache_type: str) -> str:
    normalized = cache_type.strip().lower()
    if normalized == "":
        raise ValueError("GGUF cache type must be a non-empty GGML type name")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in normalized):
        raise ValueError("GGUF cache type must contain only letters, digits, and underscores")
    return normalized


def _normalize_gguf_flash_attn_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized == "":
        raise ValueError("gguf_flash_attn must be one of: on, off, auto")
    if normalized not in _GGUF_FLASH_ATTN_ALLOWED_VALUES:
        raise ValueError("gguf_flash_attn must be one of: on, off, auto")
    return normalized


def _resolve_gguf_cache_type_constant(cache_type: str, *, llama_cpp_module: object):
    normalized = _normalize_gguf_cache_type_name(cache_type)
    constant_name = f"GGML_TYPE_{normalized.upper()}"
    for namespace in (llama_cpp_module, getattr(llama_cpp_module, "llama_cpp", None)):
        if namespace is not None and hasattr(namespace, constant_name):
            return getattr(namespace, constant_name)
    raise ValueError(f"unsupported GGUF cache type: {cache_type!r}")


def _model_definition_payload(model_settings: object, *, resolved_backend: str) -> dict[str, object | None]:
    field_names = [
        *_COMMON_MODEL_DEFINITION_FIELDS,
        *_BACKEND_MODEL_DEFINITION_FIELDS.get(resolved_backend.strip().lower(), ()),
    ]
    return {
        field_name: getattr(model_settings, field_name)
        for field_name in field_names
    }


class UnknownModelError(LookupError):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.model_name = model_name


class ModelStateError(RuntimeError):
    def __init__(self, model_name: str, code: str) -> None:
        super().__init__(model_name)
        self.model_name = model_name
        self.code = code


class RequestAdmissionError(RuntimeError):
    def __init__(self, *, code: str, status_code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.message = message


class BackendExecutionError(RuntimeError):
    def __init__(self, *, code: str, status_code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.message = message


def _require_text_input(request) -> str:
    """Extract text from a polymorphic ResponseRequest.input for text-only backends.

    Returns the input as plain text. If image content is present, raises a
    ``BackendExecutionError`` with code ``modality_unsupported`` so the API
    layer can map it to a 400 response.
    """
    from app.schemas import ModalityUnsupportedError

    try:
        return request.text_input_or_raise()
    except ModalityUnsupportedError as exc:
        raise BackendExecutionError(
            code="modality_unsupported",
            status_code=400,
            message=f"model {request.model!r} does not support image content",
        ) from exc


@dataclass
class ModelRuntimeState:
    resolved_backend: str
    configured_enabled: bool
    lifecycle: str = "unloaded"
    inflight_requests: int = 0
    last_error: str | None = None
    artifact_size_mib: int | None = None
    observed_vram_mib: int | None = None
    observed_vram_replicas: int | None = None
    load_override: dict[str, object | None] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedDecoding:
    beam_size: int
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    max_tokens: int
    stop: list[str]
