from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import logging
from pathlib import Path
import subprocess


LOGGER = logging.getLogger("llm_pool.engine")

_COMMON_MODEL_DEFINITION_FIELDS = (
    "model_path",
    "backend",
    "prompt_format",
    "enable_thinking",
    "enabled",
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
    "gguf": (
        "gguf_n_gpu_layers",
        "gguf_n_ctx",
        "gguf_flash_attn",
        "gguf_type_k",
        "gguf_type_v",
    ),
}


def _native_stop_strings(prompt_format: str) -> list[str]:
    if prompt_format in {"generic", "qwen3_template"}:
        return ["<|im_end|>"]
    if prompt_format == "gemma4_template":
        return ["<turn|>"]
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


def _estimate_model_artifact_size_mib(model_path: str) -> int | None:
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
    if normalized_backend == "gguf":
        return {
            "gguf_n_ctx": {
                "kind": "integer",
                "minimum": 1,
                "step": 1,
            },
            "gguf_type_k": {
                "kind": "string_or_null",
                "format": "ggml_type_name",
                "examples": ["f16", "q8_0", "q4_0"],
            },
            "gguf_type_v": {
                "kind": "string_or_null",
                "format": "ggml_type_name",
                "examples": ["f16", "q8_0", "q4_0"],
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
            "exllama_cache_quant": {
                "kind": "string_or_null",
                "format": "<bits>|<k_bits>,<v_bits>",
            },
        }
    return {}


def _normalize_gguf_cache_type_name(cache_type: str) -> str:
    normalized = cache_type.strip().lower()
    if normalized == "":
        raise ValueError("GGUF cache type must be a non-empty GGML type name")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in normalized):
        raise ValueError("GGUF cache type must contain only letters, digits, and underscores")
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


@dataclass
class ModelRuntimeState:
    resolved_backend: str
    configured_enabled: bool
    lifecycle: str = "unloaded"
    inflight_requests: int = 0
    last_error: str | None = None
    artifact_size_mib: int | None = None
    observed_vram_mib: int | None = None
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
