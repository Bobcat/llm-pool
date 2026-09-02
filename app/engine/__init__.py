from __future__ import annotations

from .common import LOGGER
from .common import BackendExecutionError
from .common import _empty_cuda_allocator_cache
from .common import _estimate_model_artifact_size_mib
from .common import _exception_message
from .common import _merge_stop_strings
from .common import _native_stop_strings
from .common import _query_gpu_memory
from .common import _query_primary_gpu_used_mib
from .common import ModelRuntimeState
from .common import RequestAdmissionError
from .common import SettingsReloadConflictError
from .common import SettingsReloadValidationError
from .common import ModelStateError
from .common import ResolvedDecoding
from .common import UnknownModelError
from .ct2 import Ct2Engine
from .ct2 import Ct2ModelRuntime
from .exllamav3 import ExLlamaV3Engine
from .exllamav3 import ExLlamaV3ModelRuntime
from .llama_cpp import LlamaCppEngine
from .llama_cpp import LlamaCppModelRuntime
from .llama_server import LlamaServerEngine
from .llama_server import LlamaServerModelRuntime
from .openai_remote import OpenAIRemoteEngine
from .openai_remote import OpenAIRemoteModelRuntime
from .router import ModelRouterEngine
from .stub import StubEngine
from .trtllm_serve import TrtllmServeEngine
from .trtllm_serve import TrtllmServeModelRuntime
from .vllm import VllmEngine
from .vllm import VllmModelRuntime
from .vllm_serve import VllmServeEngine
from .vllm_serve import VllmServeModelRuntime


def build_engine(settings):
    return ModelRouterEngine(settings)
