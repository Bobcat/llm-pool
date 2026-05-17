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
from .common import ModelStateError
from .common import ResolvedDecoding
from .common import UnknownModelError
from .ct2 import Ct2Engine
from .ct2 import Ct2ModelRuntime
from .exllamav3 import ExLlamaV3Engine
from .exllamav3 import ExLlamaV3ModelRuntime
from .llamacpp import LlamaCppEngine
from .llamacpp import LlamaCppModelRuntime
from .openai_compatible import OpenAICompatibleEngine
from .openai_compatible import OpenAICompatibleModelRuntime
from .router import ModelRouterEngine
from .stub import StubEngine


def build_engine(settings):
    if settings.engine.backend == "stub" and not settings.engine.models:
        return StubEngine(settings)
    return ModelRouterEngine(settings)
