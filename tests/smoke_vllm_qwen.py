"""End-to-end smoke test for the vllm backend on a real GPU.

Not part of the unit suite — this actually loads Qwen2.5-VL-3B-Instruct,
allocates VRAM, and runs inference. Run manually:

    .venv/bin/python tests/smoke_vllm_qwen.py
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path

from PIL import Image, ImageDraw

from app.config import AppSettings
from app.config import DecodingDefaults
from app.config import EngineSettings
from app.config import ModelSettings
from app.engine.vllm import VllmEngine
from app.schemas import DecodingParams
from app.schemas import ImageContent
from app.schemas import ImageUrlSpec
from app.schemas import ResponseRequest
from app.schemas import TextContent


def _build_settings() -> AppSettings:
    return AppSettings(
        engine=EngineSettings(
            backend="vllm",
            decoding=DecodingDefaults(
                top_p=0.95,
                temperature=0.2,
                max_tokens=128,
            ),
            models={
                "qwen2.5-vl-3b": ModelSettings(
                    model_path=None,
                    backend="vllm",
                    prompt_format="generic",
                    modalities=("text", "image"),
                    enabled=True,
                    vllm_model="Qwen/Qwen2.5-VL-3B-Instruct",
                    vllm_dtype="bfloat16",
                    vllm_gpu_memory_utilization=0.7,
                    vllm_max_model_len=4096,
                    vllm_limit_mm_per_prompt=(("image", 2),),
                ),
            },
        ),
    )


def _data_url_from_synthetic_image() -> str:
    image = Image.new("RGB", (320, 160), (220, 220, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 300, 140), outline=(20, 20, 20), width=3)
    draw.text((40, 60), "DANGER!", fill=(220, 20, 20))
    draw.text((40, 90), "HÆTTA!", fill=(220, 20, 20))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def main() -> None:
    print("[smoke] Building engine (this takes ~30-60s on first load) ...")
    started = time.perf_counter()
    settings = _build_settings()
    engine = VllmEngine(settings)
    print(f"[smoke] Engine ready after {time.perf_counter() - started:.1f}s")

    try:
        # 1. Text-only
        print("\n[smoke] === Text-only inference ===")
        t0 = time.perf_counter()
        text_request = ResponseRequest(
            model="qwen2.5-vl-3b",
            input="What is the capital of France? Reply in one short sentence.",
            instructions="You are a concise assistant.",
            decoding=DecodingParams(temperature=0.1, max_tokens=64),
        )
        text_result = engine.complete(text_request)
        t1 = time.perf_counter()
        print(f"[smoke] Output: {text_result.text!r}")
        print(
            f"[smoke] Wall: {(t1 - t0) * 1000:.0f}ms, "
            f"prompt_tokens={text_result.metrics.engine_prompt_tokens}, "
            f"output_tokens={text_result.metrics.engine_output_tokens}, "
            f"tok/s={text_result.metrics.engine_tokens_per_second}"
        )

        # 2. Multimodal
        print("\n[smoke] === Multimodal inference (synthetic warning sign) ===")
        t0 = time.perf_counter()
        mm_request = ResponseRequest(
            model="qwen2.5-vl-3b",
            input=[
                TextContent(text="Describe this image briefly in English."),
                ImageContent(image_url=ImageUrlSpec(url=_data_url_from_synthetic_image())),
            ],
            instructions="You are a helpful image describer.",
            decoding=DecodingParams(temperature=0.1, max_tokens=128),
        )
        mm_result = engine.complete(mm_request)
        t1 = time.perf_counter()
        print(f"[smoke] Output: {mm_result.text!r}")
        print(
            f"[smoke] Wall: {(t1 - t0) * 1000:.0f}ms, "
            f"prompt_tokens={mm_result.metrics.engine_prompt_tokens}, "
            f"output_tokens={mm_result.metrics.engine_output_tokens}, "
            f"tok/s={mm_result.metrics.engine_tokens_per_second}"
        )

    finally:
        print("\n[smoke] Shutting down engine ...")
        engine.close()
        print("[smoke] Done")


if __name__ == "__main__":
    main()
