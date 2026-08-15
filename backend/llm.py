"""Thin wrapper around llama-cpp-python.

One model instance is shared by every AI in the app (both debaters and the
judge); calls are strictly sequential. A fake implementation is
available (FAKE_LLM=1) so the whole pipeline can be exercised without
downloading a model.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from typing import Iterator

from .config import FAKE_LLM, N_CTX, N_GPU_LAYERS, N_THREADS


class LLM:
    def __init__(self, model_path: str):
        from llama_cpp import Llama

        self.llama = Llama(
            model_path=str(model_path),
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=N_GPU_LAYERS,
            verbose=False,
        )

    def chat_stream(
        self,
        messages: list[dict],
        max_tokens: int = 700,
        temperature: float = 0.8,
        json_mode: bool = False,
    ) -> Iterator[str]:
        kwargs = {}
        if json_mode:
            # llama.cpp grammar-constrains the output to valid JSON.
            kwargs["response_format"] = {"type": "json_object"}
        stream = self.llama.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            text = delta.get("content")
            if text:
                yield text

    def chat(self, messages, **kwargs) -> str:
        return "".join(self.chat_stream(messages, **kwargs))


class FakeLLM:
    """Deterministic-ish canned responses for development and testing."""

    def __init__(self, model_path: str = ""):
        pass

    def _topic(self, messages) -> str:
        for m in messages:
            hit = re.search(r'motion[^:"]*:\s*"([^"]+)"', m.get("content", ""))
            if hit:
                return hit.group(1)
        return "the motion"

    def chat_stream(self, messages, max_tokens=700, temperature=0.8,
                    json_mode=False) -> Iterator[str]:
        seed = int(hashlib.md5(json.dumps(messages).encode()).hexdigest(), 16)
        rng = random.Random(seed)
        prompt = messages[-1].get("content", "") if messages else ""
        if json_mode:
            # A per-criterion judging call: reasoning + score for each side.
            hit = re.search(r"integer from 0 to (\d+)", prompt)
            mx = int(hit.group(1)) if hit else 20
            result = {
                side: {
                    "reasoning": (
                        f"{side.upper()} handled this criterion "
                        f"{rng.choice(['capably', 'unevenly', 'strongly'])}; "
                        "this is a canned evaluation from the fake LLM used "
                        "in development mode."),
                    "score": rng.randint(mx // 2, mx),
                }
                for side in ("pro", "con")
            }
            text = json.dumps(result)
        elif "YOUR COMPLETED SCORECARD" in prompt:
            text = (
                "Weighing my scorecard as a whole, PRO built the more "
                "evidence-driven case while CON pressed the sharper "
                "rebuttals; both sides left some attacks unanswered. This "
                "is a canned judge's summary from the fake LLM used in "
                "development mode, so my ultimate finding simply follows "
                "the canned scores above."
            )
        else:
            topic = self._topic(messages)
            openers = [
                f"Ladies and gentlemen, when we examine the question of {topic}, the evidence points in one clear direction.",
                f"My opponent would have you believe otherwise, but on the matter of {topic} their case simply does not hold together.",
                f"Let us be honest about what is truly at stake in {topic}.",
            ]
            body = (
                " First, the research before us shows the practical stakes are real and measurable."
                " Second, the strongest counterarguments rest on assumptions that collapse under scrutiny."
                " Third, history teaches us that caution and evidence, not rhetoric, should guide this decision."
                " For these reasons, my side of this motion stands firm."
            )
            text = rng.choice(openers) + body
        for word in re.findall(r"\S+\s*", text):
            time.sleep(0.01)
            yield word

    def chat(self, messages, **kwargs) -> str:
        return "".join(self.chat_stream(messages, **kwargs))


def load_llm(model_path) -> "LLM | FakeLLM":
    return FakeLLM(model_path) if FAKE_LLM else LLM(model_path)
