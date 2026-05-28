"""Streaming responses + stream_options.include_usage."""
from __future__ import annotations

import time

from openai import OpenAI

from . import config
from .reporter import FAIL, PASS, Report, WARN, short

SECTION = "03_streaming"


def _c() -> OpenAI:
    return OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=60.0, max_retries=0)


def run(report: Report) -> None:
    report.section(
        SECTION,
        "03 · Streaming",
        "Server-Sent Events streaming over the OpenAI SDK plus usage delivery via stream_options.",
    )

    # 1. Plain stream
    try:
        chunks: list[str] = []
        t0 = time.perf_counter()
        ttfb = None
        stream = _c().chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": "Count from one to five, one number per line."}],
            stream=True,
        )
        for chunk in stream:
            if not ttfb:
                ttfb = time.perf_counter() - t0
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                chunks.append(delta)
        elapsed = time.perf_counter() - t0
        text = "".join(chunks)
        if text.strip():
            report.add(
                SECTION,
                "Streaming chat.completions",
                PASS,
                f"chunks={len(chunks)} | ttfb={ttfb:.2f}s | total={elapsed:.2f}s | "
                f"text={short(text, 100)}",
            )
        else:
            report.add(SECTION, "Streaming chat.completions", FAIL, "empty stream")
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "Streaming chat.completions", e)

    # 2. stream_options.include_usage — final chunk must carry usage
    try:
        stream = _c().chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": "Say hi."}],
            stream=True,
            stream_options={"include_usage": True},
        )
        last_usage = None
        for chunk in stream:
            if getattr(chunk, "usage", None):
                last_usage = chunk.usage
        if last_usage is not None:
            report.add(
                SECTION,
                "stream_options.include_usage",
                PASS,
                f"prompt={last_usage.prompt_tokens} completion={last_usage.completion_tokens} "
                f"total={last_usage.total_tokens}",
            )
        else:
            report.add(SECTION, "stream_options.include_usage", WARN,
                       "no usage chunk found in stream")
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "stream_options.include_usage", e)

    # 3. Streaming multi-turn
    try:
        c = _c()
        history: list[dict] = [
            {"role": "user", "content": "Remember the number 42."},
        ]
        # First (non-streamed) turn primes the context
        a1 = c.chat.completions.create(model=config.MODEL, messages=history).choices[0].message.content or ""
        history.append({"role": "assistant", "content": a1})
        history.append({"role": "user", "content": "What number did I ask you to remember?"})

        chunks: list[str] = []
        stream = c.chat.completions.create(model=config.MODEL, messages=history, stream=True)
        for chunk in stream:
            d = chunk.choices[0].delta.content if chunk.choices else None
            if d:
                chunks.append(d)
        out = "".join(chunks)
        if "42" in out:
            report.add(SECTION, "Streaming multi-turn", PASS, short(out, 80))
        else:
            report.add(SECTION, "Streaming multi-turn", WARN,
                       f"answer did not contain '42' | {short(out, 80)}")
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "Streaming multi-turn", e)


if __name__ == "__main__":
    r = Report()
    run(r)
    from pathlib import Path
    r.write_markdown(Path("results/test_03_only.md"))
    print("wrote results/test_03_only.md")
