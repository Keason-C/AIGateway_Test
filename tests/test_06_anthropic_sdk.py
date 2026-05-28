"""Confirm /v1/messages also works via the official Anthropic SDK.

The gateway uses the same API key for both protocols; only the base URL differs
(no `/v1` suffix because the SDK appends it).
"""
from __future__ import annotations

import anthropic

from . import config
from .reporter import FAIL, PASS, Report, WARN, short

SECTION = "06_anthropic_sdk"


def _c() -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=config.ANTHROPIC_BASE_URL,
        api_key=config.API_KEY,
        timeout=60.0,
        max_retries=0,
    )


def run(report: Report) -> None:
    report.section(
        SECTION,
        "06 · Anthropic SDK",
        f"Uses the Anthropic Python SDK against `{config.ANTHROPIC_BASE_URL}` (no /v1 suffix).",
    )

    # 1. Simple message
    try:
        msg = _c().messages.create(
            model=config.MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": "Reply with the word: pong"}],
        )
        text = msg.content[0].text if msg.content else ""
        ok = "pong" in (text or "").lower()
        report.add(
            SECTION,
            "messages.create (non-stream)",
            PASS if ok else WARN,
            f"stop_reason={msg.stop_reason} | "
            f"usage={msg.usage.input_tokens}/{msg.usage.output_tokens} | "
            f"text={short(text, 80)}",
        )
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "messages.create (non-stream)", e)

    # 2. Streaming
    try:
        chunks: list[str] = []
        with _c().messages.stream(
            model=config.MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": "Count to 3."}],
        ) as stream:
            for text in stream.text_stream:
                chunks.append(text)
        joined = "".join(chunks)
        report.add(
            SECTION,
            "messages.stream",
            PASS if joined.strip() else FAIL,
            f"chunks={len(chunks)} | text={short(joined, 80)}",
        )
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "messages.stream", e)

    # 3. Tool use (Anthropic format)
    try:
        msg = _c().messages.create(
            model=config.MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": "What is the weather in Munich?"}],
            tools=[{
                "name": "get_weather",
                "description": "Get current weather.",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }],
        )
        tool_blocks = [b for b in msg.content if getattr(b, "type", None) == "tool_use"]
        report.add(
            SECTION,
            "messages.create with tools (Anthropic schema)",
            PASS if tool_blocks else WARN,
            f"stop_reason={msg.stop_reason} | tool_calls={len(tool_blocks)}",
        )
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "messages.create with tools", e)


if __name__ == "__main__":
    r = Report()
    run(r)
    from pathlib import Path
    r.write_markdown(Path("results/test_06_only.md"))
    print("wrote results/test_06_only.md")
