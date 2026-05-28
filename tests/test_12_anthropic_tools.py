"""Tool / function calling via the **Anthropic** SDK — full continuation round-trip.

This is the direct counterpart to `test_05_tool_calling` (which exercises the
OpenAI Chat Completions wire format). Round-3 results showed:

  - OpenAI side: turn-1 `tool_calls` works ✅, but feeding `role:"tool"` back
    in turn-2 → **gateway 500** across every content-shape variant.
  - Anthropic side (test_06): turn-1 `tool_use` works ✅, **but we never
    tested the continuation**.

This module closes that gap. The Anthropic continuation wire shape is:

    user      → "What's the weather in Munich?"
    assistant → [text?, tool_use(id, name, input)]   (echo back as-is)
    user      → [tool_result(tool_use_id, content)]
    assistant → final natural-language reply that references the tool output

If turn-2 also 500s, the gateway is broken on BOTH protocols. If turn-2
succeeds, the right Round-4 advice is "use the Anthropic protocol for tools
on this gateway".
"""
from __future__ import annotations

import json

import anthropic

from . import config
from .reporter import FAIL, INFO, PASS, Report, WARN, short

SECTION = "12_anthropic_tools"

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a given city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}

TIME_TOOL = {
    "name": "get_time",
    "description": "Return the current local time for a given IANA timezone.",
    "input_schema": {
        "type": "object",
        "properties": {"timezone": {"type": "string"}},
        "required": ["timezone"],
    },
}


def _c() -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=config.ANTHROPIC_BASE_URL,
        api_key=config.API_KEY,
        timeout=60.0,
        max_retries=0,
    )


def _content_blocks_for_echo(message) -> list[dict]:
    """Re-serialize the assistant's response blocks back into wire form.

    The Anthropic SDK returns rich objects (`TextBlock`, `ToolUseBlock`).
    Continuation requires us to send the same content array back; the SDK
    accepts the dict shape directly.
    """
    out: list[dict] = []
    for b in message.content:
        t = getattr(b, "type", None)
        if t == "text":
            out.append({"type": "text", "text": b.text})
        elif t == "tool_use":
            out.append({
                "type": "tool_use",
                "id": b.id,
                "name": b.name,
                "input": b.input,
            })
        else:
            # Unknown block — best-effort dump
            out.append(getattr(b, "model_dump", lambda: {"type": t})())
    return out


def _first_tool_use(message):
    for b in message.content:
        if getattr(b, "type", None) == "tool_use":
            return b
    return None


def run(report: Report) -> None:
    report.section(
        SECTION,
        "12 · Anthropic SDK · tool calling round-trip",
        "Full two-turn tool round-trip (tool_use → tool_result → final answer) via the "
        f"Anthropic SDK against `{config.ANTHROPIC_BASE_URL}`. Direct counterpart to "
        "section 05 — same gateway, different protocol.",
    )

    c = _c()
    model = config.TOOL_MODEL

    # ── 1. Turn 1: model issues tool_use ─────────────────────────────────────
    history: list[dict] = [
        {"role": "user", "content": "What's the weather in Munich right now?"},
    ]
    try:
        r1 = c.messages.create(
            model=model,
            max_tokens=400,
            messages=history,
            tools=[WEATHER_TOOL],
        )
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "turn 1 · request tool_use", e)
        report.add(SECTION, "turn 2 · tool_result continuation", FAIL,
                   "skipped — turn 1 already failed")
        return

    tool_block = _first_tool_use(r1)
    if not tool_block:
        report.add(
            SECTION,
            "turn 1 · request tool_use",
            WARN,
            f"no tool_use block | stop_reason={r1.stop_reason} | "
            f"types={[getattr(b, 'type', '?') for b in r1.content]}",
        )
        report.add(SECTION, "turn 2 · tool_result continuation", WARN,
                   "skipped — model did not call the tool")
        return

    report.add(
        SECTION,
        "turn 1 · request tool_use",
        PASS,
        f"stop_reason={r1.stop_reason} | tool_use.name={tool_block.name} | "
        f"input={short(json.dumps(tool_block.input), 80)} | "
        f"usage={r1.usage.input_tokens}/{r1.usage.output_tokens}",
    )

    # ── 2. Turn 2: echo assistant + send tool_result back ────────────────────
    history.append({
        "role": "assistant",
        "content": _content_blocks_for_echo(r1),
    })
    history.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_block.id,
            "content": json.dumps({"temperature_c": 22, "condition": "sunny"}),
        }],
    })
    try:
        r2 = c.messages.create(
            model=model,
            max_tokens=400,
            messages=history,
            tools=[WEATHER_TOOL],
        )
        final_text = ""
        for b in r2.content:
            if getattr(b, "type", None) == "text":
                final_text += b.text
        ok = ("22" in final_text) or ("sunny" in final_text.lower())
        report.add(
            SECTION,
            "turn 2 · tool_result continuation",
            PASS if ok else WARN,
            f"stop_reason={r2.stop_reason} | "
            f"usage={r2.usage.input_tokens}/{r2.usage.output_tokens} | "
            f"final={short(final_text, 140)}",
        )
    except Exception as e:  # noqa: BLE001
        # THIS is the headline result — if it 500s, the gateway's broken on
        # both protocols and the prior "Anthropic works" claim is wrong.
        report.capture_exception(SECTION, "turn 2 · tool_result continuation", e)

    # ── 3. tool_choice variants ──────────────────────────────────────────────
    # Anthropic uses {"type": "auto"|"any"|"tool", ...}. "any" ≈ OpenAI "required".
    for label, choice in [
        ("tool_choice={'type':'auto'}", {"type": "auto"}),
        ("tool_choice={'type':'any'} (force any tool)", {"type": "any"}),
        ("tool_choice={'type':'tool', 'name':'get_time'} (specific)",
         {"type": "tool", "name": "get_time"}),
    ]:
        try:
            r = c.messages.create(
                model=model,
                max_tokens=300,
                messages=[{"role": "user", "content": "Tell me a fact."}],
                tools=[WEATHER_TOOL, TIME_TOOL],
                tool_choice=choice,
            )
            tu_blocks = [b for b in r.content if getattr(b, "type", None) == "tool_use"]
            if choice["type"] == "tool":
                ok = any(b.name == choice["name"] for b in tu_blocks)
            elif choice["type"] == "any":
                ok = bool(tu_blocks)
            else:  # auto — just want a successful call, tools optional
                ok = True
            report.add(
                SECTION,
                label,
                PASS if ok else WARN,
                f"stop={r.stop_reason} | tools={[b.name for b in tu_blocks]}",
            )
        except Exception as e:  # noqa: BLE001
            report.capture_exception(SECTION, label, e)

    # ── 4. Parallel tool use — model invokes 2 tools in one turn ─────────────
    try:
        r = c.messages.create(
            model=model,
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": "Get the weather in Munich AND the time in Asia/Shanghai in one go.",
            }],
            tools=[WEATHER_TOOL, TIME_TOOL],
            tool_choice={"type": "auto"},
        )
        tu_blocks = [b for b in r.content if getattr(b, "type", None) == "tool_use"]
        report.add(
            SECTION,
            "parallel tool_use (2 tools in one assistant turn)",
            PASS if len(tu_blocks) >= 2 else WARN,
            f"stop={r.stop_reason} | tools={[b.name for b in tu_blocks]}",
        )
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "parallel tool_use", e)

    # ── 5. Streaming with tool_use — verify the stream event surface ─────────
    try:
        tool_use_events = 0
        input_json_deltas = 0
        with c.messages.stream(
            model=model,
            max_tokens=400,
            messages=[{"role": "user", "content": "What's the weather in Berlin?"}],
            tools=[WEATHER_TOOL],
        ) as stream:
            for event in stream:
                etype = getattr(event, "type", "")
                if etype == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if getattr(block, "type", None) == "tool_use":
                        tool_use_events += 1
                elif etype == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if getattr(delta, "type", None) == "input_json_delta":
                        input_json_deltas += 1
            final = stream.get_final_message()
        tu_blocks = [b for b in final.content if getattr(b, "type", None) == "tool_use"]
        ok = bool(tu_blocks)
        report.add(
            SECTION,
            "messages.stream with tools",
            PASS if ok else WARN,
            f"tool_use blocks={len(tu_blocks)} | "
            f"content_block_start(tool_use)={tool_use_events} | "
            f"input_json_delta events={input_json_deltas} | "
            f"final stop={final.stop_reason}",
        )
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "messages.stream with tools", e)

    # ── 6. Three-turn chain: tool_use → tool_result → user follow-up ─────────
    # If turn-2 already passed, this confirms multi-step agent loops survive.
    try:
        chain: list[dict] = [
            {"role": "user", "content": "What's the weather in Munich?"},
        ]
        s1 = c.messages.create(
            model=model, max_tokens=300, messages=chain, tools=[WEATHER_TOOL],
        )
        tb = _first_tool_use(s1)
        if not tb:
            report.add(SECTION, "three-turn agent chain", WARN,
                       "model did not call the tool on step 1 — skipping the rest")
        else:
            chain.append({"role": "assistant", "content": _content_blocks_for_echo(s1)})
            chain.append({
                "role": "user",
                "content": [{
                    "type": "tool_result", "tool_use_id": tb.id,
                    "content": json.dumps({"temperature_c": 5, "condition": "snowy"}),
                }],
            })
            s2 = c.messages.create(
                model=model, max_tokens=300, messages=chain, tools=[WEATHER_TOOL],
            )
            chain.append({"role": "assistant", "content": _content_blocks_for_echo(s2)})
            chain.append({"role": "user",
                          "content": "Should I wear a coat? Answer yes or no first."})
            s3 = c.messages.create(
                model=model, max_tokens=200, messages=chain, tools=[WEATHER_TOOL],
            )
            final = "".join(b.text for b in s3.content
                            if getattr(b, "type", None) == "text")
            ok = final.strip().lower().startswith(("yes", "no")) or "coat" in final.lower()
            report.add(
                SECTION,
                "three-turn agent chain",
                PASS if ok else WARN,
                f"step3 stop={s3.stop_reason} | final={short(final, 120)}",
            )
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "three-turn agent chain", e)


if __name__ == "__main__":
    r = Report()
    run(r)
    from pathlib import Path
    r.write_markdown(Path("results/test_12_only.md"))
    print("wrote results/test_12_only.md")
