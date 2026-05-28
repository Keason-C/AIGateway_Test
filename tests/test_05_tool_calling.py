"""Function / tool calling.

Notes:
 - We use ZF_TOOL_MODEL if set, otherwise fall back to ZF_MODEL.
 - tool_choice values exercised: "auto", "required", "none", and a specific tool spec.
 - We also feed a tool result back into history and verify the model uses it.
"""
from __future__ import annotations

import json

from openai import OpenAI

from . import config
from .reporter import FAIL, PASS, Report, WARN, short

SECTION = "05_tool_calling"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}

TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Return the current local time for a given timezone.",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": ["timezone"],
        },
    },
}


def _c() -> OpenAI:
    return OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=60.0, max_retries=0)


def run(report: Report) -> None:
    report.section(
        SECTION,
        "05 · Tool / function calling",
        f"Targets assistant: `{config.TOOL_MODEL}` "
        f"({'same as ZF_MODEL' if config.TOOL_MODEL == config.MODEL else 'override via ZF_TOOL_MODEL'}).",
    )

    c = _c()
    model = config.TOOL_MODEL

    # 1. Basic tool call — prompt designed to trigger the tool
    try:
        r = c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What's the weather in Munich right now?"}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
        )
        tcs = r.choices[0].message.tool_calls or []
        if tcs:
            call = tcs[0]
            report.add(
                SECTION,
                "tools + tool_choice=auto → tool_calls returned",
                PASS,
                f"finish_reason={r.choices[0].finish_reason} | "
                f"name={call.function.name} | args={short(call.function.arguments, 80)}",
            )
        else:
            report.add(
                SECTION,
                "tools + tool_choice=auto → tool_calls returned",
                WARN,
                f"no tool_calls | finish_reason={r.choices[0].finish_reason} | "
                f"content={short(r.choices[0].message.content, 80)}",
            )
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "tools + tool_choice=auto", e)

    # 2. tool_choice="required"
    try:
        r = c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hi."}],
            tools=[WEATHER_TOOL],
            tool_choice="required",
        )
        tcs = r.choices[0].message.tool_calls or []
        report.add(
            SECTION,
            "tool_choice=required forces a call",
            PASS if tcs else WARN,
            f"calls={len(tcs)} | finish={r.choices[0].finish_reason}",
        )
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "tool_choice=required", e)

    # 3. tool_choice="none"
    try:
        r = c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What's the weather in Munich?"}],
            tools=[WEATHER_TOOL],
            tool_choice="none",
        )
        tcs = r.choices[0].message.tool_calls or []
        report.add(
            SECTION,
            "tool_choice=none suppresses calls",
            PASS if not tcs else WARN,
            f"calls={len(tcs)} | finish={r.choices[0].finish_reason} | "
            f"content={short(r.choices[0].message.content, 80)}",
        )
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "tool_choice=none", e)

    # 4. Specific tool selection
    try:
        r = c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Tell me a fact."}],
            tools=[WEATHER_TOOL, TIME_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_time"}},
        )
        tcs = r.choices[0].message.tool_calls or []
        ok = any(t.function.name == "get_time" for t in tcs)
        report.add(
            SECTION,
            "tool_choice=specific tool",
            PASS if ok else WARN,
            f"calls={[t.function.name for t in tcs]}",
        )
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "tool_choice=specific tool", e)

    # 5. Parallel tool calls — two independent calls in one response
    try:
        r = c.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": "Get the weather in Munich AND the time in Asia/Shanghai in one go.",
            }],
            tools=[WEATHER_TOOL, TIME_TOOL],
            tool_choice="auto",
            parallel_tool_calls=True,
        )
        tcs = r.choices[0].message.tool_calls or []
        report.add(
            SECTION,
            "parallel_tool_calls=true",
            PASS if len(tcs) >= 2 else WARN,
            f"calls={[t.function.name for t in tcs]}",
        )
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "parallel_tool_calls=true", e)

    # 6. Tool result continuation — does the model absorb a tool_result?
    try:
        history = [{"role": "user", "content": "What's the weather in Munich?"}]
        r1 = c.chat.completions.create(
            model=model, messages=history, tools=[WEATHER_TOOL], tool_choice="auto",
        )
        tcs = r1.choices[0].message.tool_calls or []
        if not tcs:
            report.add(SECTION, "tool result continuation", WARN,
                       "first turn did not call the tool — cannot test continuation")
        else:
            call = tcs[0]
            history.append({
                "role": "assistant",
                "content": r1.choices[0].message.content or "",
                "tool_calls": [{
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }],
            })
            history.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps({"temperature_c": 22, "condition": "sunny"}),
            })
            r2 = c.chat.completions.create(model=model, messages=history, tools=[WEATHER_TOOL])
            final = r2.choices[0].message.content or ""
            ok = "22" in final or "sunny" in final.lower()
            report.add(
                SECTION,
                "tool result continuation",
                PASS if ok else WARN,
                f"final answer: {short(final, 120)}",
            )
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "tool result continuation", e)


if __name__ == "__main__":
    r = Report()
    run(r)
    from pathlib import Path
    r.write_markdown(Path("results/test_05_only.md"))
    print("wrote results/test_05_only.md")
