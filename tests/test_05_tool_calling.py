"""Function / tool calling.

Notes:
 - We use ZF_TOOL_MODEL if set, otherwise fall back to ZF_MODEL.
 - tool_choice values exercised: "auto", "required", "none", and a specific tool spec.
 - We also feed a tool result back into history and verify the model uses it.

Wire format note (round-3 fix):
 - On the assistant-with-tool_calls turn, we **omit the `content` field entirely**
   when the model didn't return any text. MAF 1.0's OpenAIChatCompletionClient
   does the same, and the OpenAI OpenAPI spec only requires `role` for that
   message variant. Sending `content: ""` triggers 500 on some gateways that
   strictly validate the body upstream.
"""
from __future__ import annotations

import json

import httpx
from openai import OpenAI

from . import config
from .reporter import FAIL, INFO, PASS, Report, WARN, short

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

    # 6. Tool result continuation — three wire-format variants
    #
    # Round 2 sent `content: ""` on the assistant turn and got 500. MAF source
    # omits content entirely. Try all three of the spec-allowed forms so we know
    # which one this gateway is happy with:
    #   A. content key OMITTED entirely (MAF's choice, spec-compliant)
    #   B. content explicitly null   (also spec-compliant)
    #   C. content empty string ""    (technically spec-compliant; previously 500'd)
    def _continuation(variant_label: str, build_assistant_msg):
        history = [{"role": "user", "content": "What's the weather in Munich?"}]
        try:
            r1 = c.chat.completions.create(
                model=model, messages=history, tools=[WEATHER_TOOL], tool_choice="auto",
            )
        except Exception as e:  # noqa: BLE001
            report.capture_exception(SECTION, f"tool continuation {variant_label} (turn 1)", e)
            return
        tcs = r1.choices[0].message.tool_calls or []
        if not tcs:
            report.add(SECTION, f"tool continuation {variant_label}", WARN,
                       "first turn did not call the tool — cannot test continuation")
            return
        call = tcs[0]
        history.append(build_assistant_msg(r1, call))
        history.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps({"temperature_c": 22, "condition": "sunny"}),
        })
        try:
            r2 = c.chat.completions.create(model=model, messages=history, tools=[WEATHER_TOOL])
            final = r2.choices[0].message.content or ""
            ok = "22" in final or "sunny" in final.lower()
            report.add(
                SECTION,
                f"tool continuation {variant_label}",
                PASS if ok else WARN,
                f"final answer: {short(final, 120)}",
            )
        except Exception as e:  # noqa: BLE001
            report.capture_exception(SECTION, f"tool continuation {variant_label} (turn 2)", e)

    # Variant A: omit `content` entirely — MAF's wire shape
    _continuation(
        "A · content OMITTED (MAF default)",
        lambda r1, call: {
            "role": "assistant",
            "tool_calls": [{
                "id": call.id, "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }],
        },
    )
    # Variant B: explicit null
    _continuation(
        "B · content=None (explicit null)",
        lambda r1, call: {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call.id, "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }],
        },
    )
    # Variant C: empty string (the one that previously 500'd)
    _continuation(
        "C · content='' (empty string — previously 500'd)",
        lambda r1, call: {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call.id, "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }],
        },
    )

    # 7. Raw httpx: send a hand-built tool_result continuation and dump the
    # literal wire body so the user can see exactly what's going over.
    try:
        weather_call_id = "call_zf_test_001"
        body = {
            "model": model,
            "messages": [
                {"role": "user", "content": "What's the weather in Munich?"},
                {
                    # The MAF wire shape: NO content key.
                    "role": "assistant",
                    "tool_calls": [{
                        "id": weather_call_id,
                        "type": "function",
                        "function": {"name": "get_weather",
                                     "arguments": json.dumps({"city": "Munich"})},
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": weather_call_id,
                    "content": json.dumps({"temperature_c": 22, "condition": "sunny"}),
                },
            ],
            "tools": [WEATHER_TOOL],
        }
        wire_preview = short(json.dumps(body, ensure_ascii=False), 220)
        with httpx.Client(timeout=60.0) as h:
            resp = h.post(
                config.BASE_URL + "/chat/completions",
                json=body,
                headers={
                    "Authorization": f"Bearer {config.API_KEY}",
                    "Content-Type": "application/json",
                },
            )
        if resp.status_code == 200:
            data = resp.json()
            final = (data["choices"][0]["message"].get("content") or "")
            ok = "22" in final or "sunny" in final.lower()
            report.add(
                SECTION,
                "raw httpx tool continuation (MAF-shape, content omitted)",
                PASS if ok else WARN,
                f"status=200 | final={short(final, 80)} | sent={wire_preview}",
            )
        else:
            report.add(
                SECTION,
                "raw httpx tool continuation (MAF-shape, content omitted)",
                FAIL,
                f"status={resp.status_code} | body={short(resp.text, 160)} | sent={wire_preview}",
            )
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "raw httpx tool continuation", e)


if __name__ == "__main__":
    r = Report()
    run(r)
    from pathlib import Path
    r.write_markdown(Path("results/test_05_only.md"))
    print("wrote results/test_05_only.md")
