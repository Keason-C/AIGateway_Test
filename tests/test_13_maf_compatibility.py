"""MAF ↔ ZF-gateway compatibility matrix — the headline section.

Goal: answer one question cleanly — *does the Microsoft Agent Framework work
against our gateway, and which client do you pick?*

Confirmed findings (this section encodes them so they stay verified):

  • AnthropicClient + ``additional_beta_flags=[]`` → WORKS for everything,
    including @tool. This is the proven production construction.
  • AnthropicClient with MAF's DEFAULT beta flags (mcp-client / code-execution)
    → @tool returns HTTP 500. Plain chat tolerates the beta header; the moment
    `tools` are attached the gateway 500s. We keep one "default flags" row as a
    root-cause contrast so the report proves the beta header is the culprit.
  • OpenAIChatClient (Responses API, /responses) → unusable (404 / SDK parse
    error). Despite the name it is NOT Chat Completions.
  • OpenAIChatCompletionClient (Chat Completions) → single-turn + streaming
    work, but @tool 500s on the gateway's OpenAI tool path. A raw-`openai`-SDK
    round-trip below reproduces it WITHOUT MAF (turn-1 vs turn-2 isolated), so
    it's a gateway-shim limitation, not a MAF-injected field.

Construction notes:
  • OpenAI clients: base_url=ZF .../v1, api_key, model=assistant-name.
    Do NOT set max_tokens for the OpenAI SDK — the gateway 500s on it.
  • AnthropicClient(model=…, anthropic_client=AsyncAnthropic(base_url=…),
    additional_beta_flags=[]); set max_tokens via default_options (MAF's
    default of 1024 truncates multi-section answers).
"""
from __future__ import annotations

import asyncio
import importlib
import json
from typing import Annotated

from . import config
from .reporter import FAIL, INFO, PASS, Report, SKIP, WARN, short

SECTION = "13_maf_compatibility"

# Match the production construction: MAF's AnthropicClient defaults to
# max_tokens=1024 (truncates) — set it explicitly. NB: only the Anthropic
# agents get this; setting max_tokens on the OpenAI SDK 500s this gateway.
ANTHROPIC_MAX_TOKENS = 32768
ANTHROPIC_AGENT_KWARGS = {"default_options": {"max_tokens": ANTHROPIC_MAX_TOKENS}}


# ── import helpers (canonical paths first, tolerant of pre-release drift) ────
def _imp_agent_tool():
    """Canonical `agent_framework.Agent` + `@tool` — the GA 1.0 shape."""
    from agent_framework import Agent, tool  # type: ignore
    return Agent, tool


def _imp_field():
    try:
        from pydantic import Field  # type: ignore
        return Field
    except Exception:  # noqa: BLE001
        return None


def _imp_workflow_builder():
    """WorkflowBuilder (preferred) or SequentialBuilder fallback."""
    candidates = [
        ("agent_framework", "WorkflowBuilder", "workflow"),
        ("agent_framework.workflows", "WorkflowBuilder", "workflow"),
        ("agent_framework", "SequentialBuilder", "sequential"),
        ("agent_framework.workflows", "SequentialBuilder", "sequential"),
    ]
    for module_name, cls_name, kind in candidates:
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                return cls, kind, f"{module_name}.{cls_name}"
        except Exception:  # noqa: BLE001
            continue
    return None, None, ""


def _is_404(exc: Exception) -> bool:
    msg = str(exc).lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return status == 404 or "404" in msg or "not found" in msg or "/responses" in msg


# ── capability checks (each builds a fresh Agent off the shared client) ──────
async def _connectivity(
    report, Agent, label, protocol, client, *, expect_unsupported: bool, agent_kwargs=None
) -> bool:
    """Single-turn agent.run as the connectivity probe. Returns True if the
    client reached the gateway without raising."""
    name = f"{label} · connectivity (single-turn)"
    try:
        agent = Agent(client=client, name="ZFProbe", instructions="You are concise.",
                      **(agent_kwargs or {}))
        result = await agent.run("Reply with the word: pong")
        text = getattr(result, "text", None) or str(result)
        ok = "pong" in text.lower()
        if expect_unsupported:
            report.add(SECTION, name, WARN,
                       f"UNEXPECTED 200 — gateway appears to implement {protocol}: {short(text, 80)}")
        else:
            report.add(SECTION, name, PASS if ok else WARN, short(text, 120))
        return True
    except Exception as e:  # noqa: BLE001
        if expect_unsupported:
            # We never expect this client to work here — any failure is the
            # expected outcome, recorded as INFO (not a red FAIL).
            kind = "404" if _is_404(e) else "error"
            report.add(SECTION, name, INFO,
                       f"{protocol} unusable on this gateway as expected ({kind}); "
                       f"use OpenAIChatCompletionClient / AnthropicClient. ({short(str(e), 140)})")
        else:
            report.capture_exception(SECTION, name, e)
        return False


async def _cap_stream(report, Agent, label, client, agent_kwargs=None) -> None:
    try:
        agent = Agent(client=client, name="ZFStream", instructions="You are concise.",
                      **(agent_kwargs or {}))
        chunks: list[str] = []
        async for chunk in agent.run("Count from 1 to 3.", stream=True):
            t = getattr(chunk, "text", None)
            if t:
                chunks.append(t)
        joined = "".join(chunks)
        report.add(SECTION, f"{label} · streaming", PASS if joined.strip() else FAIL,
                   f"chunks={len(chunks)} text={short(joined, 100)}")
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, f"{label} · streaming", e)


async def _cap_tool(report, Agent, tool, Field, label, client, agent_kwargs=None) -> None:
    """The headline check: @tool function calling end-to-end (square(7)=49).

    Canonical skill shape: @tool + Annotated[T, Field(description=…)] + docstring.
    approval_mode="never_require" (the implicit default) is set explicitly so an
    unattended run never blocks waiting for an approval prompt.
    """
    try:
        if Field is not None:
            @tool(approval_mode="never_require")
            def square(n: Annotated[int, Field(description="The integer to square.")]) -> int:
                """Return n squared."""
                return n * n
        else:
            @tool(approval_mode="never_require")
            def square(n: Annotated[int, "The integer to square."]) -> int:  # type: ignore[no-redef]
                """Return n squared."""
                return n * n

        agent = Agent(
            client=client,
            name="ZFMath",
            instructions="When asked to square a number, call the square tool. "
                         "After it returns, reply with just the number.",
            tools=[square],
            **(agent_kwargs or {}),
        )
        result = await agent.run("Use your tool to square the number 7.")
        text = getattr(result, "text", None) or str(result)
        ok = "49" in text
        report.add(SECTION, f"{label} · @tool call (square(7)=49)",
                   PASS if ok else WARN, short(text, 160))
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, f"{label} · @tool call (square(7)=49)", e)


async def _cap_workflow(report, Agent, label, client, agent_kwargs=None) -> None:
    BuilderCls, kind, _ = _imp_workflow_builder()
    if BuilderCls is None:
        report.add(SECTION, f"{label} · workflow (2-agent)", SKIP,
                   "neither WorkflowBuilder nor SequentialBuilder importable")
        return
    try:
        kw = agent_kwargs or {}
        writer = Agent(client=client, name="Writer",
                       instructions="Write one short sentence about the topic.", **kw)
        shortener = Agent(client=client, name="Shortener",
                          instructions="Rewrite the sentence in 3 words or fewer.", **kw)

        workflow = None
        build_err: Exception | None = None
        if kind == "workflow":
            tries = [
                lambda: BuilderCls(start_executor=writer).add_edge(writer, shortener).build(),
                lambda: BuilderCls().set_start_executor(writer).add_edge(writer, shortener).build(),
            ]
            for t in tries:
                try:
                    workflow = t()
                    break
                except Exception as inner:  # noqa: BLE001
                    build_err = inner
        else:  # sequential
            try:
                workflow = BuilderCls().add_agents([writer, shortener]).build()
            except Exception:  # noqa: BLE001
                try:
                    workflow = BuilderCls().participants([writer, shortener]).build()
                except Exception as inner2:  # noqa: BLE001
                    build_err = inner2

        if workflow is None:
            report.add(SECTION, f"{label} · workflow build", FAIL,
                       f"all builder shapes failed | last={short(repr(build_err), 180)}")
            return

        events = await workflow.run("the moon")
        if hasattr(events, "get_outputs"):
            out = "; ".join(str(o) for o in (events.get_outputs() or []))
        else:
            out = getattr(events, "text", None) or getattr(events, "output", None) or str(events)
        report.add(SECTION, f"{label} · workflow run (writer → shortener) [{kind}]",
                   PASS if str(out).strip() else WARN, short(str(out), 140))
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, f"{label} · workflow", e)


def _openai_raw_tool_roundtrip(report: Report) -> None:
    """Bypass MAF: hand-build an OpenAI tool round-trip with the raw `openai`
    SDK to pinpoint WHERE the gateway 500s on the OpenAI tool path —
      • turn 1: request carries `tools` (model should answer with tool_calls)
      • turn 2: the assistant.tool_calls + role:"tool" continuation
    If turn-1 already 500s it's the tools-in-request shape; if only turn-2 500s
    it's the gateway's continuation shim. Either way this reproduces WITHOUT
    MAF — proving the OpenAI 500 is NOT a MAF-injected field (unlike the
    Anthropic beta-header issue, which clearing the flags fixes)."""
    from openai import OpenAI  # type: ignore

    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    c = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=60.0, max_retries=0)

    # turn 1 — tools in the request
    try:
        r1 = c.chat.completions.create(
            model=config.TOOL_MODEL,
            messages=[{"role": "user", "content": "What's the weather in Munich?"}],
            tools=[weather_tool],
            tool_choice="auto",
        )
        tcs = r1.choices[0].message.tool_calls or []
        if not tcs:
            report.add(SECTION, "raw openai SDK · tool turn-1 (tools in request)", WARN,
                       f"200 but no tool_calls | finish={r1.choices[0].finish_reason}")
            return
        call = tcs[0]
        report.add(SECTION, "raw openai SDK · tool turn-1 (tools in request)", PASS,
                   f"200 | tool={call.function.name} args={short(call.function.arguments, 60)}")
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "raw openai SDK · tool turn-1 (tools in request)", e)
        return

    # turn 2 — role:"tool" continuation (content omitted on the assistant turn,
    # the spec-correct / MAF wire shape)
    history = [
        {"role": "user", "content": "What's the weather in Munich?"},
        {"role": "assistant", "tool_calls": [{
            "id": call.id, "type": "function",
            "function": {"name": call.function.name, "arguments": call.function.arguments},
        }]},
        {"role": "tool", "tool_call_id": call.id,
         "content": json.dumps({"temperature_c": 22, "condition": "sunny"})},
    ]
    try:
        r2 = c.chat.completions.create(model=config.TOOL_MODEL, messages=history, tools=[weather_tool])
        final = r2.choices[0].message.content or ""
        ok = "22" in final or "sunny" in final.lower()
        report.add(SECTION, "raw openai SDK · tool turn-2 (role:tool continuation)",
                   PASS if ok else WARN, short(final, 120))
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "raw openai SDK · tool turn-2 (role:tool continuation)", e)


async def _close(obj) -> None:
    """Best-effort async close of a client's inner HTTP pool — avoids
    'Event loop is closed' noise on Windows ProactorEventLoop after asyncio.run."""
    close_fn = getattr(obj, "close", None)
    if close_fn is not None:
        try:
            await close_fn()
        except Exception:  # noqa: BLE001
            pass


# ── the matrix ───────────────────────────────────────────────────────────────
async def _run_async(report: Report) -> None:
    try:
        Agent, tool = _imp_agent_tool()
    except Exception as e:  # noqa: BLE001
        report.add(SECTION, "import agent_framework (Agent, @tool)", FAIL, short(str(e), 240))
        return
    Field = _imp_field()
    report.add(SECTION, "import agent_framework (Agent, @tool)", PASS,
               f"Field={'yes' if Field else 'no'}")

    # ── Client A: OpenAIChatClient → Responses API (/responses). Expected unusable. ──
    try:
        from agent_framework.openai import OpenAIChatClient  # type: ignore
        label = "OpenAIChatClient (Responses)"
        client = OpenAIChatClient(
            base_url=config.BASE_URL, api_key=config.API_KEY, model=config.MODEL,
        )
        reached = await _connectivity(
            report, Agent, label, "POST /responses", client, expect_unsupported=True
        )
        if reached:
            # Surprising — the gateway DOES speak Responses. Run the full matrix.
            await _cap_stream(report, Agent, label, client)
            await _cap_tool(report, Agent, tool, Field, label, client)
            await _cap_workflow(report, Agent, label, client)
        else:
            report.add(SECTION, f"{label} · streaming / @tool / workflow", SKIP,
                       "skipped — Responses endpoint unavailable on this gateway")
        await _close(getattr(client, "client", None))
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "OpenAIChatClient (Responses) · construct", e)

    # ── Client B: OpenAIChatCompletionClient → Chat Completions. ──
    # Chat + streaming work; @tool 500s on the gateway's OpenAI tool path.
    try:
        from agent_framework.openai import OpenAIChatCompletionClient  # type: ignore
        label = "OpenAIChatCompletionClient (Chat Completions)"
        client = OpenAIChatCompletionClient(
            base_url=config.BASE_URL, api_key=config.API_KEY, model=config.TOOL_MODEL,
        )
        await _connectivity(report, Agent, label, "POST /chat/completions", client,
                            expect_unsupported=False)
        await _cap_stream(report, Agent, label, client)
        await _cap_tool(report, Agent, tool, Field, label, client)
        await _cap_workflow(report, Agent, label, client)
        await _close(getattr(client, "client", None))
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "OpenAIChatCompletionClient · construct", e)

    # ── Isolate the OpenAI tool 500: raw openai SDK, no MAF (turn-1 vs turn-2). ──
    _openai_raw_tool_roundtrip(report)

    # ── Client C: AnthropicClient → /v1/messages. WORKS with empty beta flags. ──
    try:
        from agent_framework.anthropic import AnthropicClient  # type: ignore
        from anthropic import AsyncAnthropic  # type: ignore
        raw = AsyncAnthropic(
            api_key=config.API_KEY, base_url=config.ANTHROPIC_BASE_URL,
            timeout=60.0, max_retries=0,
        )

        # Working construction (matches production): clear MAF's default beta
        # flags + set max_tokens.
        label = "AnthropicClient (beta_flags=[])"
        client = AnthropicClient(
            model=config.TOOL_MODEL, anthropic_client=raw, additional_beta_flags=[],
        )
        await _connectivity(report, Agent, label, "POST /v1/messages", client,
                            expect_unsupported=False, agent_kwargs=ANTHROPIC_AGENT_KWARGS)
        await _cap_stream(report, Agent, label, client, agent_kwargs=ANTHROPIC_AGENT_KWARGS)
        await _cap_tool(report, Agent, tool, Field, label, client, agent_kwargs=ANTHROPIC_AGENT_KWARGS)
        await _cap_workflow(report, Agent, label, client, agent_kwargs=ANTHROPIC_AGENT_KWARGS)

        # Root-cause contrast: MAF's DEFAULT beta flags (mcp-client /
        # code-execution) 500 the gateway as soon as tools are attached. Same
        # client, only the flags differ — this row proves the beta header is
        # the culprit, not the gateway's tool support per se.
        client_default = AnthropicClient(model=config.TOOL_MODEL, anthropic_client=raw)
        await _cap_tool(report, Agent, tool, Field,
                        "AnthropicClient (DEFAULT beta flags — expect 500)", client_default,
                        agent_kwargs=ANTHROPIC_AGENT_KWARGS)

        # anthropic SDK uses close() (async), not aclose() like httpx
        await _close(raw)
    except Exception as e:  # noqa: BLE001
        report.capture_exception(
            SECTION, "AnthropicClient · construct", e,
            note="is `agent-framework-anthropic` installed with --pre?",
        )


def run(report: Report) -> None:
    report.section(
        SECTION,
        "13 · MAF ↔ gateway compatibility matrix",
        "Three MAF client constructions × [connectivity / streaming / @tool / workflow]. "
        "Recommended path = AnthropicClient with additional_beta_flags=[] (the default beta "
        "headers 500 the gateway when tools are present). OpenAIChatClient (Responses) is "
        "unusable; OpenAIChatCompletionClient chat/stream work but @tool 500s on the gateway's "
        "OpenAI tool path (a raw-openai-SDK round-trip below isolates turn-1 vs turn-2).",
    )
    if config.SKIP_MAF:
        report.add(SECTION, "MAF section", SKIP, "SKIP_MAF=1 set in environment")
        return
    try:
        asyncio.run(_run_async(report))
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "MAF compatibility section (top-level)", e)


if __name__ == "__main__":
    r = Report()
    run(r)
    from pathlib import Path
    r.write_markdown(Path("results/test_13_only.md"))
    print("wrote results/test_13_only.md")
