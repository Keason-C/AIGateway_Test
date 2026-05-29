"""MAF ↔ ZF-gateway compatibility matrix — the headline section.

Goal: answer one question cleanly — *does the Microsoft Agent Framework work
against our gateway, and which client do you pick?*

We probe THREE MAF client constructions side-by-side, each across the same
capability matrix (connectivity / streaming / @tool / 2-agent workflow):

  ┌───────────────────────────┬──────────────────────┬──────────────┐
  │ MAF client                │ HTTP path it hits    │ expectation  │
  ├───────────────────────────┼──────────────────────┼──────────────┤
  │ OpenAIChatClient          │ POST /responses      │ 404 (no such │
  │   (Responses API)         │                      │ endpoint)    │
  │ OpenAIChatCompletionClient│ POST /chat/completions│ ✅ works     │
  │   (Chat Completions)      │                      │              │
  │ AnthropicClient           │ POST /v1/messages    │ ✅ works     │
  └───────────────────────────┴──────────────────────┴──────────────┘

The Responses-vs-ChatCompletions split is the #1 MAF footgun (see the skill):
`OpenAIChatClient` despite its name calls the **Responses API** and 404s on any
OpenAI-compatible proxy — you must use `OpenAIChatCompletionClient`. This test
makes that concrete on the ZF gateway.

Tool calling is expected to PASS — we write the canonical MAF shapes from the
audited skill, so a 500 here would point at MAF wiring/version, not the gateway.

Construction matches the official providers reference:
  • OpenAI clients: base_url=ZF .../v1, api_key, model=assistant-name.
  • AnthropicClient(model=..., anthropic_client=AsyncAnthropic(base_url=...))
    — the Anthropic SDK appends /v1 itself, so we pass the base without /v1.
"""
from __future__ import annotations

import asyncio
import importlib
from typing import Annotated

from . import config
from .reporter import FAIL, INFO, PASS, Report, SKIP, WARN, short

SECTION = "13_maf_compatibility"


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
async def _connectivity(report, Agent, label, protocol, client, *, expect_unsupported: bool) -> bool:
    """Single-turn agent.run as the connectivity probe. Returns True if the
    client reached the gateway without raising."""
    name = f"{label} · connectivity (single-turn)"
    try:
        agent = Agent(client=client, name="ZFProbe", instructions="You are concise.")
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
        if expect_unsupported and _is_404(e):
            report.add(SECTION, name, INFO,
                       f"404 on {protocol} as expected — gateway has no Responses endpoint; "
                       f"use OpenAIChatCompletionClient instead. ({short(str(e), 140)})")
        else:
            report.capture_exception(SECTION, name, e)
        return False


async def _cap_stream(report, Agent, label, client) -> None:
    try:
        agent = Agent(client=client, name="ZFStream", instructions="You are concise.")
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


async def _cap_tool(report, Agent, tool, Field, label, client) -> None:
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
        )
        result = await agent.run("Use your tool to square the number 7.")
        text = getattr(result, "text", None) or str(result)
        ok = "49" in text
        report.add(SECTION, f"{label} · @tool call (square(7)=49)",
                   PASS if ok else WARN, short(text, 160))
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, f"{label} · @tool call (square(7)=49)", e)


async def _cap_workflow(report, Agent, label, client) -> None:
    BuilderCls, kind, _fqn = _imp_workflow_builder()
    if BuilderCls is None:
        report.add(SECTION, f"{label} · workflow (2-agent)", SKIP,
                   "neither WorkflowBuilder nor SequentialBuilder importable")
        return
    try:
        writer = Agent(client=client, name="Writer",
                       instructions="Write one short sentence about the topic.")
        shortener = Agent(client=client, name="Shortener",
                          instructions="Rewrite the sentence in 3 words or fewer.")

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

    # ── Client A: OpenAIChatClient → Responses API (/responses). Expected 404. ──
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

    # ── Client B: OpenAIChatCompletionClient → Chat Completions. Expected ✅. ──
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

    # ── Client C: AnthropicClient → /v1/messages. Expected ✅. ──
    try:
        from agent_framework.anthropic import AnthropicClient  # type: ignore
        from anthropic import AsyncAnthropic  # type: ignore
        label = "AnthropicClient"
        raw = AsyncAnthropic(
            api_key=config.API_KEY, base_url=config.ANTHROPIC_BASE_URL,
            timeout=60.0, max_retries=0,
        )
        client = AnthropicClient(model=config.TOOL_MODEL, anthropic_client=raw)
        await _connectivity(report, Agent, label, "POST /v1/messages", client,
                            expect_unsupported=False)
        await _cap_stream(report, Agent, label, client)
        await _cap_tool(report, Agent, tool, Field, label, client)
        await _cap_workflow(report, Agent, label, client)
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
        "Answers: does MAF work on the ZF gateway, and which client to pick — "
        "OpenAIChatClient (Responses /responses, expected 404) vs "
        "OpenAIChatCompletionClient (Chat Completions, expected ✅) vs "
        "AnthropicClient (/v1/messages, expected ✅).",
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
