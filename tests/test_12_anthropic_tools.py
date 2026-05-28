"""Microsoft Agent Framework + AnthropicClient driving the ZF gateway.

This is the **direct mirror** of `test_07_maf_integration` but using
`agent-framework-anthropic`'s `AnthropicClient` instead of
`agent_framework.openai.OpenAIChatCompletionClient`. Same MAF code paths
(`agent.run`, `@tool`, plain-def tool, workflow), same gateway, different
protocol underneath.

Round-3 finding to confirm:
  - test_07 (MAF + OpenAI client) → @tool and plain-def-tool both 500'd on
    the gateway's tool_result continuation path.
  - test_05 (raw OpenAI SDK + tools) → same 500.
  - test_06 (raw Anthropic SDK, turn 1 only) → 200, but continuation untested.

If THIS section's @tool checks pass, the answer for production MAF code on
this gateway is concrete: **swap to AnthropicClient**.

Wiring:
  AnthropicClient(model=..., anthropic_client=AsyncAnthropic(
      api_key=ZF_API_KEY,
      base_url=ZF gateway URL without /v1 (the Anthropic SDK appends /v1 itself),
  ))

Falls back to SKIP / FAIL with a clear reason if `agent-framework-anthropic`
or the `anthropic` SDK isn't installed.
"""
from __future__ import annotations

import asyncio
import importlib
import json
from typing import Annotated

from . import config
from .reporter import FAIL, PASS, Report, SKIP, WARN, short

SECTION = "12_maf_anthropic"


def _import_anthropic_client():
    """`agent_framework.anthropic.AnthropicClient` (lazy-shim into
    `agent_framework_anthropic`) — try both import paths.
    """
    candidates = [
        ("agent_framework.anthropic", "AnthropicClient"),
        ("agent_framework_anthropic", "AnthropicClient"),
    ]
    last_err: Exception | None = None
    for module_name, cls_name in candidates:
        try:
            mod = importlib.import_module(module_name)
            return getattr(mod, cls_name), f"{module_name}.{cls_name}"
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise ImportError(
        f"could not import AnthropicClient from any of: {candidates}; last={last_err!r}"
    )


def _import_agent_class():
    """The canonical `agent_framework.Agent` class (preferred over
    `client.as_agent(...)`) — matches the official `02_add_tools.py` sample.
    """
    try:
        from agent_framework import Agent  # type: ignore
        return Agent
    except Exception:  # noqa: BLE001
        return None


def _new_agent(client, **kwargs):
    """Build an Agent using the canonical `Agent(client=client, ...)` form
    if the class is importable; fall back to `client.as_agent(...)` (older /
    convenience API) otherwise. Both forms are semantically equivalent per
    the MAF skill.
    """
    AgentCls = _import_agent_class()
    if AgentCls is not None:
        return AgentCls(client=client, **kwargs)
    return client.as_agent(**kwargs)


def _import_async_anthropic():
    """Raw async Anthropic client — needed to point at a custom base_url."""
    from anthropic import AsyncAnthropic  # type: ignore
    return AsyncAnthropic


def _import_tool():
    """`tool` decorator + Field — used to annotate function tools."""
    try:
        from agent_framework import tool  # type: ignore
    except Exception:  # noqa: BLE001
        tool = None
    try:
        from pydantic import Field  # type: ignore
    except Exception:  # noqa: BLE001
        Field = None  # type: ignore
    return tool, Field


def _import_workflow_builder():
    """Mirror of test_07 — accept WorkflowBuilder or SequentialBuilder."""
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


async def _run_async(report: Report) -> None:
    # ── Imports & client construction ───────────────────────────────────────
    try:
        ClientCls, fqn = _import_anthropic_client()
    except Exception as e:  # noqa: BLE001
        report.add(SECTION, "import AnthropicClient", FAIL, short(str(e), 240))
        return
    report.add(SECTION, "import AnthropicClient", PASS, fqn)

    try:
        AsyncAnthropic = _import_async_anthropic()
    except Exception as e:  # noqa: BLE001
        report.add(SECTION, "import anthropic.AsyncAnthropic", FAIL, short(str(e), 240))
        return

    # Pre-built AsyncAnthropic pointed at the ZF gateway. The Anthropic SDK
    # appends /v1/messages itself, so we use config.ANTHROPIC_BASE_URL (no /v1).
    # IMPORTANT: build ONE AsyncAnthropic for the whole section and close it
    # at the end. Previously we built a fresh one per check and never aclose'd
    # them — GC fired after asyncio.run() had torn down the loop and produced
    # "RuntimeError: Event loop is closed" noise on Windows ProactorEventLoop.
    # MAF skill convention #3 ("manage credentials/agents with async with")
    # covers this — we use explicit aclose() because we want the `client` to
    # remain visible to every check below without re-indenting the world.
    _raw_client = AsyncAnthropic(
        api_key=config.API_KEY,
        base_url=config.ANTHROPIC_BASE_URL,
        timeout=60.0,
        max_retries=0,
    )
    client = ClientCls(model=config.TOOL_MODEL, anthropic_client=_raw_client)

    # ── 1. Single-turn agent.run ────────────────────────────────────────────
    try:
        agent = _new_agent(client, name="ZFAgentA", instructions="You are concise.")
        result = await agent.run("Reply with the word: pong")
        text = getattr(result, "text", None) or str(result)
        ok = "pong" in text.lower()
        report.add(SECTION, "agent.run (single-turn)", PASS if ok else WARN, short(text, 120))
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "agent.run (single-turn)", e)

    # ── 2. Streaming via agent.run(..., stream=True) ────────────────────────
    try:
        agent = _new_agent(client, name="ZFAgentB", instructions="You are concise.")
        chunks: list[str] = []
        async for chunk in agent.run("Count from 1 to 3.", stream=True):
            t = getattr(chunk, "text", None)
            if t:
                chunks.append(t)
        joined = "".join(chunks)
        report.add(
            SECTION,
            "agent.run(stream=True)",
            PASS if joined.strip() else FAIL,
            f"chunks={len(chunks)} text={short(joined, 100)}",
        )
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "agent.run(stream=True)", e)

    # ── 3. @tool function calls — THE headline check ────────────────────────
    # Mirrors test_07's 3a/3b exactly, just under a different client. If these
    # PASS here while failing in test_07, the practical advice is:
    # use MAF + AnthropicClient (not OpenAIChatCompletionClient) on this gateway.
    tool, Field = _import_tool()
    if tool is None:
        report.add(SECTION, "@tool function call", SKIP, "`tool` decorator not importable")
    else:
        # 3a — Canonical: @tool + Annotated[T, Field(description=…)] + docstring.
        # approval_mode="never_require" so the test doesn't block on a prompt.
        try:
            if Field is not None:
                @tool(approval_mode="never_require")
                def square(
                    n: Annotated[int, Field(description="The integer to square.")],
                ) -> int:
                    """Return n squared."""
                    return n * n
            else:
                @tool(approval_mode="never_require")
                def square(  # type: ignore[no-redef]
                    n: Annotated[int, "The integer to square."],
                ) -> int:
                    """Return n squared."""
                    return n * n

            agent = _new_agent(
                client,
                name="MathAgentA",
                instructions="When asked to square a number, call the square tool. "
                             "After it returns, reply with just the number.",
                tools=[square],
            )
            result = await agent.run("Use your tool to square the number 7.")
            text = getattr(result, "text", None) or str(result)
            ok = "49" in text
            report.add(
                SECTION,
                "@tool function call (canonical, square(7)=49)",
                PASS if ok else WARN,
                short(text, 160),
            )
        except Exception as e:  # noqa: BLE001
            report.capture_exception(SECTION, "@tool function call (canonical)", e)

        # 3b — Plain def with Annotated, no @tool decorator. MAF infers the
        # schema from annotations + docstring. Same gateway path, different
        # client-side construction.
        try:
            if Field is not None:
                def add(
                    a: Annotated[int, Field(description="First operand.")],
                    b: Annotated[int, Field(description="Second operand.")],
                ) -> int:
                    """Add two integers."""
                    return a + b
            else:
                def add(  # type: ignore[no-redef]
                    a: Annotated[int, "First operand."],
                    b: Annotated[int, "Second operand."],
                ) -> int:
                    """Add two integers."""
                    return a + b

            agent = _new_agent(
                client,
                name="MathAgentB",
                instructions="When asked to add numbers, call the add tool.",
                tools=[add],
            )
            result = await agent.run("Use the tool to add 15 and 27.")
            text = getattr(result, "text", None) or str(result)
            ok = "42" in text
            report.add(
                SECTION,
                "plain def tool (add(15,27)=42)",
                PASS if ok else WARN,
                short(text, 160),
            )
        except Exception as e:  # noqa: BLE001
            report.capture_exception(SECTION, "plain def tool", e)

        # 3c — Multi-tool agent: two tools, model picks the right one.
        # Anthropic protocol uses tool_choice={"type":"auto"} natively; MAF
        # handles the encoding for us.
        try:
            if Field is not None:
                @tool(approval_mode="never_require")
                def get_weather(
                    location: Annotated[str, Field(description="City name.")],
                ) -> str:
                    """Get the current weather for a city."""
                    return f"The weather in {location} is sunny, 22°C."

                @tool(approval_mode="never_require")
                def get_time(
                    timezone: Annotated[str, Field(description="IANA timezone, e.g. Asia/Shanghai.")],
                ) -> str:
                    """Get the current local time for a timezone."""
                    return f"The current time in {timezone} is 14:30."
            else:
                @tool(approval_mode="never_require")
                def get_weather(  # type: ignore[no-redef]
                    location: Annotated[str, "City name."],
                ) -> str:
                    """Get the current weather for a city."""
                    return f"The weather in {location} is sunny, 22°C."

                @tool(approval_mode="never_require")
                def get_time(  # type: ignore[no-redef]
                    timezone: Annotated[str, "IANA timezone."],
                ) -> str:
                    """Get the current local time for a timezone."""
                    return f"The current time in {timezone} is 14:30."

            agent = _new_agent(
                client,
                name="ConciergeAgent",
                instructions="You have two tools: get_weather and get_time. "
                             "Pick the right one for the user's question and answer with the result.",
                tools=[get_weather, get_time],
            )
            result = await agent.run("What's the weather in Munich right now?")
            text = getattr(result, "text", None) or str(result)
            ok = ("22" in text) or ("sunny" in text.lower()) or ("munich" in text.lower())
            report.add(
                SECTION,
                "multi-tool agent (weather vs time, model picks weather)",
                PASS if ok else WARN,
                short(text, 160),
            )
        except Exception as e:  # noqa: BLE001
            report.capture_exception(SECTION, "multi-tool agent", e)

        # ── 3d — Diagnostic: AnthropicClient WITHOUT default beta headers.
        # MAF's AnthropicClient defaults to additional_beta_flags=[
        #   "mcp-client-2025-04-04", "code-execution-2025-08-25"
        # ] which become an "anthropic-beta:" request header. The ZF gateway
        # may reject those when tools are also present (single-turn agent.run
        # works → beta headers alone are tolerated; @tool 500s → it's the
        # beta+tools combination). Pass additional_beta_flags=[] to suppress.
        try:
            client_no_beta = ClientCls(
                model=config.TOOL_MODEL,
                anthropic_client=_raw_client,
                additional_beta_flags=[],
            )

            if Field is not None:
                @tool(approval_mode="never_require")
                def square_nb(
                    n: Annotated[int, Field(description="The integer to square.")],
                ) -> int:
                    """Return n squared."""
                    return n * n
            else:
                @tool(approval_mode="never_require")
                def square_nb(  # type: ignore[no-redef]
                    n: Annotated[int, "The integer to square."],
                ) -> int:
                    """Return n squared."""
                    return n * n

            agent_nb = _new_agent(
                client_no_beta,
                name="MathAgentNoBeta",
                instructions="When asked to square a number, call the square_nb tool. "
                             "After it returns, reply with just the number.",
                tools=[square_nb],
            )
            result = await agent_nb.run("Use your tool to square the number 7.")
            text = getattr(result, "text", None) or str(result)
            ok = "49" in text
            report.add(
                SECTION,
                "@tool with additional_beta_flags=[] (suppress MAF defaults)",
                PASS if ok else WARN,
                short(text, 160),
            )
        except Exception as e:  # noqa: BLE001
            report.capture_exception(
                SECTION,
                "@tool with additional_beta_flags=[] (suppress MAF defaults)",
                e,
            )

        # ── 3e — Diagnostic: simple @tool form matching `02_add_tools.py`
        # verbatim — plain `str` parameter, no Annotated, no Field, just
        # the docstring. If THIS passes while the Annotated+Field form 500s,
        # MAF is serializing Field metadata the gateway rejects. If it ALSO
        # 500s, the gateway treats Annotated+Field and plain str the same.
        try:
            @tool(approval_mode="never_require")
            def get_weather_simple(location: str) -> str:
                """Get the weather for a given location."""
                return f"The weather in {location} is sunny, 22°C."

            client_simple = ClientCls(
                model=config.TOOL_MODEL,
                anthropic_client=_raw_client,
                additional_beta_flags=[],   # combine both fixes
            )
            agent_simple = _new_agent(
                client_simple,
                name="SimpleWeatherAgent",
                instructions="When asked about weather, call get_weather_simple "
                             "and reply with one short sentence using its result.",
                tools=[get_weather_simple],
            )
            result = await agent_simple.run("What's the weather in Berlin?")
            text = getattr(result, "text", None) or str(result)
            ok = ("22" in text) or ("sunny" in text.lower()) or ("berlin" in text.lower())
            report.add(
                SECTION,
                "@tool simple form (plain str, matches 02_add_tools.py)",
                PASS if ok else WARN,
                short(text, 160),
            )
        except Exception as e:  # noqa: BLE001
            report.capture_exception(
                SECTION,
                "@tool simple form (plain str, matches 02_add_tools.py)",
                e,
            )

    # ── 3f — Raw AsyncAnthropic two-turn round-trip diagnostic ──────────────
    # Bypasses MAF entirely: hand-builds the {"role":"user", "content":[
    # {"type":"tool_result", ...}]} continuation. Pinpoints whether the 500
    # in checks 3a/3b/3c originates from MAF wrapping (turn-1 schema or
    # headers) or from the gateway's tool_result continuation path itself.
    # Reuses the same _raw_client that everything else uses.
    try:
        history = [{"role": "user", "content": "What's the weather in Munich?"}]
        tool_def = {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
        r1 = await _raw_client.messages.create(
            model=config.TOOL_MODEL,
            max_tokens=300,
            messages=history,
            tools=[tool_def],
        )
        tool_block = next(
            (b for b in r1.content if getattr(b, "type", None) == "tool_use"), None
        )
        if not tool_block:
            report.add(
                SECTION,
                "raw AsyncAnthropic · turn 1 (tool_use)",
                WARN,
                f"no tool_use block | stop_reason={r1.stop_reason}",
            )
        else:
            report.add(
                SECTION,
                "raw AsyncAnthropic · turn 1 (tool_use)",
                PASS,
                f"stop={r1.stop_reason} | name={tool_block.name} | "
                f"input={short(json.dumps(tool_block.input), 80)}",
            )
            # Turn 2 — feed tool_result back. THIS is the deciding check.
            history.append({
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": tool_block.id,
                    "name": tool_block.name,
                    "input": tool_block.input,
                }],
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
                r2 = await _raw_client.messages.create(
                    model=config.TOOL_MODEL,
                    max_tokens=300,
                    messages=history,
                    tools=[tool_def],
                )
                final = "".join(
                    b.text for b in r2.content if getattr(b, "type", None) == "text"
                )
                ok = ("22" in final) or ("sunny" in final.lower())
                report.add(
                    SECTION,
                    "raw AsyncAnthropic · turn 2 (tool_result continuation)",
                    PASS if ok else WARN,
                    f"stop={r2.stop_reason} | final={short(final, 120)}",
                )
            except Exception as e:  # noqa: BLE001
                report.capture_exception(
                    SECTION,
                    "raw AsyncAnthropic · turn 2 (tool_result continuation)",
                    e,
                )
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "raw AsyncAnthropic · turn 1 (tool_use)", e)

    # ── 4. Two-agent workflow (writer → shortener) ──────────────────────────
    BuilderCls, kind, fqn = _import_workflow_builder()
    if BuilderCls is None:
        report.add(SECTION, "workflow (2-agent)", SKIP,
                   "neither WorkflowBuilder nor SequentialBuilder importable")
    else:
        try:
            writer = _new_agent(
                client,
                name="Writer", instructions="Write one short sentence about the topic.",
            )
            shortener = _new_agent(
                client,
                name="Shortener", instructions="Rewrite the sentence in 3 words or fewer.",
            )

            workflow = None
            build_err: Exception | None = None
            if kind == "workflow":
                # Canonical sample shape (07_first_graph_workflow.py) goes first;
                # the older fluent forms are fallbacks for older releases.
                tries = [
                    lambda: (BuilderCls(start_executor=writer)
                             .add_edge(writer, shortener)
                             .build()),
                    lambda: (BuilderCls()
                             .set_start_executor(writer)
                             .add_edge(writer, shortener)
                             .build()),
                    lambda: (BuilderCls()
                             .add_edge(writer, shortener)
                             .set_start_executor(writer)
                             .build()),
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
                report.add(SECTION, "workflow build", FAIL,
                           f"all builder shapes failed | last={short(repr(build_err), 200)}")
            else:
                # Canonical: `events = await workflow.run(...)` returns a
                # WorkflowRunResult with `.get_outputs()` / `.get_final_state()`.
                # Fall back to run_stream / plain-text extract for older shapes.
                output_text = ""
                final_state = ""
                ran = False
                try:
                    events = await workflow.run("the moon")
                    if hasattr(events, "get_outputs"):
                        outs = events.get_outputs() or []
                        output_text = "; ".join(str(o) for o in outs)
                        if hasattr(events, "get_final_state"):
                            final_state = str(events.get_final_state())
                    else:  # very old API — just stringify
                        output_text = (getattr(events, "text", None)
                                        or getattr(events, "output", None)
                                        or str(events))
                    ran = True
                except Exception as run_err:  # noqa: BLE001
                    build_err = run_err
                if not ran and hasattr(workflow, "run_stream"):
                    try:
                        async for event in workflow.run_stream("the moon"):
                            data = (getattr(event, "data", None)
                                    or getattr(event, "output", None)
                                    or getattr(event, "text", None))
                            if data:
                                output_text = str(data)
                        ran = True
                    except Exception as run_err2:  # noqa: BLE001
                        build_err = run_err2
                detail = short(output_text, 140) + (f" | state={final_state}" if final_state else "")
                report.add(
                    SECTION,
                    f"workflow run (writer → shortener) [{kind}]",
                    PASS if output_text.strip() else WARN,
                    detail if ran else f"run failed: {short(repr(build_err), 200)}",
                )
        except Exception as e:  # noqa: BLE001
            report.capture_exception(SECTION, "workflow execution", e)

    # ── Cleanup: close the shared AsyncAnthropic before the event loop dies.
    # Without this, GC closes it after asyncio.run() returns → the connection
    # pool tries to schedule callbacks on a closed loop and we get spurious
    # "RuntimeError: Event loop is closed" task-exception noise on Windows.
    # NB: anthropic SDK names this `close()` (async), not `aclose()` like httpx.
    try:
        await _raw_client.close()
    except Exception as e:  # noqa: BLE001 — best-effort cleanup
        report.add(SECTION, "AsyncAnthropic.close (cleanup)", WARN, short(repr(e), 200))


def run(report: Report) -> None:
    report.section(
        SECTION,
        "12 · MAF + AnthropicClient (mirror of section 07)",
        "Drives the gateway from `agent-framework` 1.0 via the Anthropic-protocol client. "
        "Same MAF code paths as section 07 (single-turn / streaming / @tool / plain def / "
        "workflow), but `AnthropicClient(anthropic_client=AsyncAnthropic(base_url=...))` "
        "instead of `OpenAIChatCompletionClient`. Direct apples-to-apples for the "
        "tool_result continuation 500.",
    )
    if config.SKIP_MAF:
        report.add(SECTION, "MAF section", SKIP, "SKIP_MAF=1 set in environment")
        return
    try:
        asyncio.run(_run_async(report))
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "MAF section (top-level)", e)


if __name__ == "__main__":
    r = Report()
    run(r)
    from pathlib import Path
    r.write_markdown(Path("results/test_12_only.md"))
    print("wrote results/test_12_only.md")
