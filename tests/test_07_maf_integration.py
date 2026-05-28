"""Microsoft Agent Framework 1.0 driving the ZF gateway.

Critical: we use `OpenAIChatCompletionClient` (talks to /chat/completions), NOT
`OpenAIChatClient` (talks to /responses — would 404 here).

If `agent-framework` is not installed or the API shape has shifted, every check
in this section is captured as FAIL rather than crashing the harness.
"""
from __future__ import annotations

import asyncio
import importlib
from typing import Annotated

from . import config
from .reporter import FAIL, PASS, Report, SKIP, WARN, short

SECTION = "07_maf_integration"


def _import_client():
    """Try to import OpenAIChatCompletionClient from the right place.

    The class moved a few times during agent-framework's pre-release; we try the
    documented location first and fall back to a couple of alternatives.
    """
    candidates = [
        ("agent_framework.openai", "OpenAIChatCompletionClient"),
        ("agent_framework", "OpenAIChatCompletionClient"),
    ]
    last_err = None
    for module_name, cls_name in candidates:
        try:
            mod = importlib.import_module(module_name)
            return getattr(mod, cls_name)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise ImportError(
        f"could not import OpenAIChatCompletionClient from any of: {candidates}; last error: {last_err!r}"
    )


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


def _import_agent_class():
    """Canonical `agent_framework.Agent` class — matches official
    `02_add_tools.py` sample (`Agent(client=client, ...)`).
    """
    try:
        from agent_framework import Agent  # type: ignore
        return Agent
    except Exception:  # noqa: BLE001
        return None


def _new_agent(client, **kwargs):
    """Build via canonical `Agent(client=client, ...)` if available;
    fall back to `client.as_agent(...)` (equivalent convenience form).
    """
    AgentCls = _import_agent_class()
    if AgentCls is not None:
        return AgentCls(client=client, **kwargs)
    return client.as_agent(**kwargs)


def _import_workflow_builder():
    """Find a usable workflow builder class. Tries WorkflowBuilder first
    (more general API), then falls back to the higher-level pattern builders
    in case the pre-release version still ships them.

    Returns (cls, kind) where kind is 'workflow' | 'sequential' | None.
    """
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
    try:
        ClientCls = _import_client()
    except Exception as e:  # noqa: BLE001
        report.add(SECTION, "import OpenAIChatCompletionClient", FAIL, short(str(e), 200))
        return
    report.add(SECTION, "import OpenAIChatCompletionClient", PASS, ClientCls.__module__)

    # IMPORTANT: build ONE OpenAIChatCompletionClient for the whole section
    # and best-effort close its inner AsyncOpenAI's httpx pool at the end —
    # otherwise GC closes it after asyncio.run() returns and we get spurious
    # "RuntimeError: Event loop is closed" task-exception noise on Windows
    # (ProactorEventLoop + httpx). MAF skill convention #3 covers this.
    client = ClientCls(
        base_url=config.BASE_URL,
        api_key=config.API_KEY,
        model=config.MODEL,
    )

    # 1. Single-turn agent.run
    try:
        agent = _new_agent(client, name="ZFAgent", instructions="You are concise.")
        result = await agent.run("Reply with the word: pong")
        text = getattr(result, "text", None) or str(result)
        ok = "pong" in text.lower()
        report.add(SECTION, "agent.run (single-turn)", PASS if ok else WARN, short(text, 120))
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "agent.run (single-turn)", e)

    # 2. Streaming via agent.run(..., stream=True)
    try:
        agent = _new_agent(client, name="ZFAgent", instructions="You are concise.")
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

    # 3. Function tool — canonical MAF pattern from
    #    python/samples/02-agents/providers/openai/chat_completion_client_with_function_tools.py
    #    The 500 we saw in round 2 likely originated from the gateway's
    #    `tool_result` continuation path (the same 500 we hit in test_05). MAF
    #    omits `content` on the assistant turn (spec-correct), so if test_05
    #    variant A works against the gateway, this should too.
    tool, Field = _import_tool()
    if tool is None:
        report.add(SECTION, "@tool function call", SKIP, "`tool` decorator not importable")
    else:
        # 3a — Canonical pattern: @tool + Annotated[T, Field(description=…)]
        try:
            if Field is not None:
                @tool
                def square(
                    n: Annotated[int, Field(description="The integer to square.")],
                ) -> int:
                    """Return n squared."""
                    return n * n
            else:
                @tool
                def square(  # type: ignore[no-redef]
                    n: Annotated[int, "The integer to square."],
                ) -> int:
                    """Return n squared."""
                    return n * n

            agent = _new_agent(
                client,
                name="MathAgent",
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

        # 3b — Plain `def`-with-annotations (no @tool decorator): MAF also
        # accepts this and infers the schema. Same gateway path, different
        # client-side construction — useful for telling MAF wrapping issues
        # apart from gateway issues.
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
                name="MathAgent2",
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

    # 4. 2-agent workflow — prefer WorkflowBuilder, fall back to SequentialBuilder
    BuilderCls, kind, fqn = _import_workflow_builder()
    if BuilderCls is None:
        report.add(SECTION, "workflow (2-agent)", SKIP,
                   "neither WorkflowBuilder nor SequentialBuilder importable")
    else:
        report.add(SECTION, "workflow builder import", PASS, f"using {fqn} ({kind})")
        try:
            writer = _new_agent(
                client,
                name="Writer", instructions="Write one short sentence about the topic.",
            )
            shortener = _new_agent(
                client,
                name="Shortener",
                instructions="Rewrite the sentence in 3 words or fewer.",
            )

            # Build the graph depending on which class we got
            workflow = None
            build_err: Exception | None = None
            if kind == "workflow":
                # Canonical sample shape (07_first_graph_workflow.py) goes first;
                # older fluent forms are fallbacks for older releases.
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
                except Exception as inner:  # noqa: BLE001
                    try:
                        workflow = BuilderCls().participants([writer, shortener]).build()
                    except Exception as inner2:  # noqa: BLE001
                        build_err = inner2

            if workflow is None:
                report.add(
                    SECTION,
                    "workflow build",
                    FAIL,
                    f"all builder API shapes failed | last={short(repr(build_err), 200)}",
                )
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
                    else:
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

    # Best-effort cleanup of the inner AsyncOpenAI's httpx pool. Same reason
    # as test_12: without explicit close, GC fires after asyncio.run() returns
    # and produces spurious "RuntimeError: Event loop is closed" task-exception
    # noise on Windows (ProactorEventLoop + httpx).
    inner = getattr(client, "client", None)
    close_fn = getattr(inner, "close", None) if inner is not None else None
    if close_fn is not None:
        try:
            await close_fn()
        except Exception:  # noqa: BLE001 — best-effort
            pass


def run(report: Report) -> None:
    report.section(
        SECTION,
        "07 · Microsoft Agent Framework integration",
        "Drives the gateway from `agent-framework` 1.0 via the OpenAI-compatible chat-completions client. "
        "If `agent-framework` is unavailable, the section is skipped.",
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
    r.write_markdown(Path("results/test_07_only.md"))
    print("wrote results/test_07_only.md")
