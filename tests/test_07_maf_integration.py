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


def _import_sequential():
    try:
        from agent_framework import SequentialBuilder  # type: ignore
        return SequentialBuilder
    except Exception:  # noqa: BLE001
        try:
            from agent_framework.workflows import SequentialBuilder  # type: ignore
            return SequentialBuilder
        except Exception:
            return None


async def _run_async(report: Report) -> None:
    try:
        ClientCls = _import_client()
    except Exception as e:  # noqa: BLE001
        report.add(SECTION, "import OpenAIChatCompletionClient", FAIL, short(str(e), 200))
        return
    report.add(SECTION, "import OpenAIChatCompletionClient", PASS, ClientCls.__module__)

    def _new_client():
        return ClientCls(
            base_url=config.BASE_URL,
            api_key=config.API_KEY,
            model=config.MODEL,
        )

    # 1. Single-turn agent.run
    try:
        client = _new_client()
        agent = client.as_agent(name="ZFAgent", instructions="You are concise.")
        result = await agent.run("Reply with the word: pong")
        text = getattr(result, "text", None) or str(result)
        ok = "pong" in text.lower()
        report.add(SECTION, "agent.run (single-turn)", PASS if ok else WARN, short(text, 120))
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "agent.run (single-turn)", e)

    # 2. Streaming via agent.run(..., stream=True)
    try:
        client = _new_client()
        agent = client.as_agent(name="ZFAgent", instructions="You are concise.")
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

    # 3. Function tool — Python function with Annotated args
    tool, Field = _import_tool()
    if tool is None:
        report.add(SECTION, "@tool function call", SKIP, "`tool` decorator not importable")
    else:
        try:
            @tool
            def square(
                n: Annotated[int, (Field(description="The integer to square") if Field else "n")],
            ) -> int:
                """Return n squared."""
                return n * n

            client = _new_client()
            agent = client.as_agent(
                name="MathAgent",
                instructions="Use the square tool when asked for a square.",
                tools=[square],
            )
            result = await agent.run("Use your tool to square the number 7.")
            text = getattr(result, "text", None) or str(result)
            ok = "49" in text
            report.add(
                SECTION,
                "@tool function call (square(7)=49)",
                PASS if ok else WARN,
                short(text, 140),
            )
        except Exception as e:  # noqa: BLE001
            report.capture_exception(SECTION, "@tool function call", e)

    # 4. Sequential 2-agent workflow
    Sequential = _import_sequential()
    if Sequential is None:
        report.add(SECTION, "SequentialBuilder workflow", SKIP, "SequentialBuilder not importable")
    else:
        try:
            client = _new_client()
            writer = client.as_agent(
                name="Writer", instructions="Write one short sentence about the topic.",
            )
            shortener = client.as_agent(
                name="Shortener",
                instructions="Rewrite the sentence in 3 words or fewer.",
            )
            workflow = Sequential().add_agents([writer, shortener]).build()
            output_text = ""
            try:
                async for event in workflow.run_stream("the moon"):
                    out = getattr(event, "data", None) or getattr(event, "output", None)
                    if out:
                        output_text = str(out)
            except AttributeError:
                # API variant where you call workflow.run(...) directly
                result = await workflow.run("the moon")
                output_text = getattr(result, "text", None) or str(result)
            report.add(
                SECTION,
                "SequentialBuilder workflow (writer → shortener)",
                PASS if output_text.strip() else WARN,
                short(output_text, 140),
            )
        except Exception as e:  # noqa: BLE001
            report.capture_exception(SECTION, "SequentialBuilder workflow", e)


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
