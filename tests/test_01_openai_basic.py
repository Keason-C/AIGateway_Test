"""Basic OpenAI-SDK connectivity to the ZF gateway.

Covers:
 - GET /v1/models
 - Single-turn chat.completions.create (no max_tokens — see pitfall)
 - Multi-turn (client-side history because gateway has no memory)
 - Wrong API key → 401
 - Wrong model name → 404
 - Confirm whether the documented `max_tokens` → 500 still holds
"""
from __future__ import annotations

import httpx
from openai import OpenAI

from . import config
from .reporter import FAIL, INFO, PASS, Report, WARN, short

SECTION = "01_openai_basic"


def _client(api_key: str | None = None) -> OpenAI:
    return OpenAI(
        base_url=config.BASE_URL,
        api_key=api_key or config.API_KEY,
        timeout=60.0,
        max_retries=0,
    )


def run(report: Report) -> None:
    report.section(
        SECTION,
        "01 · OpenAI SDK — basic connectivity",
        "Smoke-tests the OpenAI Python SDK against the gateway and verifies a few known error contracts.",
    )

    # 1. /v1/models
    try:
        models = _client().models.list()
        names = [m.id for m in models.data]
        if config.MODEL in names:
            report.add(SECTION, "GET /v1/models lists configured model", PASS, short(names))
        elif names:
            report.add(
                SECTION,
                "GET /v1/models lists configured model",
                WARN,
                f"got {len(names)} models but `{config.MODEL}` not in list: {short(names)}",
            )
        else:
            report.add(SECTION, "GET /v1/models lists configured model", FAIL, "empty list")
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "GET /v1/models", e)

    # 2. Single-turn chat (no max_tokens — gateway 500s if you send it)
    try:
        r = _client().chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": "Reply with the word: pong"}],
        )
        content = r.choices[0].message.content or ""
        if "pong" in content.lower():
            report.add(SECTION, "Single-turn chat (no max_tokens)", PASS, short(content))
        else:
            report.add(SECTION, "Single-turn chat (no max_tokens)", WARN,
                       f"unexpected content: {short(content)}")
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "Single-turn chat (no max_tokens)", e)

    # 3. Multi-turn (client-side memory)
    try:
        c = _client()
        history: list[dict] = []

        history.append({"role": "user", "content": "My favorite fruit is starfruit. Confirm in 3 words."})
        r1 = c.chat.completions.create(model=config.MODEL, messages=history)
        a1 = r1.choices[0].message.content or ""
        history.append({"role": "assistant", "content": a1})

        history.append({"role": "user", "content": "What is my favorite fruit? Answer with one word."})
        r2 = c.chat.completions.create(model=config.MODEL, messages=history)
        a2 = (r2.choices[0].message.content or "").lower()

        if "starfruit" in a2:
            report.add(SECTION, "Multi-turn (client-side history)", PASS, short(a2))
        else:
            report.add(SECTION, "Multi-turn (client-side history)", WARN,
                       f"did not recall context — got: {short(a2)}")
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "Multi-turn (client-side history)", e)

    # 4. Wrong API key → 401
    try:
        _client(api_key="sk-definitely-not-real-zf-key").chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": "hi"}],
        )
        report.add(SECTION, "Wrong API key → 401", FAIL, "request unexpectedly succeeded")
    except Exception as e:  # noqa: BLE001
        status = getattr(e, "status_code", None) or getattr(e, "status", None)
        msg = short(str(e))
        if status == 401 or "401" in msg or "unauthorized" in msg.lower() or "invalid_api_key" in msg.lower():
            report.add(SECTION, "Wrong API key → 401", PASS, msg)
        else:
            report.add(SECTION, "Wrong API key → 401", WARN, f"non-401 error: {msg}")

    # 5. Wrong model name → 404
    try:
        _client().chat.completions.create(
            model="definitely-not-a-real-assistant-9999",
            messages=[{"role": "user", "content": "hi"}],
        )
        report.add(SECTION, "Wrong model → 404", FAIL, "request unexpectedly succeeded")
    except Exception as e:  # noqa: BLE001
        status = getattr(e, "status_code", None) or getattr(e, "status", None)
        msg = short(str(e))
        if status == 404 or "404" in msg or "model_not_found" in msg.lower():
            report.add(SECTION, "Wrong model → 404", PASS, msg)
        else:
            report.add(SECTION, "Wrong model → 404", WARN, f"non-404 error: {msg}")

    # 6. Confirm the documented pitfall: max_tokens raises 500 on /v1/chat/completions
    try:
        _client().chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
        )
        report.add(SECTION, "max_tokens pitfall (expect 500/error)", INFO,
                   "request succeeded — gateway may have fixed the pitfall")
    except Exception as e:  # noqa: BLE001
        status = getattr(e, "status_code", None) or getattr(e, "status", None)
        msg = short(str(e))
        if status == 500 or "500" in msg:
            report.add(SECTION, "max_tokens pitfall (expect 500/error)", PASS,
                       f"500 confirmed: {msg}")
        else:
            report.add(SECTION, "max_tokens pitfall (expect 500/error)", WARN,
                       f"errored with non-500: {msg}")

    # 7. Raw HTTP GET /v1/models with httpx (to capture status + headers)
    try:
        with httpx.Client(timeout=30.0) as h:
            resp = h.get(
                config.BASE_URL + "/models",
                headers={"Authorization": f"Bearer {config.API_KEY}"},
            )
            ct = resp.headers.get("content-type", "")
            report.add(
                SECTION,
                "Raw httpx GET /models",
                PASS if resp.status_code == 200 else WARN,
                f"status={resp.status_code} content-type={ct}",
            )
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "Raw httpx GET /models", e)


if __name__ == "__main__":
    from .reporter import Report
    r = Report()
    run(r)
    from pathlib import Path
    r.write_markdown(Path("results/test_01_only.md"))
    print("wrote results/test_01_only.md")
