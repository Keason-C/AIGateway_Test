"""Concurrency / throughput probe.

Each level fires `requests_per_level` identical chat completions concurrently
with httpx.AsyncClient. We capture:
  - success rate
  - p50 / p95 / max latency (seconds)
  - first HTTP status / error string that appears

We stop early once a level exceeds `failure_threshold` (default 50 %) — that
becomes the practical concurrency ceiling.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass

import httpx

from . import config
from .reporter import FAIL, INFO, PASS, Report, SKIP, WARN, short

SECTION = "08_concurrency"

REQUESTS_PER_LEVEL = 20
FAILURE_THRESHOLD = 0.5  # > this fraction failing → consider the level overloaded
REQUEST_TIMEOUT = 60.0
COOLDOWN_SECONDS = 5.0


@dataclass
class RequestResult:
    ok: bool
    status: int | None
    latency: float
    error: str = ""


async def _one_request(client: httpx.AsyncClient, body: dict, headers: dict) -> RequestResult:
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            config.BASE_URL + "/chat/completions",
            json=body,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        latency = time.perf_counter() - t0
        ok = resp.status_code == 200
        return RequestResult(ok=ok, status=resp.status_code, latency=latency,
                             error="" if ok else short(resp.text, 200))
    except Exception as e:  # noqa: BLE001
        return RequestResult(ok=False, status=None, latency=time.perf_counter() - t0,
                             error=short(f"{type(e).__name__}: {e}", 200))


async def _run_level(concurrency: int, total: int) -> list[RequestResult]:
    headers = {
        "Authorization": f"Bearer {config.API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": config.MODEL,
        "messages": [{"role": "user", "content": "Reply with the word: ok"}],
    }
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(
        max_connections=max(concurrency * 2, 10),
        max_keepalive_connections=concurrency,
    )

    async with httpx.AsyncClient(limits=limits, timeout=REQUEST_TIMEOUT) as client:
        async def _bounded():
            async with sem:
                return await _one_request(client, body, headers)
        return await asyncio.gather(*[_bounded() for _ in range(total)])


def _stats(latencies: list[float]) -> tuple[float, float, float]:
    if not latencies:
        return 0.0, 0.0, 0.0
    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted)
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)] if len(latencies_sorted) > 1 else latencies_sorted[0]
    return p50, p95, max(latencies_sorted)


def _ceiling_from(results: dict[int, list[RequestResult]]) -> int | None:
    """Highest concurrency level where success-rate stayed above the threshold."""
    ceiling = None
    for level, rs in sorted(results.items()):
        rate = sum(1 for r in rs if r.ok) / len(rs) if rs else 0.0
        if rate >= 1 - FAILURE_THRESHOLD:
            ceiling = level
    return ceiling


def run(report: Report) -> None:
    report.section(
        SECTION,
        "08 · Concurrency",
        f"Ramps concurrency through {config.CONCURRENCY_LEVELS}, "
        f"{REQUESTS_PER_LEVEL} requests/level, "
        f"failure threshold {int(FAILURE_THRESHOLD*100)}%.",
    )
    if config.SKIP_CONCURRENCY:
        report.add(SECTION, "concurrency probe", SKIP, "SKIP_CONCURRENCY=1 set")
        return

    results: dict[int, list[RequestResult]] = {}

    async def _all():
        for level in config.CONCURRENCY_LEVELS:
            rs = await _run_level(level, REQUESTS_PER_LEVEL)
            results[level] = rs
            ok = sum(1 for r in rs if r.ok)
            rate = ok / len(rs) if rs else 0.0
            p50, p95, mx = _stats([r.latency for r in rs if r.ok])
            status = PASS if rate >= 1 - FAILURE_THRESHOLD else WARN
            first_error = next((r.error for r in rs if not r.ok), "")
            statuses = sorted({r.status for r in rs if r.status is not None})
            report.add(
                SECTION,
                f"concurrency={level} (n={len(rs)})",
                status,
                f"success={ok}/{len(rs)} ({rate:.0%}) | "
                f"p50={p50:.2f}s p95={p95:.2f}s max={mx:.2f}s | "
                f"http_statuses={statuses}"
                + (f" | first_error={first_error}" if first_error else ""),
            )
            # Stop early if this level mostly failed — saves quota
            if rate < 1 - FAILURE_THRESHOLD:
                break
            await asyncio.sleep(COOLDOWN_SECONDS)

    try:
        asyncio.run(_all())
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "concurrency probe (top-level)", e)
        return

    # ASCII summary
    lines = ["", "**Latency summary (successful requests only):**", "",
             "| Concurrency | Success | p50 (s) | p95 (s) | Max (s) | Throughput (req/s) |",
             "|---:|---:|---:|---:|---:|---:|"]
    for level, rs in sorted(results.items()):
        ok = sum(1 for r in rs if r.ok)
        p50, p95, mx = _stats([r.latency for r in rs if r.ok])
        # rough throughput = ok / max_latency-of-success
        thr = ok / mx if mx > 0 else 0.0
        lines.append(
            f"| {level} | {ok}/{len(rs)} | {p50:.2f} | {p95:.2f} | {mx:.2f} | {thr:.2f} |"
        )

    ceiling = _ceiling_from(results)
    if ceiling is None:
        lines += ["", "Could not determine a stable concurrency ceiling — every level failed."]
    else:
        lines += ["", f"**Practical concurrency ceiling for this assistant: ≈ {ceiling} parallel requests** "
                       f"(highest level still under {int(FAILURE_THRESHOLD*100)}% failures)."]
    report.block(SECTION, "\n".join(lines))


if __name__ == "__main__":
    r = Report()
    run(r)
    from pathlib import Path
    r.write_markdown(Path("results/test_08_only.md"))
    print("wrote results/test_08_only.md")
