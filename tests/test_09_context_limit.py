"""Binary-search the gateway's effective input-token ceiling.

Round-3 rewrite — addresses two failures from round 2:

  • token counting was off: asking for 8 000 tokens actually produced ~2 668.
    The new builder iteratively appends words and rechecks `len(enc.encode(...))`
    until it lands within ±2% of the target.
  • baseline 8 000 → 500 (NOT 429) means test_08's burst left junk state.
    We now start the search at 2 000 tokens, do 90 s cooldown after concurrency,
    and retry once on 500/503 with a longer back-off.

This module is also runnable standalone — `python -m tests.test_09_context_limit`
— which is the recommended way to get a clean reading without test_08
contaminating rate-limit / connection state.
"""
from __future__ import annotations

import os
import time

import httpx
import tiktoken

from . import config
from .reporter import FAIL, INFO, PASS, Report, SKIP, WARN, short

SECTION = "09_context_limit"

TOLERANCE_TOKENS = 4_000          # binary-search precision
RESPONSE_TIMEOUT = 240.0          # large prompts take a while
START_LOWER = 2_000               # was 8 000 — lower so we always have a baseline
RETRY_SLEEP = 25.0                # back-off when we get 500/503/429 on a probe
MAX_PROBE_RETRIES = 2


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _build_text_of_n_tokens(target_tokens: int, enc) -> tuple[str, int]:
    """Iteratively build a passage whose cl100k_base token count is within ±2%
    of `target_tokens`. Returns (text, actual_token_count)."""
    chunk = "lorem ipsum dolor sit amet consectetur adipiscing elit "
    chunk_tokens = len(enc.encode(chunk))
    # First-pass estimate
    repeats = max(1, target_tokens // chunk_tokens)
    text = chunk * repeats
    actual = len(enc.encode(text))
    # Iteratively close the gap (max ~6 corrections is plenty)
    for _ in range(6):
        if abs(actual - target_tokens) <= max(50, target_tokens // 50):
            break
        delta_tokens = target_tokens - actual
        delta_repeats = max(1, abs(delta_tokens) // chunk_tokens)
        if delta_tokens > 0:
            text += chunk * delta_repeats
        else:
            # trim from the end
            new_len = len(text) - len(chunk) * delta_repeats
            if new_len <= 0:
                break
            text = text[:new_len]
        actual = len(enc.encode(text))
    return text, actual


def _probe(target_tokens: int) -> tuple[bool, str, int]:
    """Returns (ok, detail, actual_tokens_used). Retries once on 5xx/429."""
    enc = tiktoken.get_encoding("cl100k_base")
    filler, actual = _build_text_of_n_tokens(target_tokens, enc)
    prompt = (
        "You will receive a long passage of filler text. "
        "Ignore it. After the passage, when asked, reply with only: ACK.\n\n"
        f"PASSAGE:\n{filler}\n\n"
        "Question: please reply with ACK only."
    )
    body = {
        "model": config.MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Authorization": f"Bearer {config.API_KEY}",
        "Content-Type": "application/json",
    }
    last_detail = ""
    for attempt in range(MAX_PROBE_RETRIES + 1):
        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=RESPONSE_TIMEOUT) as h:
                resp = h.post(
                    config.BASE_URL + "/chat/completions",
                    json=body, headers=headers,
                )
            latency = time.perf_counter() - t0
            if resp.status_code == 200:
                return True, (
                    f"actual_tokens≈{actual} (target={target_tokens}) | "
                    f"status=200 | latency={latency:.1f}s"
                ), actual
            last_detail = (
                f"actual_tokens≈{actual} (target={target_tokens}) | "
                f"status={resp.status_code} | latency={latency:.1f}s | "
                f"body={short(resp.text, 160)}"
            )
            # 5xx and 429 are worth retrying with back-off
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < MAX_PROBE_RETRIES:
                time.sleep(RETRY_SLEEP)
                continue
            return False, last_detail, actual
        except Exception as e:  # noqa: BLE001
            last_detail = f"actual_tokens≈{actual} | exception={short(str(e), 200)}"
            if attempt < MAX_PROBE_RETRIES:
                time.sleep(RETRY_SLEEP)
                continue
            return False, last_detail, actual
    return False, last_detail, actual


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def run(report: Report) -> None:
    report.section(
        SECTION,
        "09 · Context-window ceiling",
        f"Binary-search for the largest accepted input from {START_LOWER:,} up to "
        f"{config.CONTEXT_UPPER_TOKENS:,} tokens (precision ±{TOLERANCE_TOKENS:,}). "
        f"Run this module standalone (`python -m tests.test_09_context_limit`) "
        f"if test_08 contaminated rate-limit state.",
    )
    if config.SKIP_CONTEXT_LIMIT:
        report.add(SECTION, "context-limit probe", SKIP, "SKIP_CONTEXT_LIMIT=1 set")
        return

    # Cooldown after test_08 burst — bumped to 90s for round 3 because 35s
    # wasn't enough to recover from the concurrency=100 wave.
    cooldown = int(os.getenv("CONTEXT_PROBE_COOLDOWN_S", "90"))
    if cooldown > 0:
        report.add(
            SECTION,
            f"rate-limit cooldown ({cooldown}s)",
            INFO,
            "letting the gateway's per-key budget recover after test_08",
        )
        time.sleep(cooldown)

    lower = START_LOWER
    upper = config.CONTEXT_UPPER_TOKENS

    # Baseline at the new low (2 000 tokens) — if this fails, the gateway
    # has a fundamental problem unrelated to context size.
    ok, detail, _ = _probe(lower)
    report.add(SECTION, f"probe @ {lower:,} tokens (baseline)",
               PASS if ok else FAIL, detail)
    if not ok:
        report.add(SECTION, "context-limit probe", FAIL,
                   "baseline (2k) failed — gateway is unhealthy or auth is wrong; "
                   "try running this module standalone after a fresh login")
        return

    # Sanity-check the upper end too. Often it will fail fast, which is fine —
    # that tells us we need to binary-search between START_LOWER and upper.
    ok_top, detail_top, _ = _probe(upper)
    report.add(SECTION, f"probe @ {upper:,} tokens (ceiling sanity)",
               PASS if ok_top else INFO, detail_top)

    best_known_ok = lower
    if ok_top:
        best_known_ok = upper
        report.add(
            SECTION,
            "context-limit probe",
            PASS,
            f"gateway accepted {upper:,} tokens — upper bound was not actually a limit; "
            "raise CONTEXT_UPPER_TOKENS to push further",
        )
        return

    # ── Binary search between known-good `lower` and known-bad `upper` ────
    bad = upper
    steps: list[str] = []
    iteration = 0
    while bad - best_known_ok > TOLERANCE_TOKENS:
        iteration += 1
        mid = (best_known_ok + bad) // 2
        # Per-iteration breathing room — keeps from triggering rate limits
        # mid-search.
        if iteration > 1:
            time.sleep(3)
        mid_ok, mid_detail, _ = _probe(mid)
        steps.append(f"- step {iteration}: mid={mid:,} → {'OK' if mid_ok else 'FAIL'} | {mid_detail}")
        if mid_ok:
            best_known_ok = mid
        else:
            bad = mid

    report.add(
        SECTION,
        "binary search converged",
        PASS,
        f"largest accepted input ≈ {best_known_ok:,} tokens "
        f"(±{TOLERANCE_TOKENS:,}) | smallest rejection at ≥ {bad:,}",
    )
    if steps:
        report.block(SECTION, "**Binary-search trace:**\n\n" + "\n".join(steps))

    # Headline number repeated in its own row so it's grep-able.
    report.add(
        SECTION,
        "🎯 effective context ceiling",
        PASS,
        f"≈ {best_known_ok:,} input tokens (model={config.MODEL})",
    )


if __name__ == "__main__":
    # Standalone run: writes results/test_09_only.md so you can read the result
    # without re-running the full suite.
    print("=" * 70)
    print("Context-window ceiling probe — STANDALONE mode")
    print("(skipping the test_08-recovery cooldown is OK here; we're clean)")
    print("=" * 70)
    # Allow standalone runs to skip the 90s cooldown.
    os.environ.setdefault("CONTEXT_PROBE_COOLDOWN_S", "0")
    r = Report()
    run(r)
    from pathlib import Path
    out = Path("results/test_09_only.md")
    r.write_markdown(out)
    print(f"\nWrote {out}")
