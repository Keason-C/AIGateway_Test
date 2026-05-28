"""Binary-search the gateway's effective input-token ceiling.

Strategy:
 - Lower bound starts at 8k tokens (we know small calls work).
 - Upper bound starts at CONTEXT_UPPER_TOKENS (200k by default).
 - Each probe constructs a user message of approximately N cl100k_base tokens,
   asks the model to repeat one short phrase (so the response stays small), and
   classifies the outcome as success or failure.
 - Converge until upper-lower ≤ tolerance and report the largest known-good size.
"""
from __future__ import annotations

import string
import time

import httpx
import tiktoken

from . import config
from .reporter import FAIL, INFO, PASS, Report, SKIP, WARN, short

SECTION = "09_context_limit"

TOLERANCE_TOKENS = 4_000  # binary-search precision
RESPONSE_TIMEOUT = 180.0
START_LOWER = 8_000


def _build_text_of_n_tokens(n: int, enc) -> tuple[str, int]:
    """Return a text whose token count is approximately `n`. Uses single-token words."""
    # Use a deterministic stream of short words; each "lorem " encodes to ~2 tokens.
    word = "lorem "
    word_tokens = len(enc.encode(word))
    needed = max(1, n // word_tokens)
    text = word * needed
    actual = len(enc.encode(text))
    return text, actual


def _probe(token_size: int) -> tuple[bool, str]:
    enc = tiktoken.get_encoding("cl100k_base")
    filler, actual = _build_text_of_n_tokens(token_size, enc)
    prompt = (
        "You will receive a long passage of filler text. "
        "Ignore it. After the passage, when I ask, "
        "just reply with the single word: ACK.\n\n"
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
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=RESPONSE_TIMEOUT) as h:
            resp = h.post(config.BASE_URL + "/chat/completions", json=body, headers=headers)
        latency = time.perf_counter() - t0
        if resp.status_code == 200:
            return True, f"actual_tokens≈{actual} | status=200 | latency={latency:.1f}s"
        return False, (
            f"actual_tokens≈{actual} | status={resp.status_code} | latency={latency:.1f}s | "
            f"body={short(resp.text, 180)}"
        )
    except Exception as e:  # noqa: BLE001
        return False, f"actual_tokens≈{actual} | exception={short(str(e), 200)}"


def run(report: Report) -> None:
    report.section(
        SECTION,
        "09 · Context-window ceiling",
        f"Binary-search for the largest accepted input from {START_LOWER:,} up to "
        f"{config.CONTEXT_UPPER_TOKENS:,} tokens (precision ±{TOLERANCE_TOKENS:,}).",
    )
    if config.SKIP_CONTEXT_LIMIT:
        report.add(SECTION, "context-limit probe", SKIP, "SKIP_CONTEXT_LIMIT=1 set")
        return

    lower = START_LOWER
    upper = config.CONTEXT_UPPER_TOKENS

    # Sanity-check the lower bound first
    ok, detail = _probe(lower)
    report.add(SECTION, f"probe @ {lower:,} tokens (baseline)", PASS if ok else FAIL, detail)
    if not ok:
        report.add(SECTION, "context-limit probe", FAIL,
                   "lower-bound baseline failed; gateway is too small for the configured search range")
        return

    # And the upper bound (often this already fails fast)
    ok_top, detail_top = _probe(upper)
    report.add(SECTION, f"probe @ {upper:,} tokens (top)", PASS if ok_top else INFO, detail_top)

    best_known_ok = lower
    if ok_top:
        best_known_ok = upper
        report.add(SECTION, "context-limit probe", PASS,
                   f"gateway accepted {upper:,} tokens — the upper bound was not actually a limit")
    else:
        # Binary search between lower (known-good) and upper (known-bad)
        bad = upper
        steps: list[str] = []
        while bad - best_known_ok > TOLERANCE_TOKENS:
            mid = (best_known_ok + bad) // 2
            mid_ok, mid_detail = _probe(mid)
            steps.append(f"- mid={mid:,} → {'OK' if mid_ok else 'FAIL'} | {mid_detail}")
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


if __name__ == "__main__":
    r = Report()
    run(r)
    from pathlib import Path
    r.write_markdown(Path("results/test_09_only.md"))
    print("wrote results/test_09_only.md")
