"""Does the gateway actually honor reasoning-control parameters?

We compare four `reasoning_effort` levels and several alternative parameter shapes
on a single moderately-hard prompt. For each call we record:

  - server latency
  - completion_tokens
  - reasoning_tokens (from usage.completion_tokens_details, if the gateway returns it)
  - whether the visible answer was correct

If the parameter is honored we expect the numbers to *change monotonically* with
effort (high > medium > low > minimal). If every call returns nearly identical
counts, the gateway is silently dropping the param.

Run this in BOTH "默认 mode" and "Reasoning mode" — only the latter should show
real differences. Comparing the two reports tells you whether the gateway
routes `reasoning_effort` differently per assistant.
"""
from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass

from openai import OpenAI

from . import config
from .reporter import FAIL, INFO, PASS, Report, SKIP, WARN, short

SECTION = "10_reasoning_params"

# A classic age-puzzle that needs a few algebra steps. Not impossibly hard
# (even GPT-4o usually gets it), but enough that "high" reasoning models will
# spend more thinking tokens than "low".
HARD_PROMPT = (
    "Solve this puzzle and answer with exactly two integers separated by a comma, "
    "chickens first then rabbits, with no other text:\n"
    "A farmer has chickens and rabbits. Together they have 35 heads and 94 legs. "
    "How many chickens and how many rabbits?"
)
EXPECTED_ANSWER = (23, 12)
ANSWER_RE = re.compile(r"(\d+)\s*[,，]\s*(\d+)")

EFFORTS = ["minimal", "low", "medium", "high"]


@dataclass
class ProbeResult:
    label: str
    latency: float
    completion_tokens: int | None
    reasoning_tokens: int | None
    answer: str
    correct: bool
    error: str = ""

    def as_row(self) -> str:
        rt = self.reasoning_tokens if self.reasoning_tokens is not None else "—"
        ct = self.completion_tokens if self.completion_tokens is not None else "—"
        return (f"| {self.label} | {self.latency:.2f}s | {ct} | {rt} | "
                f"{'✅' if self.correct else '❌'} | `{short(self.answer, 60)}` |")


def _client() -> OpenAI:
    return OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY,
                  timeout=180.0, max_retries=0)


def _extract_reasoning_tokens(usage) -> int | None:
    """Read usage.completion_tokens_details.reasoning_tokens if present."""
    if usage is None:
        return None
    details = getattr(usage, "completion_tokens_details", None)
    if details is None and isinstance(usage, dict):
        details = usage.get("completion_tokens_details")
    if details is None:
        return None
    if hasattr(details, "reasoning_tokens"):
        return details.reasoning_tokens
    if isinstance(details, dict):
        return details.get("reasoning_tokens")
    return None


def _check_answer(text: str) -> bool:
    m = ANSWER_RE.search(text or "")
    if not m:
        return False
    return (int(m.group(1)), int(m.group(2))) == EXPECTED_ANSWER


def _probe(label: str, **kwargs) -> ProbeResult:
    """Send the puzzle prompt with the given kwargs; capture timing + usage."""
    c = _client()
    t0 = time.perf_counter()
    try:
        resp = c.chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": HARD_PROMPT}],
            **kwargs,
        )
        latency = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return ProbeResult(
            label=label,
            latency=latency,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            reasoning_tokens=_extract_reasoning_tokens(usage),
            answer=text.strip(),
            correct=_check_answer(text),
        )
    except Exception as e:  # noqa: BLE001
        return ProbeResult(
            label=label,
            latency=time.perf_counter() - t0,
            completion_tokens=None,
            reasoning_tokens=None,
            answer="",
            correct=False,
            error=short(str(e), 200),
        )


def _detect_variation(results: list[ProbeResult]) -> tuple[bool, str]:
    """Decide whether reasoning_effort produced *any* observable variation."""
    cts = [r.completion_tokens for r in results if r.completion_tokens is not None]
    rts = [r.reasoning_tokens for r in results if r.reasoning_tokens is not None]
    lats = [r.latency for r in results if r.latency > 0]

    # If even one effort level returned a non-zero reasoning_tokens, that's
    # already conclusive evidence the backend is a reasoning model.
    any_reasoning = any((r.reasoning_tokens or 0) > 0 for r in results)
    if any_reasoning and rts:
        spread = max(rts) - min(rts)
        # Need *meaningful* spread across efforts to count as honored
        if spread >= max(rts) * 0.25:  # at least 25 % spread
            return True, (
                f"reasoning_tokens spread {min(rts)}–{max(rts)} (Δ {spread}) "
                f"across efforts — gateway honors the param"
            )
        return False, (
            f"reasoning_tokens reported but barely changes "
            f"({min(rts)}–{max(rts)}, Δ {spread}) — param probably ignored"
        )

    # No reasoning_tokens field at all → fall back to completion_tokens / latency
    if cts and (max(cts) - min(cts)) >= max(cts) * 0.25:
        return True, (
            f"completion_tokens varies {min(cts)}–{max(cts)} — "
            f"effort levels probably do something"
        )
    if lats and (max(lats) - min(lats)) >= 2.0:  # ≥ 2 s spread
        return True, (
            f"latency spreads {min(lats):.1f}–{max(lats):.1f}s — "
            f"effort levels probably do something"
        )
    return False, (
        f"all efforts looked the same — completion_tokens={cts} "
        f"latency_s={[round(x,1) for x in lats]} reasoning_tokens={rts or 'absent'}"
    )


def run(report: Report) -> None:
    report.section(
        SECTION,
        "10 · Reasoning parameter effectiveness",
        f"Hard prompt + sweep over `reasoning_effort` levels and alternative param shapes. "
        f"Compares latency / completion_tokens / reasoning_tokens to decide whether the "
        f"gateway actually routes the parameter to a reasoning backend.",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Sanity: does the gateway ever return usage.completion_tokens_details?
    # ─────────────────────────────────────────────────────────────────────────
    baseline = _probe("baseline (no reasoning params)")
    if baseline.error:
        report.add(SECTION, "baseline call", FAIL, baseline.error)
        return
    if baseline.reasoning_tokens is None:
        report.add(
            SECTION,
            "usage.completion_tokens_details.reasoning_tokens present?",
            INFO,
            "field absent — cannot detect reasoning directly; will rely on token-count / latency spread",
        )
    else:
        report.add(
            SECTION,
            "usage.completion_tokens_details.reasoning_tokens present?",
            PASS,
            f"reasoning_tokens={baseline.reasoning_tokens} on baseline call",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Variant A — flat extra_body={"reasoning_effort": <level>}
    # ─────────────────────────────────────────────────────────────────────────
    flat_results: list[ProbeResult] = [baseline]
    for effort in EFFORTS:
        r = _probe(
            f"flat reasoning_effort={effort}",
            extra_body={"reasoning_effort": effort},
        )
        flat_results.append(r)
        if r.error:
            report.add(SECTION, r.label, FAIL, r.error)
        else:
            report.add(
                SECTION,
                r.label,
                PASS,
                f"latency={r.latency:.2f}s | completion={r.completion_tokens} | "
                f"reasoning={r.reasoning_tokens if r.reasoning_tokens is not None else '—'} | "
                f"correct={r.correct} | answer={short(r.answer, 40)}",
            )
        time.sleep(2)  # small cooldown so we don't hit rate limit mid-sweep

    flat_varies, flat_detail = _detect_variation(flat_results)
    report.add(
        SECTION,
        "VERDICT — flat reasoning_effort honored?",
        PASS if flat_varies else WARN,
        flat_detail,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Variant B — nested extra_body={"reasoning": {"effort": <level>}}
    # ─────────────────────────────────────────────────────────────────────────
    nested_results: list[ProbeResult] = []
    for effort in EFFORTS:
        r = _probe(
            f"nested reasoning.effort={effort}",
            extra_body={"reasoning": {"effort": effort}},
        )
        nested_results.append(r)
        if r.error:
            report.add(SECTION, r.label, FAIL, r.error)
        else:
            report.add(
                SECTION,
                r.label,
                PASS,
                f"latency={r.latency:.2f}s | completion={r.completion_tokens} | "
                f"reasoning={r.reasoning_tokens if r.reasoning_tokens is not None else '—'} | "
                f"correct={r.correct}",
            )
        time.sleep(2)

    nested_varies, nested_detail = _detect_variation(nested_results)
    report.add(
        SECTION,
        "VERDICT — nested reasoning.effort honored?",
        PASS if nested_varies else WARN,
        nested_detail,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Anthropic-style thinking — does the gateway translate it for OpenAI clients?
    # ─────────────────────────────────────────────────────────────────────────
    r_thinking = _probe(
        "anthropic-style thinking.enabled (budget=4096)",
        extra_body={"thinking": {"type": "enabled", "budget_tokens": 4096}},
    )
    if r_thinking.error:
        report.add(SECTION, r_thinking.label, FAIL, r_thinking.error)
    else:
        delta_ct = (r_thinking.completion_tokens or 0) - (baseline.completion_tokens or 0)
        report.add(
            SECTION,
            r_thinking.label,
            PASS,
            f"latency={r_thinking.latency:.2f}s | completion={r_thinking.completion_tokens} "
            f"(Δ vs baseline {delta_ct:+d}) | reasoning={r_thinking.reasoning_tokens or '—'} | "
            f"correct={r_thinking.correct}",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Verify the OpenAI-claim that temperature is ignored on reasoning models.
    # ─────────────────────────────────────────────────────────────────────────
    r_temp0 = _probe(
        "temperature=0 + reasoning_effort=high",
        temperature=0.0,
        extra_body={"reasoning_effort": "high"},
    )
    r_temp2 = _probe(
        "temperature=2 + reasoning_effort=high",
        temperature=2.0,
        extra_body={"reasoning_effort": "high"},
    )
    if r_temp0.error or r_temp2.error:
        report.add(SECTION, "temperature vs reasoning interaction",
                   WARN, f"t=0 err={r_temp0.error or '—'} | t=2 err={r_temp2.error or '—'}")
    else:
        same = r_temp0.answer.strip() == r_temp2.answer.strip()
        report.add(
            SECTION,
            "temperature vs reasoning interaction",
            INFO,
            f"identical_output_across_t0_vs_t2={same} | "
            f"if identical → temperature is being ignored under reasoning, matching OpenAI's spec",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Side-by-side summary block
    # ─────────────────────────────────────────────────────────────────────────
    md = ["", "**Flat `reasoning_effort` sweep:**", "",
          "| Variant | Latency | completion_tokens | reasoning_tokens | Correct | Answer |",
          "|---|---:|---:|---:|:---:|---|"]
    for r in flat_results:
        md.append(r.as_row())

    md += ["", "**Nested `reasoning.effort` sweep:**", "",
           "| Variant | Latency | completion_tokens | reasoning_tokens | Correct | Answer |",
           "|---|---:|---:|---:|:---:|---|"]
    for r in nested_results:
        md.append(r.as_row())

    # Overall interpretive line
    if flat_varies or nested_varies:
        md += ["", "**结论:** Gateway 真的把 reasoning_effort 转发到了后端（看 token / 延迟差异）。"]
    else:
        md += ["", "**结论:** Gateway 接受了参数但**没把它传给后端** — token / 延迟在四个 effort 等级几乎没变化。"]
    report.block(SECTION, "\n".join(md))


if __name__ == "__main__":
    r = Report()
    run(r)
    from pathlib import Path
    r.write_markdown(Path("results/test_10_only.md"))
    print("wrote results/test_10_only.md")
