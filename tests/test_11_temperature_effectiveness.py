"""Does the API-level `temperature` parameter actually override the backend
configured temperature?

The ZF gateway's assistant config has its own `temperature` value baked in.
Question: when a client sends `temperature=0.0` or `temperature=2.0`, does
that *win* over the backend config, or is it silently dropped?

Detection strategy
-----------------
We never see the model's logits, so we infer temperature behaviour from the
*statistical spread* of responses to the same prompt:

  • If temperature is honored:
      t=0.0  →  multiple runs of the same prompt should be **near-identical**
      t=2.0  →  multiple runs of the same prompt should be **highly varied**
  • If temperature is ignored (backend wins):
      both t=0.0 and t=2.0 give the same level of similarity — whatever the
      backend's fixed temperature dictates.

We also do a `seed=42 + temperature=0.0` determinism check: two consecutive
calls with the same seed *should* be identical. If they're not, the gateway
isn't truly running at temperature 0.

We also try out-of-range temperatures (−1, 3, 'hot') to see whether the gateway
validates or accepts silently.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from openai import OpenAI

from . import config
from .reporter import FAIL, INFO, PASS, Report, WARN, short

SECTION = "11_temperature"

# Open-ended creative prompts — more room for variation at high temperatures.
CREATIVE_PROMPTS = [
    "Invent one fictional country name. Reply with only the name, nothing else.",
    "Write a single short sentence (≤ 12 words) about the moon. No markdown.",
    "Suggest one unusual hobby in 5 words or fewer. No punctuation.",
]
N_SAMPLES = 4  # samples per (prompt, temperature) cell
PAIR_SIM_THRESHOLD_T0 = 0.65   # if avg pairwise similarity at t=0 ≥ this → "stable"
PAIR_SIM_DELTA_THRESHOLD = 0.15  # t=0 must be at least this much above t=2 to count as "honored"


@dataclass
class Sample:
    text: str
    latency: float
    completion_tokens: int | None
    error: str = ""


def _client() -> OpenAI:
    return OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY,
                  timeout=60.0, max_retries=0)


def _call(prompt: str, **kwargs) -> Sample:
    t0 = time.perf_counter()
    try:
        r = _client().chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return Sample(
            text=(r.choices[0].message.content or "").strip(),
            latency=time.perf_counter() - t0,
            completion_tokens=getattr(r.usage, "completion_tokens", None) if r.usage else None,
        )
    except Exception as e:  # noqa: BLE001
        return Sample(text="", latency=time.perf_counter() - t0,
                      completion_tokens=None, error=short(str(e), 200))


def _jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity. 1.0 = identical word set."""
    sa = set((a or "").lower().split())
    sb = set((b or "").lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def _avg_pairwise_similarity(samples: list[str]) -> float:
    if len(samples) < 2:
        return 1.0
    sims = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            sims.append(_jaccard(samples[i], samples[j]))
    return statistics.mean(sims) if sims else 1.0


def _identical_fraction(samples: list[str]) -> float:
    """Fraction of pairs that are byte-identical."""
    if len(samples) < 2:
        return 1.0
    eq, total = 0, 0
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            total += 1
            if samples[i].strip() == samples[j].strip():
                eq += 1
    return eq / total if total else 1.0


def run(report: Report) -> None:
    report.section(
        SECTION,
        "11 · Temperature parameter effectiveness",
        "Multi-sample same-prompt similarity comparison to decide whether the "
        "client-passed `temperature` actually overrides the gateway/assistant's "
        "back-end temperature configuration.",
    )

    # ──────────────────────────────────────────────────────────────────
    # 1. Determinism check: same seed + temperature=0 → identical output?
    # ──────────────────────────────────────────────────────────────────
    det_prompt = "Pick a single random color name. Reply with the color word only."
    s1 = _call(det_prompt, temperature=0.0, seed=42)
    s2 = _call(det_prompt, temperature=0.0, seed=42)
    if s1.error or s2.error:
        report.add(SECTION, "determinism: temp=0 + seed=42 (×2)", FAIL,
                   f"err1={s1.error or '—'} | err2={s2.error or '—'}")
    else:
        identical = s1.text.strip() == s2.text.strip()
        report.add(
            SECTION,
            "determinism: temp=0 + seed=42 (×2)",
            PASS if identical else WARN,
            f"identical={identical} | r1='{short(s1.text, 40)}' | r2='{short(s2.text, 40)}'",
        )

    # ──────────────────────────────────────────────────────────────────
    # 2. Spread sweep: collect N samples per (prompt, temperature)
    # ──────────────────────────────────────────────────────────────────
    aggregated: dict[float, list[float]] = {0.0: [], 2.0: []}  # avg-sim per prompt
    identical_pct: dict[float, list[float]] = {0.0: [], 2.0: []}
    all_outputs_t0: list[str] = []
    all_outputs_t2: list[str] = []

    for prompt in CREATIVE_PROMPTS:
        for temp in (0.0, 2.0):
            samples: list[Sample] = []
            for _ in range(N_SAMPLES):
                samples.append(_call(prompt, temperature=temp))
                time.sleep(0.4)  # gentle cooldown
            texts = [s.text for s in samples if not s.error]
            errs = [s.error for s in samples if s.error]
            if len(texts) < 2:
                report.add(
                    SECTION,
                    f"sample sweep — t={temp} — '{short(prompt, 50)}'",
                    FAIL,
                    f"only {len(texts)} usable samples; errors={errs[:2]}",
                )
                continue
            sim = _avg_pairwise_similarity(texts)
            ident = _identical_fraction(texts)
            aggregated[temp].append(sim)
            identical_pct[temp].append(ident)
            (all_outputs_t0 if temp == 0.0 else all_outputs_t2).extend(texts)
            report.add(
                SECTION,
                f"sample sweep — t={temp} — '{short(prompt, 40)}'",
                INFO,
                f"n={len(texts)} | avg pairwise word-Jaccard={sim:.2f} | "
                f"identical_pairs={ident:.0%} | examples={[short(t, 30) for t in texts[:2]]}",
            )

    # ──────────────────────────────────────────────────────────────────
    # 3. VERDICT
    # ──────────────────────────────────────────────────────────────────
    if not aggregated[0.0] or not aggregated[2.0]:
        report.add(SECTION, "VERDICT — temperature honored?", FAIL,
                   "insufficient data to compare")
    else:
        avg_t0 = statistics.mean(aggregated[0.0])
        avg_t2 = statistics.mean(aggregated[2.0])
        ident_t0 = statistics.mean(identical_pct[0.0])
        ident_t2 = statistics.mean(identical_pct[2.0])
        delta = avg_t0 - avg_t2

        if delta >= PAIR_SIM_DELTA_THRESHOLD and avg_t0 >= PAIR_SIM_THRESHOLD_T0:
            verdict = PASS
            interp = (
                f"t=0.0 is {delta:.2f} more self-similar than t=2.0 "
                f"(0.0→{avg_t0:.2f} vs 2.0→{avg_t2:.2f}) — **temperature is honored**"
            )
        elif delta >= PAIR_SIM_DELTA_THRESHOLD:
            verdict = WARN
            interp = (
                f"some spread (Δ={delta:.2f}) but t=0 isn't very stable "
                f"(avg sim {avg_t0:.2f}) — temperature partially honored or backend baseline is mid-range"
            )
        else:
            verdict = WARN
            interp = (
                f"t=0.0 and t=2.0 look about equally varied "
                f"(0.0→{avg_t0:.2f} vs 2.0→{avg_t2:.2f}, Δ={delta:.2f}) — "
                f"**API-level temperature appears to be ignored by the gateway**"
            )
        report.add(
            SECTION,
            "VERDICT — temperature honored?",
            verdict,
            f"{interp} | identical_pairs@t0={ident_t0:.0%} t2={ident_t2:.0%}",
        )

    # ──────────────────────────────────────────────────────────────────
    # 4. Out-of-range value handling
    # ──────────────────────────────────────────────────────────────────
    for bad_val in (-1.0, 3.0):
        r = _call("Say hi.", temperature=bad_val)
        if r.error:
            report.add(
                SECTION,
                f"out-of-range temperature={bad_val}",
                PASS,
                f"rejected | {r.error}",
            )
        else:
            report.add(
                SECTION,
                f"out-of-range temperature={bad_val}",
                WARN,
                f"accepted silently — gateway does not validate range | answer={short(r.text, 60)}",
            )

    # Non-numeric type
    try:
        r = _client().chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"temperature": "hot"},
        )
        report.add(SECTION, "temperature='hot' (string)", WARN,
                   f"accepted | answer={short(r.choices[0].message.content, 60)}")
    except Exception as e:  # noqa: BLE001
        report.add(SECTION, "temperature='hot' (string)", PASS,
                   f"rejected | {short(str(e), 160)}")

    # ──────────────────────────────────────────────────────────────────
    # 5. Side-by-side markdown
    # ──────────────────────────────────────────────────────────────────
    md = [
        "",
        "**Aggregated similarity by temperature (averaged across prompts):**",
        "",
        "| Temperature | Avg pairwise word-Jaccard | Identical-pair rate | Meaning |",
        "|---:|---:|---:|---|",
    ]
    if aggregated[0.0]:
        md.append(
            f"| 0.0 | {statistics.mean(aggregated[0.0]):.2f} | "
            f"{statistics.mean(identical_pct[0.0]):.0%} | "
            f"{'tightly clustered' if statistics.mean(aggregated[0.0]) >= 0.65 else 'mixed'} |"
        )
    if aggregated[2.0]:
        md.append(
            f"| 2.0 | {statistics.mean(aggregated[2.0]):.2f} | "
            f"{statistics.mean(identical_pct[2.0]):.0%} | "
            f"{'highly varied' if statistics.mean(aggregated[2.0]) <= 0.40 else 'mixed'} |"
        )
    if all_outputs_t0:
        md += ["", "**Sample t=0.0 outputs:**", ""]
        for t in all_outputs_t0[:6]:
            md.append(f"- `{short(t, 80)}`")
    if all_outputs_t2:
        md += ["", "**Sample t=2.0 outputs:**", ""]
        for t in all_outputs_t2[:6]:
            md.append(f"- `{short(t, 80)}`")

    report.block(SECTION, "\n".join(md))


if __name__ == "__main__":
    r = Report()
    run(r)
    from pathlib import Path
    r.write_markdown(Path("results/test_11_only.md"))
    print("wrote results/test_11_only.md")
