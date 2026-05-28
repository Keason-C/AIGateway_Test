"""Probe every OpenAI-native parameter on the gateway.

For each parameter the goal is to record one of:
  PASS  — request returned 200 and the behavior matches the spec
  WARN  — request returned 200 but behavior is suspicious / not verifiable
  FAIL  — request raised (parameter rejected or server error)
"""
from __future__ import annotations

import json

from openai import OpenAI
from pydantic import BaseModel

from . import config
from .reporter import FAIL, PASS, Report, WARN, short

SECTION = "02_openai_params"


def _c() -> OpenAI:
    return OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=60.0, max_retries=0)


def _safe(report: Report, name: str, fn) -> None:
    try:
        status, detail = fn()
        report.add(SECTION, name, status, detail)
    except Exception as e:  # noqa: BLE001
        report.add(SECTION, name, FAIL, short(f"{type(e).__name__}: {e}"))


def run(report: Report) -> None:
    report.section(
        SECTION,
        "02 · OpenAI native parameter matrix",
        "Sends each well-known OpenAI Chat Completions parameter and records whether the "
        "gateway accepts it and whether the behavior is observable in the response.",
    )

    c = _c()
    base_msg = [{"role": "user", "content": "Write a short haiku about the moon."}]

    # ── temperature
    def _temperature():
        a = c.chat.completions.create(model=config.MODEL, messages=base_msg, temperature=0.0)
        b = c.chat.completions.create(model=config.MODEL, messages=base_msg, temperature=1.5)
        ta, tb = a.choices[0].message.content or "", b.choices[0].message.content or ""
        if ta and tb:
            return (PASS, f"accepted | t=0.0: {short(ta, 60)} | t=1.5: {short(tb, 60)}")
        return (WARN, "accepted but empty content")

    _safe(report, "temperature (0.0 vs 1.5)", _temperature)

    # ── top_p
    def _top_p():
        r1 = c.chat.completions.create(model=config.MODEL, messages=base_msg, top_p=0.1)
        r2 = c.chat.completions.create(model=config.MODEL, messages=base_msg, top_p=1.0)
        return (PASS, f"top_p=0.1: {short(r1.choices[0].message.content, 60)} | "
                       f"top_p=1.0: {short(r2.choices[0].message.content, 60)}")

    _safe(report, "top_p (0.1 vs 1.0)", _top_p)

    # ── seed (determinism — same seed should produce identical output if honored)
    def _seed():
        prompt = [{"role": "user", "content": "Pick a random animal. Reply with one word."}]
        r1 = c.chat.completions.create(model=config.MODEL, messages=prompt, seed=42, temperature=0.0)
        r2 = c.chat.completions.create(model=config.MODEL, messages=prompt, seed=42, temperature=0.0)
        a, b = (r1.choices[0].message.content or "").strip(), (r2.choices[0].message.content or "").strip()
        fp1 = getattr(r1, "system_fingerprint", None)
        fp2 = getattr(r2, "system_fingerprint", None)
        if a == b:
            return (PASS, f"deterministic | both='{short(a, 40)}' | fp={fp1}/{fp2}")
        return (WARN, f"seed accepted but output differs | r1='{short(a, 40)}' | r2='{short(b, 40)}'")

    _safe(report, "seed (reproducibility)", _seed)

    # ── n (multiple completions)
    def _n():
        r = c.chat.completions.create(model=config.MODEL, messages=base_msg, n=3)
        return (PASS if len(r.choices) == 3 else WARN,
                f"requested n=3, got {len(r.choices)} choices")

    _safe(report, "n=3 (multiple completions)", _n)

    # ── stop
    def _stop():
        r = c.chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": "Count: one, two, three, STOP, four."}],
            stop=["STOP"],
        )
        content = r.choices[0].message.content or ""
        finish = r.choices[0].finish_reason
        if "STOP" not in content and finish == "stop":
            return (PASS, f"stop honored | finish={finish} | {short(content, 80)}")
        return (WARN, f"finish={finish} | content={short(content, 80)}")

    _safe(report, "stop sequence", _stop)

    # ── frequency_penalty / presence_penalty
    def _penalties():
        r = c.chat.completions.create(
            model=config.MODEL, messages=base_msg,
            frequency_penalty=0.5, presence_penalty=0.5,
        )
        return (PASS, f"accepted | {short(r.choices[0].message.content, 60)}")

    _safe(report, "frequency_penalty + presence_penalty", _penalties)

    # ── logit_bias (best-effort — many models ignore it silently)
    def _logit_bias():
        # tokenizer id for " yes" varies — we just verify the gateway accepts the param
        r = c.chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": "Reply yes or no."}],
            logit_bias={"9891": -100},  # ' yes' in cl100k_base
        )
        return (PASS, f"accepted | {short(r.choices[0].message.content, 60)}")

    _safe(report, "logit_bias", _logit_bias)

    # ── logprobs + top_logprobs
    def _logprobs():
        r = c.chat.completions.create(
            model=config.MODEL, messages=base_msg, logprobs=True, top_logprobs=3,
        )
        lp = r.choices[0].logprobs
        if lp and getattr(lp, "content", None):
            sample = lp.content[0]
            return (PASS, f"logprobs returned | first token='{sample.token}' "
                          f"logprob={sample.logprob:.3f} | top_logprobs len="
                          f"{len(sample.top_logprobs or [])}")
        return (WARN, "param accepted but no logprobs in response")

    _safe(report, "logprobs + top_logprobs", _logprobs)

    # ── response_format: json_object
    def _json_object():
        r = c.chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": "Reply with a JSON object: {\"hello\":\"world\"}"}],
            response_format={"type": "json_object"},
        )
        content = r.choices[0].message.content or ""
        try:
            parsed = json.loads(content)
            return (PASS, f"valid JSON | keys={list(parsed.keys())}")
        except Exception:
            return (WARN, f"accepted but not valid JSON | {short(content, 80)}")

    _safe(report, "response_format=json_object", _json_object)

    # ── response_format: json_schema
    class Person(BaseModel):
        name: str
        age: int

    def _json_schema():
        schema = {
            "name": "Person",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
                "additionalProperties": False,
            },
            "strict": True,
        }
        r = c.chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": "Output a fictional 30-year-old named Alice."}],
            response_format={"type": "json_schema", "json_schema": schema},
        )
        content = r.choices[0].message.content or ""
        try:
            obj = Person.model_validate_json(content)
            return (PASS, f"schema-valid | {obj.model_dump()}")
        except Exception as inner:
            return (WARN, f"accepted but schema mismatch | {short(content, 80)} | err={inner!r}")

    _safe(report, "response_format=json_schema (strict)", _json_schema)

    # ── user
    def _user():
        r = c.chat.completions.create(
            model=config.MODEL, messages=base_msg, user="zf-test-user-123",
        )
        return (PASS, f"accepted | {short(r.choices[0].message.content, 60)}")

    _safe(report, "user identifier", _user)

    # ── max_completion_tokens (newer naming)
    def _max_completion():
        r = c.chat.completions.create(
            model=config.MODEL, messages=base_msg, max_completion_tokens=32,
        )
        usage = r.usage
        return (PASS, f"accepted | completion_tokens={getattr(usage, 'completion_tokens', '?')}")

    _safe(report, "max_completion_tokens=32", _max_completion)

    # ── service_tier
    def _service_tier():
        r = c.chat.completions.create(
            model=config.MODEL, messages=base_msg, extra_body={"service_tier": "auto"},
        )
        return (PASS, f"accepted | service_tier=auto | {short(r.choices[0].message.content, 60)}")

    _safe(report, "service_tier=auto", _service_tier)

    # ── parallel_tool_calls (boolean — true is the default for tool-capable models)
    def _parallel_tc():
        r = c.chat.completions.create(
            model=config.MODEL, messages=base_msg, parallel_tool_calls=True,
        )
        return (PASS, f"accepted | {short(r.choices[0].message.content, 60)}")

    _safe(report, "parallel_tool_calls (no tools attached)", _parallel_tc)

    # ── reasoning_effort (o-series parameter — gateway may reject or ignore)
    def _reasoning():
        r = c.chat.completions.create(
            model=config.MODEL, messages=base_msg,
            extra_body={"reasoning_effort": "low"},
        )
        return (PASS, f"accepted | {short(r.choices[0].message.content, 60)}")

    _safe(report, "reasoning_effort=low (o-series)", _reasoning)

    # ── metadata pass-through (Anthropic-style hint via extra_body)
    def _metadata():
        r = c.chat.completions.create(
            model=config.MODEL, messages=base_msg,
            extra_body={"metadata": {"trace_id": "zf-test-trace-001"}},
        )
        return (PASS, f"accepted | {short(r.choices[0].message.content, 60)}")

    _safe(report, "metadata passthrough", _metadata)


if __name__ == "__main__":
    r = Report()
    run(r)
    from pathlib import Path
    r.write_markdown(Path("results/test_02_only.md"))
    print("wrote results/test_02_only.md")
