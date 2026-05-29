# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **diagnostic test harness** (not a library or app) that probes the ZF internal "AI Assistant Suite" gateway. You run it on a machine that *can* reach the gateway (company server / VPN), and it produces a single `results/TEST_REPORT.md` to send back for analysis.

**Current focus: Microsoft Agent Framework (MAF) compatibility.** The default run answers "does MAF work against our gateway, and which client do you pick?" — the broader OpenAI/Anthropic-parameter probes (sections 01-06, 08-11) are kept on disk but no longer run by default. The repo is an iterative investigation: failures get diagnosed, fixed, and re-probed (see commit history).

> Maintainer's standing note: **tool calling on this gateway works** (confirmed by hand previously). So if a MAF `@tool` cell 500s, suspect the MAF wiring/version or the client choice — *not* the gateway. The new test code deliberately uses the canonical shapes from the audited `microsoft-agent-framework` skill to keep that distinction clean.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then put a real ZF_API_KEY in .env

# Run — DEFAULT is section 13 only (the MAF compatibility matrix)
python run_all_tests.py

# Run the COMPLETE suite (13 + deep MAF diagnostics 07/12 + non-MAF 01-06, 08-11)
python run_all_tests.py --full

# Run / re-probe a single section standalone (writes results/test_NN_only.md)
python -m tests.test_13_maf_compatibility
python -m tests.test_09_context_limit                 # recommended standalone — see below

# Skip slow / quota-heavy sections
SKIP_CONCURRENCY=1 SKIP_CONTEXT_LIMIT=1 SKIP_MAF=1 python run_all_tests.py
```

There is **no pytest, linter, or build step** — `python_requires` is 3.10+. "Tests" are plain modules executed for their side effect (appending rows to a report). `run_all_tests.py` puts the repo root on `sys.path` so `tests.*` and `assets.*` imports resolve regardless of cwd.

⚠️ `python run_all_tests.py` with no flag does **not** run everything — it runs only `DEFAULT_MODULES` (just `tests.test_13_maf_compatibility`). Use `--full` to also run `EXTRA_MODULES` (the deep MAF diagnostics 07/12 and the non-MAF sections). The two lists live at the top of `run_all_tests.py`.

## Architecture

Three pieces plus N independent test modules:

- **`run_all_tests.py`** — orchestrator. Imports each `tests.test_NN_*` module and calls its module-level `run(report)`. Each module is isolated in try/except, so one crash is captured as a FAIL row and the rest still run. Holds the two module lists (`PASSED_MODULES`, `FAILING_MODULES`) that decide default vs `--full` behavior.
- **`tests/config.py`** — single source of config, loaded from `.env` via python-dotenv. `_require()` hard-exits if `ZF_API_KEY` is missing. Derives `ANTHROPIC_BASE_URL` by stripping the trailing `/v1` from `BASE_URL` (the Anthropic SDK re-appends it).
- **`tests/reporter.py`** — the `Report`/`Section`/`Row` model and markdown writer. Status icons: `PASS ✅ / FAIL ❌ / WARN ⚠️ / SKIP ⏭️ / INFO ℹ️`. Output is an overview table + per-section detail tables + an auto-generated conclusion. **`capture_exception()` stashes the full traceback** (incl. any HTTP response body the SDK embedded in the message) in `Row.extra["traceback"]`; the writer renders these as a collapsible "🪵 Failure logs" block per section — so a 500 is debuggable from the report alone, not just a one-line summary. Pass `note=` for a human hint alongside the exception.
- **`assets/generate_assets.py`** — idempotently generates `test_image.jpg` (Pillow) and `test_doc.pdf` (reportlab) on first use. The PDF embeds canary tokens (`ALPHA-FIRST-PAGE-CANARY-7421`, `BETA-SECOND-PAGE-CANARY-8842`) so multimodal/PDF strategies can be verified by checking whether the model recovered the canary.

### The test-module contract (follow this when adding one)

Every `tests/test_NN_name.py`:
1. Defines a `SECTION` string constant.
2. Exposes `def run(report: Report) -> None:` as its only entry point.
3. Calls `report.section(...)` once, then `report.add(SECTION, name, STATUS, detail)` per check.
4. **Never raises out of a check** — wrap each in try/except and record failures via `report.capture_exception(SECTION, name, e)`. The harness should always reach the report-writing step.
5. Distinguishes `PASS` (200 + behavior matches spec), `WARN` (200 but suspicious/unverifiable), `FAIL` (raised / server error).
6. Ends with an `if __name__ == "__main__"` block that runs standalone and writes `results/test_NN_only.md`.
7. Gets registered in the appropriate list in `run_all_tests.py`.

## Gateway-specific knowledge (the reason this repo exists)

These are the gateway's quirks that the tests encode — internalize them before changing test logic:

- **Dual protocol, one key.** The gateway speaks OpenAI at `BASE_URL` (`.../v1`) *and* Anthropic at `ANTHROPIC_BASE_URL` (no `/v1`). Same `ZF_API_KEY` for both.
- **`model` = assistant name, not a model ID.** Values are gateway assistant names like `pureGPT` (default) or `AIWEB_TEST`. Use `ZF_TOOL_MODEL` to point tool tests at a tool-enabled assistant if the default has none.
- **No server-side memory.** Multi-turn requires rebuilding the full `messages` list client-side every call.
- **MAF client choice is the headline finding (the #1 MAF footgun).** MAF's `OpenAIChatClient` — despite the name — calls the OpenAI **Responses API** (`POST /responses`) and 404s against this gateway (and any OpenAI-compatible proxy). You must use **`OpenAIChatCompletionClient`** (`POST /chat/completions`). `test_13_maf_compatibility` probes both head-to-head, plus `AnthropicClient` (`/v1/messages`), across [connectivity / streaming / @tool / workflow]. Construction: OpenAI clients take `base_url=.../v1, api_key, model`; `AnthropicClient(model=…, anthropic_client=AsyncAnthropic(base_url=ANTHROPIC_BASE_URL))`.
- **The `tool_result` continuation path is the active 500.** On the assistant-with-`tool_calls` turn, send `content` *omitted entirely* (MAF's wire shape, spec-compliant) — `content: ""` has 500'd. test_05 probes all three shapes (omit/null/`""`); test_07 (MAF + OpenAI client) and test_12 (MAF + AnthropicClient) are the apples-to-apples diagnostic for whether swapping to the Anthropic protocol fixes it.
- **`max_tokens` on `/chat/completions`** historically returned 500; round-2 found it fixed. test_01 still probes it as a regression canary.
- **Async cleanup matters.** test_07/test_12 build one client for the whole section and explicitly close the inner async HTTP client at the end — otherwise GC closes it after `asyncio.run()` tears down the loop, producing spurious `RuntimeError: Event loop is closed` noise on Windows (ProactorEventLoop + httpx). Note the anthropic SDK uses `close()` (async), **not** `aclose()` like httpx.
- **MAF imports are defensively wrapped** (`_import_*` helpers with fallback candidate lists) because `agent-framework` shifted class locations during pre-release. Keep new MAF code tolerant of both `agent_framework.X` and `agent_framework_provider.X` import paths.
- **`test_09_context_limit` is contamination-sensitive.** The test_08 concurrency burst can poison rate-limit state and make its baseline probe spuriously 500. Run it standalone (`python -m tests.test_09_context_limit`) for a clean reading.

## Configuration (.env)

Only `ZF_API_KEY` is required. Defaults: `ZF_BASE_URL=https://ai-assistant-suite-staging.azurewebsites.net/v1`, `ZF_MODEL=pureGPT`, `ZF_TOOL_MODEL`→`ZF_MODEL`. Skip flags (`SKIP_CONCURRENCY`/`SKIP_CONTEXT_LIMIT`/`SKIP_MAF`) and tuning (`CONCURRENCY_LEVELS`, `CONTEXT_UPPER_TOKENS`) are documented in `.env.example`. `.env`, `results/`, generated assets, and `TEST_REPORT*.md` are git-ignored.
