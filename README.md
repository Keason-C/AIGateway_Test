# ZF AI Gateway — Compatibility & Performance Test Suite

A one-shot test harness for the ZF internal AI Assistant Suite gateway (`https://ai-assistant-suite-staging.azurewebsites.net`). Run it on a machine that **can** reach the gateway (company server / VPN), then send the generated `results/TEST_REPORT.md` back for analysis.

> **Default run is just section 13.** `python run_all_tests.py` runs only the MAF compatibility matrix (**13**) — it answers "does MAF work on the gateway, and which client do you pick?" in one table. Everything else (the deep MAF diagnostics 07/12 and the non-MAF sections 01-06, 08-11) is kept on disk and only runs with `--full`. Failures in the report include a collapsible **🪵 Failure logs** block with the full traceback for debugging.

## What it tests

| # | Test module | What it answers |
|---|---|---|
| 01 | `test_01_openai_basic` | Does the OpenAI SDK even talk to the gateway? `GET /v1/models`, single-turn, multi-turn, wrong-key 401, wrong-model 404, the documented `max_tokens` → 500 pitfall. |
| 02 | `test_02_openai_params` | Which OpenAI-native parameters does the gateway honor? `temperature`, `top_p`, `seed`, `n`, `stop`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `logprobs`+`top_logprobs`, `response_format` (`json_object` and `json_schema`), `user`, `max_completion_tokens`, `service_tier`, `reasoning_effort` (o-series). |
| 03 | `test_03_streaming` | SSE streaming and `stream_options.include_usage`. |
| 04 | `test_04_multimodal` | Image input (base64 + HTTPS URL) and PDF input — three different strategies (PyPDF2 text extract, pymupdf page-as-image, OpenAI native `file` block). |
| 05 | `test_05_tool_calling` | Function-calling end-to-end: `tools`, `tool_choice` (`auto` / `required` / `none` / specific), `parallel_tool_calls`, `tool_result` continuation. |
| 06 | `test_06_anthropic_sdk` | Confirms `/v1/messages` also works via the official Anthropic Python SDK. |
| 07 | `test_07_maf_integration` | Microsoft Agent Framework 1.0 driving the gateway via `OpenAIChatCompletionClient`. Single-turn, streaming, `@tool` function calling, sequential 2-agent workflow. |
| 08 | `test_08_concurrency` | Concurrency ramp 1 → 5 → 10 → 20 → 50 → 100, records p50/p95/max latency, identifies the first level that starts failing (= practical concurrency ceiling). |
| 09 | `test_09_context_limit` | Binary search for the largest user-message size the gateway will accept, measured in cl100k_base tokens. |
| 10 | `test_10_reasoning_params` | Whether the gateway honors `reasoning_effort` (o-series) — compares token/latency across effort levels. |
| 11 | `test_11_temperature_effectiveness` | Whether API-level `temperature` overrides the assistant's baked-in value (inferred from response spread). |
| 12 | `test_12_anthropic_tools` | **(MAF, --full)** Deep diagnostics for MAF + `AnthropicClient` (`/v1/messages`): single-turn, streaming, `@tool` (several forms), multi-tool, raw round-trip, workflow. |
| 13 | `test_13_maf_compatibility` | **(MAF, DEFAULT — headline)** Cross-client matrix: `OpenAIChatClient` (Responses `/responses`, expect 404) vs `OpenAIChatCompletionClient` (Chat Completions, expect ✅) vs `AnthropicClient` (expect ✅), each × [connectivity / streaming / `@tool` / 2-agent workflow]. |

Every test result is collected by `tests/reporter.py` and dumped at the end into a single `results/TEST_REPORT.md`.

## Prerequisites

- Python 3.10+
- A valid API key for an assistant on the gateway (default assistant name: `pureGPT`)
- Outbound HTTPS to `ai-assistant-suite-staging.azurewebsites.net`

## One-shot run

```bash
git clone https://github.com/Keason-C/AIGateway_Test.git
cd AIGateway_Test

python -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env: put your real ZF_API_KEY (and ZF_MODEL if you use a different assistant)

python run_all_tests.py
```

When it finishes, the report is at `results/TEST_REPORT.md`. Send that file back.

## Skipping slow tests

The concurrency and context-ceiling tests are slow (a few minutes each) and use noticeable quota. Skip them with env flags:

```bash
SKIP_CONCURRENCY=1 SKIP_CONTEXT_LIMIT=1 python run_all_tests.py
```

Or run a single module directly:

```bash
python -m tests.test_02_openai_params
```

### Re-running just the context-window probe

Round 3 made `test_09_context_limit` runnable standalone, because the
test_08 concurrency burst sometimes contaminates rate-limit state and makes
the binary-search baseline fail spuriously. To get a clean reading:

```bash
python -m tests.test_09_context_limit
# → writes results/test_09_only.md
```

## Configuration

`.env` keys (only `ZF_API_KEY` is required):

| Key | Default | Notes |
|---|---|---|
| `ZF_API_KEY` | — | Required. Your gateway-issued API key. |
| `ZF_BASE_URL` | `https://ai-assistant-suite-staging.azurewebsites.net/v1` | OpenAI-style base. The Anthropic test strips `/v1` automatically. |
| `ZF_MODEL` | `pureGPT` | Assistant name (gateway treats `model` as the assistant name). |
| `ZF_TOOL_MODEL` | same as `ZF_MODEL` | Optional. If `pureGPT` has no server-side tools, point this at a tool-enabled assistant (e.g. `AIWEB_TEST`) so test_05 has something to validate against. |
| `SKIP_CONCURRENCY` | `0` | `1` to skip test_08. |
| `SKIP_CONTEXT_LIMIT` | `0` | `1` to skip test_09. |
| `SKIP_MAF` | `0` | `1` to skip test_07 (in case `agent-framework` install fails on the server). |
| `CONCURRENCY_LEVELS` | `1,5,10,20,50,100` | Comma-separated. |
| `CONTEXT_UPPER_TOKENS` | `200000` | Upper bound for context binary search. |

## Known gotchas (already baked into the tests)

- The gateway used to return **HTTP 500 when `max_tokens` is sent on `/v1/chat/completions`**, but round-2 testing showed it's been fixed. Test 01 still probes it so we'll know if it regresses.
- There is **no server-side memory**. Multi-turn tests pre-build the `messages` list each call.
- PDFs: the speculative OpenAI `file` block now also works, in addition to the PyPDF2-text and pymupdf-as-image strategies. Test 04 exercises all three.
- The Microsoft Agent Framework's `OpenAIChatClient` hits `/responses` (OpenAI Responses API) and 404s on this gateway. We use `OpenAIChatCompletionClient` instead — which hits `/chat/completions`. Test 13 probes both head-to-head so the report states it plainly.
- **Tool calling on this gateway is known to work** (confirmed by hand). So a `@tool` failure in the MAF sections points at MAF wiring/version or the wrong client, not the gateway. The MAF tests use the canonical shapes from the audited `microsoft-agent-framework` skill, and every failure is logged with its full traceback in the report.
- On the **assistant-with-tool_calls** turn, the gateway 500s if `content: ""` is sent. The spec-correct shape (and MAF's wire shape) is to **omit `content` entirely**. Test 05 now exercises all three variants — omit / null / `""` — so we can see which the gateway accepts.
