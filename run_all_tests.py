"""Run every test module and produce results/TEST_REPORT.md.

A failure in any single module is captured into the report and does not
prevent the rest of the suite from running.

Round-4 mode (current default):
  Round 3 produced clean passes for sections 01, 02, 03, 06, 08, 10, 11.
  We skip them by default to save quota and focus on the open failures:
    - 04 Multimodal       (HTTPS image URL still 500)
    - 05 Tool calling     (OpenAI role:"tool" continuation 500 — all 3 shapes)
    - 07 MAF integration  (downstream of 05)
    - 09 Context limit    (baseline 2k probe 500)
    - 12 Anthropic tools  (NEW — full Anthropic tool_result continuation)

Run the full suite again with `python run_all_tests.py --full`.
"""
from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Sections that already passed in Round 3 — re-enable with --full.
PASSED_MODULES = [
    "tests.test_01_openai_basic",
    "tests.test_02_openai_params",
    "tests.test_03_streaming",
    "tests.test_06_anthropic_sdk",
    "tests.test_10_reasoning_params",
    "tests.test_11_temperature_effectiveness",
    "tests.test_08_concurrency",
]

# Sections that still have ❌ failures, plus the new Anthropic tool round-trip.
FAILING_MODULES = [
    "tests.test_04_multimodal",
    "tests.test_05_tool_calling",
    "tests.test_07_maf_integration",
    "tests.test_12_anthropic_tools",
    # Context limit is quota-heavy, keep it last.
    "tests.test_09_context_limit",
]


def main() -> int:
    # ensure project root on sys.path so `tests.*` imports work regardless of cwd
    sys.path.insert(0, str(ROOT))

    full = "--full" in sys.argv
    modules = (PASSED_MODULES + FAILING_MODULES) if full else FAILING_MODULES

    from tests import config
    from tests.reporter import FAIL, Report

    print("=" * 70)
    print(f"ZF AI Gateway — {'FULL' if full else 'ROUND-4 (failures only)'} test suite")
    print("=" * 70)
    print(config.summary())
    print(f"Modules to run: {len(modules)}")
    print("=" * 70)

    report = Report()
    report.env_info = {
        "BASE_URL": config.BASE_URL,
        "MODEL": config.MODEL,
        "TOOL_MODEL": config.TOOL_MODEL,
        "SUITE_MODE": "full" if full else "round-4 (failures + new anthropic tools)",
    }

    for mod_name in modules:
        print(f"\n>>> {mod_name}")
        t0 = time.perf_counter()
        try:
            mod = importlib.import_module(mod_name)
            run_fn = getattr(mod, "run", None)
            if run_fn is None:
                report.add(mod_name, "module loader", FAIL, "no run(report) function found")
                continue
            run_fn(report)
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            print(tb)
            report.add(mod_name, "module crashed", FAIL,
                       tb.splitlines()[-1] if tb.splitlines() else "unknown")
        finally:
            print(f"    elapsed {time.perf_counter() - t0:.1f}s")

    out = ROOT / "results" / "TEST_REPORT.md"
    report.write_markdown(out)
    print("\n" + "=" * 70)
    print(f"Report written to: {out}")
    print("Copy this file and paste it back to Claude.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
