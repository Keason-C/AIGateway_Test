"""Run the test modules and produce results/TEST_REPORT.md.

A failure in any single module is captured into the report (with its FULL
traceback, see tests/reporter.py) and does not stop the rest of the suite.

Default mode (current):
  By default we run ONLY section 13 — the MAF compatibility matrix
  (3 clients × [connectivity / streaming / @tool / workflow]). That single
  section answers "does MAF work on the gateway, and which client to pick?".

  Everything else — the deep MAF diagnostics (07, 12) and the non-MAF sections
  (01-06, 08-11) — is kept on disk but only runs with `python run_all_tests.py --full`.
"""
from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Default run: just the headline MAF compatibility matrix.
DEFAULT_MODULES = [
    "tests.test_13_maf_compatibility",
]

# Everything else — kept on disk, only run with --full. Includes the deeper
# MAF per-client diagnostics (07, 12) that pinpoint where a 500 originates if
# a cell in section 13 fails, plus all the non-MAF sections.
EXTRA_MODULES = [
    "tests.test_07_maf_integration",
    "tests.test_12_anthropic_tools",
    "tests.test_01_openai_basic",
    "tests.test_02_openai_params",
    "tests.test_03_streaming",
    "tests.test_04_multimodal",
    "tests.test_05_tool_calling",
    "tests.test_06_anthropic_sdk",
    "tests.test_10_reasoning_params",
    "tests.test_11_temperature_effectiveness",
    "tests.test_08_concurrency",
    # Context limit is quota-heavy, keep it last.
    "tests.test_09_context_limit",
]


def main() -> int:
    # ensure project root on sys.path so `tests.*` imports work regardless of cwd
    sys.path.insert(0, str(ROOT))

    full = "--full" in sys.argv
    modules = (DEFAULT_MODULES + EXTRA_MODULES) if full else DEFAULT_MODULES

    from tests import config
    from tests.reporter import FAIL, Report

    print("=" * 70)
    print(f"ZF AI Gateway — {'FULL' if full else 'section 13 only'} test suite")
    print("=" * 70)
    print(config.summary())
    print(f"Modules to run: {len(modules)}")
    print("=" * 70)

    report = Report()
    report.env_info = {
        "BASE_URL": config.BASE_URL,
        "MODEL": config.MODEL,
        "TOOL_MODEL": config.TOOL_MODEL,
        "SUITE_MODE": "full" if full else "section 13 only (MAF compatibility matrix)",
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
