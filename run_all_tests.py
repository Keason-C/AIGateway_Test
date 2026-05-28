"""Run every test module and produce results/TEST_REPORT.md.

A failure in any single module is captured into the report and does not
prevent the rest of the suite from running.
"""
from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODULES = [
    "tests.test_01_openai_basic",
    "tests.test_02_openai_params",
    "tests.test_03_streaming",
    "tests.test_04_multimodal",
    "tests.test_05_tool_calling",
    "tests.test_06_anthropic_sdk",
    "tests.test_07_maf_integration",
    "tests.test_08_concurrency",
    "tests.test_09_context_limit",
]


def main() -> int:
    # ensure project root on sys.path so `tests.*` imports work regardless of cwd
    sys.path.insert(0, str(ROOT))

    from tests import config
    from tests.reporter import FAIL, Report

    print("=" * 70)
    print("ZF AI Gateway — full test suite")
    print("=" * 70)
    print(config.summary())
    print("=" * 70)

    report = Report()
    report.env_info = {
        "BASE_URL": config.BASE_URL,
        "MODEL": config.MODEL,
        "TOOL_MODEL": config.TOOL_MODEL,
    }

    for mod_name in MODULES:
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
