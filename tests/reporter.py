"""Result collector. Each test module calls .add(...) and the runner dumps to markdown."""
from __future__ import annotations

import datetime as _dt
import platform
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Status icons
PASS = "✅"
FAIL = "❌"
WARN = "⚠️"
SKIP = "⏭️"
INFO = "ℹ️"


@dataclass
class Row:
    name: str
    status: str  # one of PASS/FAIL/WARN/SKIP/INFO
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Section:
    title: str
    description: str = ""
    rows: list[Row] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)  # free-form markdown chunks

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.rows if r.status == PASS)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.rows if r.status == FAIL)

    @property
    def warn_count(self) -> int:
        return sum(1 for r in self.rows if r.status == WARN)


class Report:
    def __init__(self) -> None:
        self.sections: dict[str, Section] = {}
        self.start = _dt.datetime.now()
        self.env_info: dict[str, str] = {}

    def section(self, key: str, title: str | None = None, description: str = "") -> Section:
        if key not in self.sections:
            self.sections[key] = Section(title or key, description)
        return self.sections[key]

    def add(
        self,
        section_key: str,
        name: str,
        status: str,
        detail: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.section(section_key).rows.append(Row(name, status, detail, extra or {}))

    def block(self, section_key: str, markdown: str) -> None:
        self.section(section_key).blocks.append(markdown)

    def capture_exception(self, section_key: str, name: str, exc: BaseException) -> None:
        tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        self.add(section_key, name, FAIL, f"`{tb}`")

    def write_markdown(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        elapsed = (_dt.datetime.now() - self.start).total_seconds()

        out: list[str] = []
        out.append("# ZF AI Gateway — Test Report")
        out.append("")
        out.append(f"- Generated: {self.start.isoformat(timespec='seconds')}")
        out.append(f"- Duration: {elapsed:.1f}s")
        out.append(f"- Python: {sys.version.split()[0]} on {platform.platform()}")
        for k, v in self.env_info.items():
            out.append(f"- {k}: `{v}`")
        out.append("")

        # Overview table
        out.append("## Overview")
        out.append("")
        out.append("| Section | Pass | Fail | Warn | Notes |")
        out.append("|---|---:|---:|---:|---|")
        for key, sec in self.sections.items():
            notes = sec.description.replace("\n", " ")[:80]
            out.append(
                f"| {sec.title} | {sec.pass_count} | {sec.fail_count} | "
                f"{sec.warn_count} | {notes} |"
            )
        out.append("")

        # Per-section detail
        for key, sec in self.sections.items():
            out.append(f"## {sec.title}")
            if sec.description:
                out.append("")
                out.append(sec.description)
            out.append("")
            if sec.rows:
                out.append("| Check | Status | Detail |")
                out.append("|---|:---:|---|")
                for r in sec.rows:
                    d = r.detail.replace("\n", " ").replace("|", "\\|")
                    if len(d) > 350:
                        d = d[:347] + "…"
                    out.append(f"| {r.name} | {r.status} | {d} |")
                out.append("")
            for blk in sec.blocks:
                out.append(blk)
                out.append("")

        # Auto-generated conclusion
        out.append("## Conclusion")
        out.append("")
        out.append(self._conclusion_md())
        out.append("")

        path.write_text("\n".join(out), encoding="utf-8")

    def _conclusion_md(self) -> str:
        lines: list[str] = []
        total_pass = sum(s.pass_count for s in self.sections.values())
        total_fail = sum(s.fail_count for s in self.sections.values())
        total_warn = sum(s.warn_count for s in self.sections.values())
        lines.append(
            f"- Overall: **{total_pass} pass / {total_fail} fail / {total_warn} warn** across "
            f"{len(self.sections)} sections."
        )

        for key, sec in self.sections.items():
            if sec.fail_count == 0 and sec.warn_count == 0 and sec.pass_count > 0:
                lines.append(f"- {sec.title}: all checks passed.")
            elif sec.fail_count > 0:
                failing = ", ".join(r.name for r in sec.rows if r.status == FAIL)
                lines.append(f"- {sec.title}: **{sec.fail_count} failure(s)** — {failing}.")
            elif sec.warn_count > 0:
                warned = ", ".join(r.name for r in sec.rows if r.status == WARN)
                lines.append(f"- {sec.title}: {sec.warn_count} warning(s) — {warned}.")
        return "\n".join(lines)


def short(text: str, n: int = 120) -> str:
    """Trim and single-line a string for markdown cells."""
    if text is None:
        return ""
    s = str(text).replace("\n", " ").replace("\r", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"
