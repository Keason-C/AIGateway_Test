"""Shared configuration loaded from .env / environment variables."""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val or val.strip().lower() in {"replace-me", ""}:
        print(
            f"[config] ERROR: environment variable {name} is not set. "
            f"Copy .env.example to .env and fill in your real value.",
            file=sys.stderr,
        )
        sys.exit(2)
    return val.strip()


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


API_KEY: str = _require("ZF_API_KEY")
BASE_URL: str = os.getenv(
    "ZF_BASE_URL", "https://ai-assistant-suite-staging.azurewebsites.net/v1"
).rstrip("/")
MODEL: str = os.getenv("ZF_MODEL", "pureGPT").strip()
TOOL_MODEL: str = os.getenv("ZF_TOOL_MODEL", MODEL).strip()

# Anthropic base = BASE_URL without trailing /v1
ANTHROPIC_BASE_URL: str = BASE_URL[:-3] if BASE_URL.endswith("/v1") else BASE_URL

SKIP_CONCURRENCY: bool = _flag("SKIP_CONCURRENCY")
SKIP_CONTEXT_LIMIT: bool = _flag("SKIP_CONTEXT_LIMIT")
SKIP_MAF: bool = _flag("SKIP_MAF")

CONCURRENCY_LEVELS: list[int] = [
    int(x) for x in os.getenv("CONCURRENCY_LEVELS", "1,5,10,20,50,100").split(",") if x.strip()
]
CONTEXT_UPPER_TOKENS: int = int(os.getenv("CONTEXT_UPPER_TOKENS", "200000"))


def summary() -> str:
    masked = API_KEY[:4] + "…" + API_KEY[-4:] if len(API_KEY) > 8 else "****"
    return (
        f"BASE_URL={BASE_URL}\n"
        f"MODEL={MODEL}\n"
        f"TOOL_MODEL={TOOL_MODEL}\n"
        f"API_KEY={masked}\n"
        f"SKIP_CONCURRENCY={SKIP_CONCURRENCY} | "
        f"SKIP_CONTEXT_LIMIT={SKIP_CONTEXT_LIMIT} | "
        f"SKIP_MAF={SKIP_MAF}\n"
        f"CONCURRENCY_LEVELS={CONCURRENCY_LEVELS}\n"
        f"CONTEXT_UPPER_TOKENS={CONTEXT_UPPER_TOKENS}"
    )
