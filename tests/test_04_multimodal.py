"""Multimodal inputs: images and PDFs.

Three PDF strategies are exercised because the gateway is known not to accept
a native `document` block:

  A) Client-side text extraction with PyPDF2 (the reference snippet's approach)
  B) Convert each page to a JPEG and send as image_url blocks (the skill's recipe)
  C) Speculative: send an OpenAI-style `file` content block with base64 PDF — we
     just want to know whether the gateway accepts it.
"""
from __future__ import annotations

import base64
import importlib

from openai import OpenAI

from . import config
from .reporter import FAIL, PASS, Report, WARN, short
from assets.generate_assets import ensure_all

SECTION = "04_multimodal"


def _c() -> OpenAI:
    return OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=120.0, max_retries=0)


def _b64(path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def run(report: Report) -> None:
    report.section(
        SECTION,
        "04 · Multimodal",
        "Image input via base64 / URL and PDF input via three different strategies.",
    )

    image_path, pdf_path = ensure_all()

    # 1. Image as data URL (base64)
    try:
        b64 = _b64(image_path)
        r = _c().chat.completions.create(
            model=config.MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Briefly describe what you see in the image."},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
        )
        content = r.choices[0].message.content or ""
        if content.strip():
            report.add(SECTION, "Image input — base64 data URL", PASS, short(content, 140))
        else:
            report.add(SECTION, "Image input — base64 data URL", WARN, "empty content")
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "Image input — base64 data URL", e)

    # 2. Image as HTTPS URL (use a stable Wikimedia asset)
    try:
        r = _c().chat.completions.create(
            model=config.MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What animal is in this picture? One word."},
                    {"type": "image_url",
                     "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"}},
                ],
            }],
        )
        content = (r.choices[0].message.content or "").strip().lower()
        if content:
            ok = "ant" in content
            report.add(
                SECTION,
                "Image input — HTTPS URL",
                PASS if ok else WARN,
                f"answer: {short(content, 80)}",
            )
        else:
            report.add(SECTION, "Image input — HTTPS URL", WARN, "empty content")
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "Image input — HTTPS URL", e)

    # 3. PDF strategy A — client-side text extract with PyPDF2 (reference snippet)
    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(str(pdf_path))
        pdf_text = "\n".join((page.extract_text() or "") for page in reader.pages)
        r = _c().chat.completions.create(
            model=config.MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "Here is the document text:\n\n"
                    f"{pdf_text}\n\n"
                    "What is the marker token on page 1?"
                ),
            }],
        )
        out = (r.choices[0].message.content or "")
        ok = "ALPHA-FIRST-PAGE-CANARY-7421" in out
        report.add(
            SECTION,
            "PDF strategy A — PyPDF2 text extract",
            PASS if ok else WARN,
            f"page-1 marker recovered={ok} | {short(out, 100)}",
        )
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "PDF strategy A — PyPDF2 text extract", e)

    # 4. PDF strategy B — pymupdf each page → JPEG → image_url blocks
    try:
        fitz = importlib.import_module("fitz")  # pymupdf
        doc = fitz.open(str(pdf_path))
        image_blocks = []
        for page in doc:
            png = page.get_pixmap(dpi=150).tobytes("jpeg")
            page_b64 = base64.b64encode(png).decode()
            image_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{page_b64}"},
            })
        r = _c().chat.completions.create(
            model=config.MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": "These are PDF pages as images. What is the marker token on page 1?"},
                    *image_blocks,
                ],
            }],
        )
        out = r.choices[0].message.content or ""
        ok = "ALPHA-FIRST-PAGE-CANARY-7421" in out
        report.add(
            SECTION,
            "PDF strategy B — pymupdf pages as JPEG",
            PASS if ok else WARN,
            f"pages={len(image_blocks)} | marker recovered={ok} | {short(out, 100)}",
        )
    except Exception as e:  # noqa: BLE001
        report.capture_exception(SECTION, "PDF strategy B — pymupdf pages as JPEG", e)

    # 5. PDF strategy C — speculative `file` block (OpenAI's newer content type)
    try:
        b64 = _b64(pdf_path)
        r = _c().chat.completions.create(
            model=config.MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize the attached PDF."},
                    {
                        "type": "file",
                        "file": {
                            "filename": "test_doc.pdf",
                            "file_data": f"data:application/pdf;base64,{b64}",
                        },
                    },
                ],
            }],
        )
        out = r.choices[0].message.content or ""
        report.add(
            SECTION,
            "PDF strategy C — OpenAI `file` block (speculative)",
            PASS if "ALPHA-FIRST-PAGE-CANARY-7421" in out else WARN,
            short(out, 140),
        )
    except Exception as e:  # noqa: BLE001
        report.add(
            SECTION,
            "PDF strategy C — OpenAI `file` block (speculative)",
            FAIL,
            f"gateway rejected the file block: {short(str(e), 200)}",
        )


if __name__ == "__main__":
    r = Report()
    run(r)
    from pathlib import Path
    r.write_markdown(Path("results/test_04_only.md"))
    print("wrote results/test_04_only.md")
