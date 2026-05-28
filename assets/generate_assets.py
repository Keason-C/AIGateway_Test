"""Generate small test assets (image + PDF) on first run.

Idempotent: skips files that already exist. Uses Pillow for the image and
reportlab for the PDF — both pure-Python and present in requirements.txt.
"""
from __future__ import annotations

from pathlib import Path

ASSETS = Path(__file__).resolve().parent
IMAGE_PATH = ASSETS / "test_image.jpg"
PDF_PATH = ASSETS / "test_doc.pdf"


def ensure_image() -> Path:
    if IMAGE_PATH.exists():
        return IMAGE_PATH
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (640, 480), color=(245, 244, 240))
    draw = ImageDraw.Draw(img)
    # Couple of recognizable shapes so a VLM has something to describe
    draw.rectangle([60, 80, 280, 220], outline=(20, 80, 200), width=6)
    draw.ellipse([340, 80, 580, 320], outline=(200, 40, 40), width=6)
    draw.line([60, 380, 580, 380], fill=(20, 140, 20), width=8)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    draw.text((60, 410), "ZF-AI-GATEWAY VLM SMOKE TEST", fill=(10, 10, 10), font=font)
    img.save(IMAGE_PATH, "JPEG", quality=85)
    return IMAGE_PATH


def ensure_pdf() -> Path:
    if PDF_PATH.exists():
        return PDF_PATH
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(PDF_PATH), pagesize=LETTER)
    w, h = LETTER
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, h - 90, "ZF AI Gateway — PDF ingestion test document")
    c.setFont("Helvetica", 12)
    c.drawString(72, h - 130, "Page 1 marker token: ALPHA-FIRST-PAGE-CANARY-7421")
    c.drawString(
        72,
        h - 160,
        "If you can read the canary on page 1, the gateway received the first page.",
    )
    c.drawString(
        72,
        h - 200,
        "This document is intentionally short so it can be base64-encoded inline.",
    )
    c.showPage()

    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, h - 90, "Page 2")
    c.setFont("Helvetica", 12)
    c.drawString(72, h - 130, "Page 2 marker token: BETA-SECOND-PAGE-CANARY-8842")
    c.drawString(
        72,
        h - 160,
        "Pages exist so multi-page extraction strategies can be validated.",
    )
    c.showPage()

    c.save()
    return PDF_PATH


def ensure_all() -> tuple[Path, Path]:
    return ensure_image(), ensure_pdf()


if __name__ == "__main__":
    img, pdf = ensure_all()
    print(f"image: {img}")
    print(f"pdf:   {pdf}")
