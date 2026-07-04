"""One-time setup: download the Unicode fonts needed for PDF translation output.

Thai and Japanese glyphs aren't covered by PDF base-14 fonts, so pdf_replacer
embeds these into translated PDFs. Run once after cloning:

    python scripts/fetch_fonts.py
"""

import urllib.request
from pathlib import Path

FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"

# OFL-licensed, from the Google Fonts repository.
FONT_URLS = {
    "NotoSansThai-Regular.ttf": (
        "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansthai/"
        "NotoSansThai%5Bwdth%2Cwght%5D.ttf"
    ),
    "NotoSansJP-Regular.ttf": (
        "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"
    ),
}


def main() -> None:
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in FONT_URLS.items():
        dest = FONTS_DIR / filename
        if dest.exists():
            print(f"skip (already present): {dest}")
            continue
        print(f"downloading {filename} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"saved: {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
