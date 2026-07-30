"""Stage 3: extract text from every downloaded PDF using poppler's pdftotext.

Requires poppler-utils installed and `pdftotext` on PATH.
- Windows: download poppler for Windows, add its `Library\\bin` (or `bin`)
  folder to PATH, then open a NEW terminal so PATH changes apply.
- macOS: brew install poppler
- Linux: apt install poppler-utils

Run from the project root:
    python extract_text.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

PDF_DIR = Path("data/pdfs")
TEXT_DIR = Path("data/text")


def main():
    if shutil.which("pdftotext") is None:
        print("pdftotext not found on PATH. Install poppler and make sure its")
        print("bin folder is on PATH, then open a new terminal and try again.")
        sys.exit(1)

    if not PDF_DIR.exists():
        print(f"{PDF_DIR} not found — run `scrapy crawl nas_pdfs` first")
        sys.exit(1)

    pdfs = sorted(PDF_DIR.rglob("*.pdf"))
    print(f"found {len(pdfs)} pdfs")

    ok, failed = 0, []
    for i, pdf in enumerate(pdfs, 1):
        uid = pdf.parent.name
        out_dir = TEXT_DIR / uid
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (pdf.stem + ".txt")

        if out_path.exists():
            ok += 1
            continue

        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf), str(out_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            failed.append((pdf.name, result.stderr.strip()[:200]))
        else:
            ok += 1

        if i % 25 == 0 or i == len(pdfs):
            print(f"  {i}/{len(pdfs)} processed")

    print(f"\ndone: {ok} ok, {len(failed)} failed")
    for name, err in failed[:20]:
        print(f"  FAILED {name}: {err}")


if __name__ == "__main__":
    main()
