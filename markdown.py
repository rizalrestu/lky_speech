"""Wrap data/text into Markdown with YAML frontmatter from records.jsonl."""
import json
import re
from pathlib import Path

RECORDS = Path("data/records.jsonl")
TEXT_DIR = Path("data/text")
MD_DIR = Path("data/markdown")


def load_records():
    by_uid = {}
    with RECORDS.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_uid[row["uid"]] = row
    return by_uid


def yaml_escape(value):
    """Render a value as a double-quoted YAML scalar."""
    if value is None:
        return '""'
    value = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{value}"'


def clean_body(text):
    # pdftotext leaves runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def frontmatter(row, source_file):
    fields = {
        "uid": row["uid"],
        "title": row["title"],
        "speaker": row.get("speaker"),
        "date": row.get("date"),
        "source": row.get("source"),
        "record_url": row["record_url"],
        "source_file": source_file,
    }
    lines = ["---"]
    lines += [f"{k}: {yaml_escape(v)}" for k, v in fields.items()]
    lines.append("---\n")
    return "\n".join(lines)


def main():
    if not RECORDS.exists():
        print(f"{RECORDS} not found — run the nas_speeches spider first")
        return
    if not TEXT_DIR.exists():
        print(f"{TEXT_DIR} not found — run extract.py first")
        return

    by_uid = load_records()
    txt_files = sorted(TEXT_DIR.rglob("*.txt"))
    print(f"found {len(txt_files)} text files")

    n_ok, n_missing_meta = 0, 0
    for txt_path in txt_files:
        uid = txt_path.parent.name
        row = by_uid.get(uid)
        if row is None:
            n_missing_meta += 1
            continue

        body = clean_body(txt_path.read_text(encoding="utf-8", errors="ignore"))
        md = frontmatter(row, txt_path.name) + f"\n# {row['title']}\n\n{body}\n"

        out_dir = MD_DIR / uid
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / (txt_path.stem + ".md")).write_text(md, encoding="utf-8")
        n_ok += 1

    print(f"\nwrote {n_ok} markdown files -> {MD_DIR}")
    if n_missing_meta:
        print(f"skipped {n_missing_meta} file(s) with no matching uid in records.jsonl")


if __name__ == "__main__":
    main()
