"""Download the PDFs listed in data/records.jsonl -> data/pdfs/<uid>/*.pdf

Reads the local file and hands file_urls to Scrapy's FilesPipeline.

    scrapy crawl nas_pdfs
"""
import json
from pathlib import Path

import scrapy

RECORDS = Path("data/records.jsonl")


class NasPdfsSpider(scrapy.Spider):
    name = "nas_pdfs"

    # everything else is inherited from settings.py
    custom_settings = {
        "ITEM_PIPELINES": {
            "nas_speech.pipelines.NasPdfDownloaderPipeline": 1,
        },
        "FILES_STORE": "data/pdfs",
        "MEDIA_ALLOW_REDIRECTS": True,
    }

    async def start(self):
        if not RECORDS.exists():
            self.logger.error(f"{RECORDS} not found — run the nas_speeches spider first")
            return

        n_items, n_pdfs = 0, 0
        with RECORDS.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("pdf"):
                    n_items += 1
                    n_pdfs += len(row["pdf"])
                    yield {"uid": row["uid"], "file_urls": row["pdf"]}
        self.logger.info(f"queued {n_pdfs} pdf(s) across {n_items} record(s)")
