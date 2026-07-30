"""Stage 2: download every PDF listed in data/records.jsonl -> data/pdfs/<uid>/*.pdf

This spider doesn't crawl any web page itself — it just reads the local
records.jsonl and yields items with file_urls; Scrapy's FilesPipeline
(subclassed in pipelines.py as NasPdfDownloaderPipeline) takes care of
actually downloading each PDF, retrying on failure, and skipping files
that already exist on disk from a previous run.

Run from the project root (where scrapy.cfg lives):

    scrapy crawl nas_pdfs
"""
import json
from pathlib import Path

import scrapy

RECORDS = Path("data/records.jsonl")


class NasPdfsSpider(scrapy.Spider):
    name = "nas_pdfs"

    custom_settings = {
        "ITEM_PIPELINES": {
            "nas_speech.pipelines.NasPdfDownloaderPipeline": 1,
        },
        "FILES_STORE": "data/pdfs",
        "MEDIA_ALLOW_REDIRECTS": True,
        "USER_AGENT": "curl/8.0",
        "DEFAULT_REQUEST_HEADERS": {"From": "hyperfloo26@gmail.com"},
        "ROBOTSTXT_OBEY": False,
        "DOWNLOAD_DELAY": 1,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "RETRY_TIMES": 3,
        "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        "LOG_LEVEL": "INFO",
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
