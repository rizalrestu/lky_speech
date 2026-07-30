import os
from urllib.parse import urlparse

import scrapy
from scrapy.exceptions import DropItem
from scrapy.pipelines.files import FilesPipeline


class DedupeSummaryPipeline:
    """Drops duplicate uids (in case pagination overlaps) and prints the same
    summary the original stdlib script did. Actual JSONL writing is handled
    by Scrapy's feed exporter via the -O command-line switch, not here."""

    def open_spider(self, spider):
        self.seen = set()
        self.rows = []

    def process_item(self, item, spider):
        if item["uid"] in self.seen:
            raise DropItem(f"duplicate uid {item['uid']}")
        self.seen.add(item["uid"])
        self.rows.append(item)
        return item

    def close_spider(self, spider):
        rows = self.rows
        n_htm = sum(1 for r in rows if r["htm"])
        n_pdf = sum(1 for r in rows if r["pdf"] and not r["htm"])
        print(f"\nscraped {len(rows)} items (expected {getattr(spider, 'total', '?')})")
        print(f"has .htm: {n_htm} | pdf-only: {n_pdf} | "
              f"no attachment: {len(rows) - n_htm - n_pdf}")


class NasPdfDownloaderPipeline(FilesPipeline):
    """Downloads PDFs listed in item['file_urls'] and stores them as
    data/pdfs/<uid>/<original-filename>.pdf instead of the default
    hash-based filenames, so files stay traceable back to records.jsonl."""

    def get_media_requests(self, item, info):
        for url in item.get("file_urls", []):
            yield scrapy.Request(url, meta={"uid": item["uid"]})

    def file_path(self, request, response=None, info=None, *, item=None):
        uid = request.meta.get("uid", "unknown")
        name = os.path.basename(urlparse(request.url).path) or "file.pdf"
        return f"{uid}/{name}"
