"""Stage 1: crawl NAS speeches search-result pages -> data/records.jsonl

Pagination is plain server-side: ?page-num=N, 20 items/page.
Run from the project root (where scrapy.cfg lives):

    scrapy crawl nas_speeches -O data/records.jsonl
    scrapy crawl nas_speeches -a last_page=3 -O data/records.jsonl   # testing
"""
import re
from html import unescape as _unescape

import scrapy

BASE = ("https://www.nas.gov.sg/archivesonline/speeches/search-result"
        "?search-type=advanced&thesaurus=Off&speaker=Lee+Kuan+Yew&page-num=")
R_TOTAL = re.compile(r"([\d,]+)\s+items")


def clean(parts):
    """Join text nodes, unescape entities, collapse whitespace."""
    text = " ".join(p.strip() for p in parts if p and p.strip())
    text = _unescape(text)
    return re.sub(r"\s+", " ", text).strip()


class NasSpeechesSpider(scrapy.Spider):
    name = "nas_speeches"

    def __init__(self, last_page=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_page = int(last_page) if last_page else None
        self.total = None
        self.pages = None

    async def start(self):
        yield scrapy.Request(BASE + "1", callback=self.parse, meta={"page": 1})

    def parse(self, response):
        page = response.meta["page"]

        if page == 1:
            m = R_TOTAL.search(response.text)
            self.total = int(m.group(1).replace(",", "")) if m else 0
            self.pages = self.last_page or -(-self.total // 20)
            self.logger.info(f"{self.total} items -> {self.pages} pages")

        items = list(self.parse_items(response))
        self.logger.info(f"page {page}: {len(items)} parsed")
        yield from items

        if not items:
            self.logger.info("empty page, stopping")
            return

        if page < self.pages:
            yield scrapy.Request(
                BASE + str(page + 1),
                callback=self.parse,
                meta={"page": page + 1},
            )

    def parse_items(self, response):
        for block in response.css("div.searchResultItem"):
            uid = block.css('input[name="selectedItems"]::attr(value)').get()
            title_parts = block.css("a.linkRow ::text").getall()
            if not (uid and title_parts):
                continue

            hrefs = block.css("a::attr(href)").getall()
            atts = ["https:" + h for h in hrefs if "/pdfdoc/" in h]

            yield {
                "uid": uid,
                "title": clean(title_parts),
                "speaker": clean(block.css("span.speaker ::text").getall()) or None,
                "date": clean(block.css("span.date ::text").getall()) or None,
                "source": clean(block.css("p.source ::text").getall()) or None,
                "record_url": f"https://www.nas.gov.sg/archivesonline/speeches/record-details/{uid}",
                "htm": [a for a in atts if a.lower().endswith(".htm")],
                "pdf": [a for a in atts if a.lower().endswith(".pdf")],
            }
