BOT_NAME = "nas_speech"

SPIDER_MODULES = ["nas_speech.spiders"]
NEWSPIDER_MODULE = "nas_speech.spiders"

# ponytail: NAS's WAF answers 202 + empty body to browser-ish UAs (JS challenge)
# but serves plain HTML to curl. If this stops working, the fallback is Playwright.
USER_AGENT = "curl/8.0"
DEFAULT_REQUEST_HEADERS = {
    "From": "hyperfloo26@gmail.com",
}

ROBOTSTXT_OBEY = False

DOWNLOAD_DELAY = 1
CONCURRENT_REQUESTS_PER_DOMAIN = 1
RETRY_TIMES = 3

ITEM_PIPELINES = {
    "nas_speech.pipelines.DedupeSummaryPipeline": 300,
}

FEED_EXPORT_ENCODING = "utf-8"

# Explicit reactor, same as what `scrapy startproject` normally fills in.
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

LOG_LEVEL = "INFO"
