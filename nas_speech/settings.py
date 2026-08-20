BOT_NAME = "nas_speech"

SPIDER_MODULES = ["nas_speech.spiders"]
NEWSPIDER_MODULE = "nas_speech.spiders"

# NAS's WAF has answered 202 + empty body to anything but a curl UA. Spoofing it
# back is circumventing an access control, so ask NAS for access instead.
USER_AGENT = "nas-speech-research/0.1 (+mailto:hyperfloo26@gmail.com)"
DEFAULT_REQUEST_HEADERS = {
    "From": "hyperfloo26@gmail.com",
}

ROBOTSTXT_OBEY = True

DOWNLOAD_DELAY = 1
CONCURRENT_REQUESTS_PER_DOMAIN = 1
RETRY_TIMES = 3

AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0

ITEM_PIPELINES = {
    "nas_speech.pipelines.DedupeSummaryPipeline": 300,
}

FEED_EXPORT_ENCODING = "utf-8"

TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

LOG_LEVEL = "INFO"
