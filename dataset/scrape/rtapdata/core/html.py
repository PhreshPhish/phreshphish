import hashlib


class HtmlPage(object):
    def __init__(self, 
                 url, 
                 html_content=None, 
                 status=None, 
                 crawl_time=None, 
                 feed_source=None, 
                 target=None,
                 source_id=None,
                 ip=None,
                 asn=None,
                 submission_time=None):
        self.sha256 = hashlib.sha256(url.encode('utf-8')).hexdigest()
        self.url = url
        self.html_content = html_content
        self.status = status
        self.crawl_time = crawl_time
        self.feed_source = feed_source
        self.target = target
        self.source_id = source_id
        self.ip = ip
        self.asn = asn
        self.submission_time = submission_time

