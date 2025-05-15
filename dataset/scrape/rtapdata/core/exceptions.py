

class RTAPCrawlerException(Exception):
    def __init__(self, message, *args):
        self.message = message
        super(RTAPCrawlerException, self).__init__(message, *args) 

class FeedPollException(RTAPCrawlerException):
    def __init__(self, message, *args):
        super(FeedPollException, self).__init__(message, *args)