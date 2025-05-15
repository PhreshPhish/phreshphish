from ..log import RTAPDataLogger


class URLFeed(object):
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.logger = RTAPDataLogger(self.name)