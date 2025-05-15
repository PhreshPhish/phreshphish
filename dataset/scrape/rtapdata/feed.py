import requests
import ijson
import json
import io
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from .core.statemachine import ETagStateMachine
from .core.feed_base import URLFeed
from .core import exceptions
from .log import RTAPDataLogger

from rtapdata import utils

import boto3
import gzip
import random
import shutil


class HttpResponseIo(io.IOBase):
    """Takes a streamed requests.Response object and makes the streaming response 'read'-able per Python IO requirements."""
    def __init__(self, response):
        self.chunks = response.iter_content(chunk_size=65536)

    def read(self, n):
        if n == 0:
            return b''
        return next(self.chunks, b'')


class PhishTankFeed(URLFeed):
    def __init__(self, uri, user_agent, statemachine_path, **kwargs):
        super(PhishTankFeed, self).__init__('phishtank-feed', 'feed')
        self.uri = uri
        self.request_headers = {'User-Agent': user_agent}
        self.state_machine = ETagStateMachine(state_root=statemachine_path)

    def format_urls(self, urls):
        formatted_urls = []
        for url in urls:
            ip = url.get('details', '')
            if len(ip) > 0:
                ip = ip[0].get('ip_address', '')
            formatted_urls.append({
                'source': self.name, 
                'url': url['url'], 
                'target': url['target'], 
                'submission_time': url['verification_time'],
                'source_id': int(url['phish_id']),
                'ip': ip
            })
        return formatted_urls

    def poll(self):
        """Performs an HTTP HEAD request to fetch the ETAG for the resource."""
        self.logger.debug('Polling for updates...')
        response = requests.head(self.uri, headers=self.request_headers, allow_redirects=True)
        return response.headers['ETag'], response.status_code

    def is_update_available(self):
        """Checks to see if the etag matches the latest that we have."""
        self.logger.debug('Checking to see if an update is available')
        self.last_etag = self.state_machine.get_etag()
        self.current_etag = None
        self.current_etag, poll_stat = self.poll()
        if poll_stat != 200:
            self.logger.error(f'Unable to poll for feed updates.')
            self.logger.debug(f'Last etag crawled: ...{self.last_etag[-4:]}')
            return False, poll_stat
        else:
            self.logger.debug(f'Last etag crawled: ...{self.last_etag[-4:]}')
            self.logger.debug(f'Current etag available: ...{self.current_etag[-4:]}')
            return (False if self.current_etag == self.last_etag else True), poll_stat


    def get_urls(self):
        """Streams the full database from the feed but stops processing the download
        once we have fetched everything we don't already have."""
        last_phish_id = self.state_machine.get_last_phish()
        self.logger.debug(f'Last phish-id crawled: {last_phish_id}')
        latest_urls = []
        with requests.get(self.uri, headers=self.request_headers, stream=True) as response:
            self.logger.debug(f'Streaming response code: {response.status_code}')
            all_urls = list(ijson.items(HttpResponseIo(response), prefix='item'))
            self.logger.debug(f'Total count of URLS in database: {len(all_urls)}')
            latest_urls = [obj for obj in all_urls if int(obj['phish_id']) > int(last_phish_id)]
            self.logger.debug(f'Found {len(latest_urls)} new URLs to crawl.')
            """
            etag = response.headers['ETag']
            last_phish = max([int(obj['phish_id']) for obj in latest_urls])
            self.state_machine.update_etag(etag)
            self.state_machine.update_last_phish(last_phish)
            """
        return self.format_urls(latest_urls)


class DummyPhishTankFeed(PhishTankFeed):
    """For testing the crawling logic since PhishTank has a rate limiter
    which prevents us from querying the PhishTankFeed too often.
    
    Mostly just overloads the `get_urls` function so that we can ensure there are always
    some updates."""
    def __init__(self, uri, user_agent, statemachine_path, **kwargs):
        super(DummyPhishTankFeed, self).__init__(uri, user_agent, statemachine_path, **kwargs)
        self.name = 'dummy-phishtank-feed'
        self.uri = uri
        self.request_headers = {'User-Agent': user_agent}
        self.state_machine = ETagStateMachine(state_root=statemachine_path)

    def is_update_available(self):
        return True, 200

    def get_urls(self):
        last_phish_id = self.state_machine.get_last_phish()
        self.logger.debug(f'Last phish-id crawled: {last_phish_id}')
        file = Path(self.uri)
        with open(file, 'r') as fp:
            latest_urls = [obj for obj in json.load(fp) if int(obj['phish_id']) > int(last_phish_id)]
            self.logger.debug(f'Found {len(latest_urls)} new URLs to crawl.')
        return self.format_urls(latest_urls)


class Top5kFeed(URLFeed):
    def __init__(self, path):
        super(Top5kFeed, self).__init__('top5k-feed', 'list')
        self.path = path

    def format_urls(self, urls):
        return [{'source': self.name, 'url': url['domain']} for url in urls]

    def get_urls(self):
        self.logger.debug('Getting URLs...')
        with open(self.path, 'r') as fp:
            urls = json.load(fp)['top_sites']
        return self.format_urls(urls)


class APWGFeed(URLFeed):
    def __init__(self, 
                 uri, 
                 endpoint, 
                 authkey, 
                 user_agent, 
                 poll_seconds,
                 pagewise_sleep_seconds, 
                 statemachine_path):
        super(APWGFeed, self).__init__('apwg-feed', 'feeds')
        self.uri = uri
        self.endpoint = endpoint
        self.request_headers = {'User-Agent': user_agent}
        self.sleep_sec = pagewise_sleep_seconds
        self.state_machine = ETagStateMachine(state_root = statemachine_path)
        key_kwargs = {"Authorization": authkey}
        keys = utils.fetch_env_keys(**key_kwargs)
        self.headers = {
            "Authorization": keys["Authorization"],
            "X-API-Token": keys["Authorization"],
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        self.poll_seconds = poll_seconds

    def format_urls(self, urls):
        return [
            {
                'source': self.name, 
                'url': url['url'], 
                'target': url['brand'], 
                'submission_time': datetime.fromtimestamp(
                    url['updatedAt']
                ).strftime('%Y-%m-%d %H:%M:%S'),
                'source_id': int(url['id']),
                'ip': url['ip'],
                'asn': url['asn']
            } 
            for url in urls
        ]
    
    def is_update_available(self):
        self.last_etag = None
        self.current_etag = None
        return True, 200

    def get_page_urls(self, last_phish_id):
        new_phishes = []

        url = self.uri + self.endpoint
        self.logger.debug(f"APWG: reading {url}")
        response = requests.get(url, headers = self.headers)
        self.logger.debug(f"APWG: response status code was {response.status_code}")
        self.endpoint = "end"

        if response.status_code == 200:
            self.resp = json.dumps(response.json())
            self.resp = json.loads(self.resp)
            n_phishes_page = self.resp['per_page']
            new_phishes = [
                x for x in self.resp["data"]
                if int(x["id"]) > int(last_phish_id)
            ]
            
            if len(new_phishes) < n_phishes_page:
                self.logger.debug("APWG: all new phishes collected")
            else:
                self.logger.debug("APWG: all new phishes in current page")
                if self.resp["next_page_url"] is not None:
                    self.logger.debug("APWG: reading next page")
                    self.endpoint = '/' + self.resp["next_page_url"].split('/')[-1]
                    # for testing limit the reading to 3 pages
                    # if self.endpoint == "/phish?page=3": self.endpoint = "end"
                else:
                    self.logger.debug("APWG: no more pages")
        
        return new_phishes

    def get_urls(self):
        last_phish_id = self.state_machine.get_last_phish()
        latest_urls = []
        self.logger.debug(f"APWG: last phish id was {last_phish_id}")

        while self.endpoint != "end":
            latest_urls.extend(self.get_page_urls(last_phish_id))
            # self.logger.debug(f"APWG: sleeping for {self.sleep_sec} to avoid throttling")
            # time.sleep(self.sleep_sec)
        
        self.logger.debug(f"APWG: {len(latest_urls)} new phishes collected")

        return self.format_urls(latest_urls)

class TargetBenignFeed(URLFeed):
    def __init__(self, 
                 benign_type, 
                 path, 
                 user_agent, 
                 statemachine_path, 
                 **kwargs):
        super(TargetBenignFeed, self).__init__(benign_type, 'feed')
        self.path = path
        self.state_machine = ETagStateMachine(
            state_root = statemachine_path
        )

    def format_urls(self, urls):
        return [
            {
                'source': self.name, 
                'url': url['url'], 
                'target': url['target'], 
                'source_id': int(url['phish_id'])
            } for url in urls
        ]
    
    def is_update_available(self):
        self.last_etag = None
        self.current_etag = None
        outfile = Path(self.path)
        return outfile.exists(), 200
    
    def get_urls(self):
        last_phish_id = self.state_machine.get_last_phish()
        self.logger.debug(f'Last benign-id crawled: {last_phish_id}')
        latest_urls = []
        outfile = Path(self.path)
        with open(outfile, 'r') as f:
            latest_urls = [
                obj for obj in json.load(f) 
                if int(obj["phish_id"]) > last_phish_id
            ]
        
        return self.format_urls(latest_urls)


class BenignFeed(URLFeed):
    def __init__(self, 
                 benign_type, 
                 bucket, 
                 creds,
                 tempdata_path,
                 nconnect_range,
                 nbenigns,
                 buffer_size,
                 min_html_size,
                 init_last_date,
                 statemachine_path):
        super(BenignFeed, self).__init__(benign_type, 'benign')
        self.bucket = bucket
        aws_access = creds.split(".")
        self.s3_client = boto3.client(
            "s3", 
            aws_access_key_id=aws_access[0], 
            aws_secret_access_key=aws_access[1]
        )
        self.tempdata_path = Path(tempdata_path)
        self.tempdata_path.mkdir(parents=True, exist_ok=True)
        self.nconnect_range = tuple([int(x) for x in nconnect_range.split(',')])
        self.nbenigns = int(1.2 * int(nbenigns))
        self.buffer_size = int(buffer_size)
        self.pgen = 0.75
        self.min_html_size = int(min_html_size)
        self.state_machine = ETagStateMachine(statemachine_path, init_last_date)
    
    def _get_filename(self):
        s3_file = None
        try:
            objects = self.s3_client.list_objects_v2(
                Bucket=self.bucket, Prefix=self.current_etag
            ).get("Contents", [])
        except Exception as e:
            self.logger.info(
                f"failed s3 list objects. bucket: {self.bucket} prefix: {self.current_etag}. " +
                f"exception: {e}"
            )
        else:
            s3_files = [x["Key"] for x in objects if x["Key"].endswith("gz")]
            self.logger.info(f"s3 files: {s3_files}")
            if len(s3_files) > 0:
                s3_file = s3_files[-1]
                self.logger.info(f"using {s3_file} of {s3_files}")
            else:
                self.logger.info(f"no files on {self.current_etag}")
        return s3_file
    
    def _download_gz(self, s3_file, gz):
        download_status = False
        try:           
            self.s3_client.download_file(self.bucket, s3_file, gz)
        except Exception as e:
            gz = None
            self.logger.info(
                f"failed s3 download. s3 file: {s3_file} gz file: {gz}. exception: {e}"
            )
        else:
            download_status = True
        return download_status, gz
    
    def is_update_available(self):
        update_available = False
        poll_stat = 404
        self.gz = None

        self.last_etag = self.state_machine.get_etag()
        self.current_etag = (
            datetime.strptime(self.last_etag, '%Y-%m-%d') + timedelta(days=1)
        ).strftime('%Y-%m-%d')

        print(f'last etag: {self.last_etag}')
        print(f'current etag: {self.current_etag}')
        
        s3_file = self._get_filename()

        if s3_file is not None:
            fname = s3_file.replace('/', '-')
            self.gz = self.tempdata_path / fname
            if self.gz.exists():
                update_available = True
                poll_stat = self.gz
                self.logger.info(f"{self.gz} already in {self.tempdata_path}")
            else:
                update_available, self.gz = self._download_gz(s3_file, self.gz)
                if update_available: poll_stat = self.gz
        return  update_available, poll_stat
    
    def _filter_urls(self, line):
        lnsplit = line.split(',')
        if len(lnsplit) >= 3:
            nblocks = int(lnsplit[-1])
            nconnects = int(lnsplit[-2])
            origurl = ''.join(lnsplit[:-2])
            if nblocks == 0:
                if self.nconnect_range[0] <= nconnects <= self.nconnect_range[1]:
                    return origurl

    def _url_gen(self):
        incomplete = ''
        with gzip.open(self.gz, 'rb') as f:
            while True:
                data = f.read(self.buffer_size)
                if not data: break
                try:
                    buffer = io.StringIO(data.decode('utf-8'))
                except Exception as e:
                    self.logger.info(f'line failed with {e}')
                else:
                    line = incomplete + buffer.readline()
                    while line:
                        if line[-1] == '\n':
                            yield line
                            incomplete = ''
                        else:
                            incomplete = line
                        line = buffer.readline()
                    # if buffer.tell() < self.buffer_size: break

    def get_urls(self):
        latest_urls = []
        if self.gz is not None:
            try:
                i = 0
                for line in self._url_gen():
                    if i >= self.nbenigns: break
                    url = self._filter_urls(line)
                    if url:
                        random.seed(1)
                        if random.random() <= self.pgen:
                            latest_urls.append({
                                "url": url,
                                "source": self.name,
                                "source_id": self.current_etag + f"-{str(i)}",
                                "submission_time": self.current_etag,
                                "min_html_size": self.min_html_size
                            })
                            i += 1
            except Exception as e:
                self.logger.info(f"failed reading gz file: {self.gz}. exception: {e}")
        return latest_urls
