from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime
import os
import requests
import json
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.utils import find_connectable_ip

from pathlib import Path

from .core.html import HtmlPage
from .log import RTAPDataLogger
import urllib


logger = RTAPDataLogger(__file__)


def crawl_url(url, hub_uri, save_path):
    logger.debug(f'Crawling {url["url"]}')
    crawler = PageCrawl(hub_uri, save_path)
    print(f'crawler set up')
    print(f'crawler driver: {crawler.driver}')
    page = None
    if crawler.driver is not None:
        page = crawler.crawl(url)        
        if page is None:
            logger.debug(f'Crawling for url {url} failed.')
        else:
            logger.debug(f'Crawling for url {url["url"]} complete. Saving page to disk.')
            crawler.save_page(page)
        crawler.driver.quit()
    return page


class PageCrawl(object):
    def __init__(self, driver_uri, save_path, headless=True):
        ch_options = webdriver.ChromeOptions()
        ch_options.add_argument('--disable-gpu')
        if headless: ch_options.add_argument('--headless')
        ch_options.add_argument("--start-maximized")
        # ch_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        # ch_options.add_experimental_option('useAutomationExtension', False)
        # ch_options.add_argument("--disable-blink-features=AutomationControlled")

        ff_options = webdriver.FirefoxOptions()
        ff_options.add_argument('--disable-gpu')
        if headless: ff_options.add_argument('--headless')
        
        # # browser_options.add_argument('--load-extension=extension/extension')
        options = ff_options
        try:
            self.driver = webdriver.Remote(command_executor=driver_uri, options=options)
        except Exception as e:
            print(f'coudnt set up driver: {e}')
            self.driver = None
        else:
            self.driver.set_page_load_timeout(10)
            self.driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            self.save_path = Path(save_path)

    def crawl(self, url):
        status_code = 0
        html_content = None
        time_elapsed = -1
        try:
            start = time.time()
            urlurl = url["url"] if url["url"].startswith("http") else "http://" + url["url"]
            self.driver.get(urlurl)
            time.sleep(2)
            end = time.time()
            time_elapsed = end - start
            html_content = self.driver.page_source
            # html_content = "never gonna give you up, never gonna let you down"
            ip = url.get("ip", "")
        except WebDriverException as exp:
            status_code = -1
            if 'net::ERR_NAME_NOT_RESOLVED' in str(exp):
                # raise RTAPCrawlerException('DNS error: unable to resolve domain.')
                print(f'DNS error: unable to resolve domain. URL: {url["url"]}')
            if 'net::ERR_CONNECTION_TIMED_OUT' in str(exp):
                print(f'Connection timed out for URL {url["url"]}')
            else:
                print(f'Unknown error occured while processing {url["url"]}: {exp}')
        else:
            try:
                if len(ip) == 0:
                    urlurl = url["url"] if url["url"].startswith("http") else "http://" + url["url"]
                    urlsplit = urllib.parse.urlsplit(urlurl)._asdict()
                    host = urlsplit["netloc"]
                    scheme = urlsplit["scheme"]
                    port = 443 if scheme == "https" else 80
                    url["ip"] = find_connectable_ip(host, port)
            except Exception as e:
                url["ip"] = ""
                print(f"failed at find_connectable_ip with {e}")
        finally:
            self.driver.quit()

        if html_content is None:
            status_code = -1
        elif len(html_content) < int(url.get("min_html_size", 5000)):
            status_code = -1
        else:
            status_code = 200

        page = None
        if status_code != -1:
            try:
                page = HtmlPage(
                    url=url['url'], 
                    html_content=html_content, 
                    status=status_code, 
                    crawl_time=time_elapsed, 
                    feed_source=url['source'], 
                    target=url.get('target', None),
                    source_id=url['source_id'],
                    ip=url.get('ip', None),
                    asn=url.get('asn', None),
                    submission_time=url.get(
                        "submission_time", 
                        datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S")
                    )
                )
            except:
                status_code = -1

        return page

    def save_page(self, page):        
        path = self.save_path / page.feed_source
        path.mkdir(parents = True, exist_ok = True)
        if page.feed_source == "benign-feed":
            fn = path / (page.source_id + '-' + page.sha256 + ".json")
        else:
            flnm = page.sha256 + ".json"
            if page.feed_source == "benign-feed":
                flnm = page.submission_time + '-' + flnm
            fn = path / flnm
        with open(fn, 'w') as fp:
            fp.write(json.dumps(page.__dict__, sort_keys=True, indent=4))


class SeleniumCrawler(object):
    def __init__(self, uri, save_path):
        self.uri = uri
        self.save_path = save_path

    def submit(self, urls):
        results = []
        for url in urls:
            res = crawl_url(url, self.uri, self.save_path)
            time.sleep(5)
            if res is not None:
                results.append(res)


        return results
