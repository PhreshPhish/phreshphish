from configparser import ConfigParser, ExtendedInterpolation
import argparse
import os, time

from rtapdata.log import RTAPDataLogger
from rtapdata.utils import init_feed
from rtapdata.crawler import SeleniumCrawler
from rtapdata.core.statemachine import ETagStateMachine

import json

logger = RTAPDataLogger(__file__)


def fetch_crawl_list(config, feed):
    feed = init_feed(config, feed)
    
    urls = []
    etag_states = {}
    logger.debug(f'Fetching crawl list for feed: {feed}')
    update_available, poll_stat = feed.is_update_available()
    if update_available:
        logger.debug(f'There are new URLs to crawl.')
        urls.extend(feed.get_urls())
        etag_states[feed.name] = feed.current_etag

    return urls, etag_states, poll_stat


def submit_to_crawler(crawler_uri, urls):
    crawler = SeleniumCrawler(uri=crawler_uri, save_path=config['general']['content-path'])
    return crawler.submit(urls)


def update_state(config, phish_states=None, etag_states=None):
    for feed in etag_states.keys():
        statemachine_path = config[feed]["statemachine_path"]
        state_machine = ETagStateMachine(state_root = statemachine_path)
        state_machine.update_etag(etag = etag_states[feed])
    for feed in phish_states.keys():
        statemachine_path = config[feed]["statemachine_path"]
        state_machine = ETagStateMachine(state_root = statemachine_path)
        state_machine.update_last_phish(phish_id = phish_states[feed])
    return

def main(config, feed):
    logger.info(f'Initiating data collection for feeds: {feed}...')
    urls, etag_states, poll_stat = fetch_crawl_list(config, feed)
    logger.info(f'  {len(urls)} new URLs to crawl.')

    crawled_pages = []
    nfail = 0
    if len(urls) > 0:
        if feed != "benign-feed":
            crawled_pages = list(filter(None, submit_to_crawler(
                config['crawler']['hub-uri'], urls
            )))
            nfail = len(urls) - len(crawled_pages)
        else:
            nbenigns = int(config["benign-feed"]["nbenigns"])
            if nbenigns < 0:
                crawled_pages = list(filter(None, submit_to_crawler(
                    config['crawler']['hub-uri'], urls
                )))
                nfail = len(urls) - len(crawled_pages)
            else:
                isucc = 0
                while len(crawled_pages) < nbenigns:
                    if len(urls) <= 0: break
                    nscrape = nbenigns - len(crawled_pages)
                    scrape_urls = urls[:nscrape]
                    urls = urls[nscrape:]
                    crawled_pages.extend(list(filter(None, submit_to_crawler(
                        config['crawler']['hub-uri'], scrape_urls
                    ))))
                    isucc = len(crawled_pages) - isucc
                    nfail += len(scrape_urls) - isucc
                    logger.info(
                        f'{isucc} urls succeeded in this iteration. ' +
                        f'total crawled: {len(crawled_pages)}'
                    )
        logger.info(f"{nfail} htmls failed. {len(crawled_pages)} collected successfully")
        logger.info(f'data collection complete.')
    else:
        logger.info('no data to collect in this iteration.')
    
    if len(crawled_pages) > 0:
        phish_states = {}
        
        for page in crawled_pages:
            if phish_states.get(page.feed_source, 0) != page.source_id:
                phish_states[page.feed_source] = page.source_id
        logger.info(f"ids to be updated for phish states: {phish_states}")

        update_state(config, phish_states, etag_states)
        logger.info(f"phish_states & etag_states updated.")
    return poll_stat




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fetches URLs from PhishTank and crawls.'
    )
    parser.add_argument(
        '--config', default = r"config/config.ini",
        help = 'Path to config file.'
    )
    parser.add_argument(
        '--feed', required=True,
        help='Which feed to fetch & crawl.'
    )

    args = parser.parse_args()

    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(args.config)

    if args.feed not in config.keys():
        print(f"{args.feed} not in config")
        print(f"available choices: {list(config.keys())}")
    else:
        # setup defaults
        os.environ['LOG_LEVEL'] = config['general']['log-level']

        # get new urls and scrape. if db request fails, try again after an hour
        poll_stat = main(config, args.feed)
        
        if args.feed != "benign-feed" and poll_stat != 200:
            logger.info(
                f"failed to fetch {args.feed}. " +
                f"poll stat: {poll_stat}. " +
                f"second attempt after 1 hour"
            )
            time.sleep(360)
            logger.info("second attempt...")
            poll_stat = main(config, args.feed)
            if poll_stat != 200:
                logger.info(f"failed to fetch in the second attempt")
        
        if args.feed == "benign-feed" and poll_stat != 404:
            os.remove(poll_stat)
