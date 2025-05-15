from .feed import (
    PhishTankFeed, 
    DummyPhishTankFeed, 
    APWGFeed, 
    Top5kFeed,
    TargetBenignFeed,
    BenignFeed,
    NetcraftFeed
)

feed_dict = {'phishtank-feed': PhishTankFeed,
             'dummy-phishtank-feed': DummyPhishTankFeed,
             'apwg-feed': APWGFeed,
             'top5k-feed': Top5kFeed,
             'target-benign-feed': TargetBenignFeed,
             'benign-feed': BenignFeed,
             'netcraft-feed': NetcraftFeed
            }


def init_feed(config, feed):
    return feed_dict[feed](**config[feed])

import argparse
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path


def fetch_config(path):
    try:
        rc = None
        config = ConfigParser(interpolation = ExtendedInterpolation())
        config.read(Path(path))
    except Exception as e:
        rc = e
        config = None
    finally:
        return config, rc


import os
from dotenv import find_dotenv, load_dotenv

def fetch_env_keys(**key_kwargs):
    load_dotenv(find_dotenv())
    for key, key_val in key_kwargs.items():
        key_kwargs[key] = os.environ[key_val]
    return key_kwargs
