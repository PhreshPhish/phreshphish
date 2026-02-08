import json
from tqdm import tqdm

def load_sample(filename):
    try:
        """Loads a sample from a JSON file"""
        with open(filename, 'r') as fp:
            sample = json.load(fp)
        return sample
    except Exception as e:
        fn = os.path.dirname(filename)
        print(f"Error loading file {fn}. Error: {e}")
        return fn

import os
import glob
import random
import multiprocessing as mp
import time
from tqdm import tqdm

def create_single(dir, n, seed, single_input, num_workers=72):

    try:
        path = os.path.join(dir, '*.json')
        data_files = glob.glob(path)
        data_files.sort()
        if (n > 0) & (n <= len(data_files)):
            random.seed(seed)
            data_files = random.sample(data_files, n)

        start_time = time.time()
        with mp.Pool(num_workers) as pool:
            data = pool.map(
                load_sample, 
                tqdm(data_files, desc=f"Loading samples of {os.path.basename(dir)} ({num_workers} workers):")
            )#, chunksize=100)
        print(f"    complete in {time.time() - start_time : .4f} seconds")
        
        with open(single_input, 'w') as f:
            json.dump(data, f)
        print(f"    json dumped in {single_input}")
        rc = 0
    except Exception as e:
        rc = -1
        print(f"  Error creating single json: {e}")

    return rc

import pandas as pd

def load_json(filename):
    """ 
    - Load
    - Change all the targets to lower case
    - Sort by collect_date
    """
    df = pd.read_json(filename)
    df.target = df.target.str.lower()
    df = df.sort_values(['date', 'sha256']).reset_index(drop=True)
    return df

def load_df(dir, num_workers=72):
    path = os.path.join(dir, '*.json')
    data_files = glob.glob(path)
    with mp.Pool(num_workers) as pool:
        data = pool.map(
            load_sample, 
            tqdm(data_files, desc=f"Loading samples of {os.path.basename(dir)} ({num_workers} workers):"), 
            chunksize=10
        )
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], format='ISO8601')
    df.target = df.target.str.lower()
    df = df.sort_values(['date', 'sha256']).reset_index(drop=True)

    return df

import matplotlib.pyplot as plt
import seaborn as sns

def time_analysis(df):
    sns.set_theme()

    plt.figure(figsize=(10,5))

    sns.histplot(df['date'])
    plt.xticks(rotation=90)
    plt.show()


from sklearn.feature_extraction.text import TfidfVectorizer
import urllib

def train_tfidf(elem_type, elements):

    if elem_type == 'url':
        elements = [x.lower() for x in elements]
        analyzer = 'char'
        stop_words = None
        ngram_range = (1, 3)
        max_features = None
    else:
        analyzer = 'word'
        stop_words = 'english'
        ngram_range = (1, 3)
        max_features = 1_000_000

    tfidf = TfidfVectorizer(
        analyzer=analyzer, ngram_range=ngram_range, 
        min_df=0.0, stop_words=stop_words, max_features=max_features
    )
    tfidf.fit(elements)
    return tfidf

def vectorize(elem_type, elements):
    elements = map(str, elements)

    if elem_type == 'url':
        elements = [x.lower() for x in elements]
        analyzer = 'char'
        stop_words = None
        ngram_range = (1, 3)
        max_features = None
    else:
        analyzer = 'word'
        stop_words = 'english'
        ngram_range = (1, 3)
        max_features = 1_000_000

    tfidf = TfidfVectorizer(
        analyzer=analyzer, ngram_range=ngram_range, 
        min_df=0.0, stop_words=stop_words, max_features=max_features
    )
    return tfidf.fit_transform(elements)


def split_by_time(df, train_ratio=0.8):
    """
    - df is already sorted by collect_date
    - Split by train_ratio
    """
    train_size = int(len(df) * train_ratio)
    return (df.iloc[:train_size,], df.iloc[train_size:,])


import dill
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_distances(nn_paths, class_type):
    url_path = nn_paths['url'][class_type]
    with open(url_path, 'rb') as f:
        test_min_dist = dill.load(f)
    df_url = pd.DataFrame(
        [(k, v[0], v[1]) for k, v in test_min_dist.items()],
        columns=['idx', 'nearest_bin_url', 'distance_url']
    )

    html_path = nn_paths['html'][class_type]
    with open(html_path, 'rb') as f:
        test_min_dist = dill.load(f)
    df_html = pd.DataFrame(
        [(k, v[0], v[1]) for k, v in test_min_dist.items()],
        columns=['idx', 'nearest_bin_html', 'distance_html']
    )

    return df_url.merge(df_html)

def plot_dist_dist(df, suptitle):
    sns.set_theme()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), tight_layout=True)
    fig.suptitle(suptitle)

    sns.histplot(df.distance_url, ax=axes[0])
    axes[0].set_title(f"url")
    axes[0].set_xlim(0.0, 1.0)
    # axes[0].set_ylim(0, 2500)
    sns.histplot(df.distance_html, ax=axes[1])
    axes[1].set_title(f"html")
    axes[1].set_xlim(0.0, 1.0)
    # axes[1].set_ylim(0, 16000)

    plt.show()

def show_distributions(class_type, n_total, train_ratio, nn_paths, dist_thresholds):
    sns.set_theme()

    df = get_distances(nn_paths, class_type)

    n_train = int(n_total * train_ratio)
    n_test = len(df)
    suptitle = f"Nearest train distance distribution of {n_test} test data before filtering. Train: {n_train}"
    plot_dist_dist(df, suptitle)

    # filter_by = 'url'
    # thresh = dist_thresholds[filter_by][class_type]
    # filter_df = df.loc[df[f'distance_{filter_by}'] > thresh]
    # n = len(filter_df)
    # n_filtered = n_test - n
    # min_index = filter_df.idx.min()
    # suptitle = (
    #     f"After filtering data with {filter_by} at {thresh} cosine distance of train, leaving {n} test data." + 
    #     f"Train: {n_train}.\nTrain + filtered test before first unfiltered: {n_train + min_index}"
    # )
    # suptitle = f"After filtering data with {filter_by} at {thresh} cosine distance of train, leaving {n} test data. Train: {n_train}. Train + filtered test: {n_train + n_filtered}"
    # plot_dist_dist(filter_df, suptitle)

    # filter_by = 'html'
    # thresh = dist_thresholds[filter_by][class_type]
    # filter_df = df.loc[df[f'distance_{filter_by}'] > thresh]
    # n = len(filter_df)
    # n_filtered = n_test - n
    # min_index = filter_df.idx.min()
    # suptitle = (
    #     f"After filtering data with {filter_by} at {thresh} cosine distance of train, leaving {n} test data." + 
    #     f"Train: {n_train}.\nTrain + filtered test before first unfiltered: {n_train + min_index}"
    # )
    # suptitle = f"After filtering data with {filter_by} at {thresh} cosine distance of train, leaving {n} test data. Train: {n_train}. Train + filtered test: {n_train + n_filtered}"
    # plot_dist_dist(filter_df, suptitle)


    filter_by = ('url', '&', 'html')
    thresh = (dist_thresholds['url'][class_type], dist_thresholds['html'][class_type])
    filter_df = df.loc[(df[f'distance_{filter_by[0]}'] > thresh[0]) & (df[f'distance_{filter_by[-1]}'] > thresh[-1])]
    n = len(filter_df)
    n_filtered = n_test - n
    min_index = filter_df.idx.min()
    suptitle = (
        f"After filtering data with {filter_by} at {thresh} cosine distance of train, leaving {n} test data." + 
        f"Train: {n_train}.\nTrain + filtered test before first unfiltered: {n_train + min_index}"
    )
    plot_dist_dist(filter_df, suptitle)

    # filter_by = ('url', 'or', 'html')
    # thresh = (dist_thresholds['url'][class_type], dist_thresholds['html'][class_type])
    # filter_df = df.loc[(df[f'distance_{filter_by[0]}'] > thresh[0]) | (df[f'distance_{filter_by[-1]}'] > thresh[-1])]
    # n = len(filter_df)
    # n_filtered = n_test - n
    # min_index = filter_df.idx.min()
    # suptitle = (
    #     f"After filtering data with {filter_by} at {thresh} cosine distance of train, leaving {n} test data." + 
    #     f"Train: {n_train}.\nTrain + filtered test before first unfiltered: {n_train + min_index}"
    # )
    # suptitle = f"After filtering data with {filter_by} at {thresh} cosine distance of train, leaving {n} test data. Train: {n_train}. Train + filtered test: {n_train + n_filtered}"
    # plot_dist_dist(filter_df, suptitle)

    return df

from tqdm import tqdm
def write_jsons(df, dir):
    """
    """
    i = 0
    for row in tqdm(df.iterrows()):
        data = row[1].to_dict()
        filename = os.path.join(dir, data['sha256']+'.json')
        data['date'] = str(data['date'])
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            i += 1
        except Exception as e:
            print(f"Couldn't write {filename} because of {e}")
    return i


def get_fnames(dir):
    fnames = [
        os.path.join(dir, x) for x in os.listdir(dir)
        if x.endswith('.json')
    ]    
    return fnames

from bs4 import BeautifulSoup
from ftlangdetect import detect

def get_lang(filename):
    try:
        with open(filename) as f:
            data = json.load(f)
            if type(data['html']) == str:
                soup = BeautifulSoup(data['html'], 'html.parser')
                text = soup.text
            else:
                text = ''
        data['text'] = text
        data['lang'] = 'no text in html'
        data['lang_score'] = -1
        text = text.replace('\n', ' ')
        if text:
            resp = detect(text=text, low_memory=False)
            data['lang'] = resp['lang']
            data['lang_score'] = resp['score']

        with open(filename, 'w') as f:
            json.dump(data, f)
        
        ret = 0
    except Exception as e:
        print(f"Text couldn't be extracted from {filename}")
        ret = -1
    
    return ret

def read_json(params):
    filename, fields = params
    with open(filename) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k in fields}

import shutil
def move_json(params):
    try:
        src, dst = params
        shutil.move(src, dst)
        return 0
    except:
        return -1

import logging

def save_singles(dir, class_type, n, seed, single_input):
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    logger.debug(f"Creating single {class_type} json...")
    create_single(dir, n, seed, single_input)
    end_time = time.time()
    logger.debug(f"Created single json for {class_type} at {single_input} in {end_time - start_time : .4f} secs")

from bs4 import BeautifulSoup

def save_tfidf_model(df, class_type, feat_type, tfidf_path):
    start_time = time.time()
    print(f"  For {class_type} {feat_type} tfidf model training started...")
    if feat_type == 'html':
        # elements = df['html'].map(str).map(lambda x: BeautifulSoup(x, 'html.parser').text if x else ' ')
        elements = df['html'].map(str)
    elif feat_type == 'url':
        elements = df['url'].map(str)
    _tfidf = train_tfidf(feat_type, elements)
    end_time = time.time()
    print(f"  For {class_type} {feat_type} tfidf model training time: {end_time - start_time : .4f} secs")
    
    with open(tfidf_path, 'wb') as f:
        dill.dump(_tfidf, f)
    del _tfidf
    print(f"Writing tfidf model complete!")


def save_vectors(df, class_type, feat_type, tfidf_path, fv_path):
    start_time = time.time()
    print(f"  For {class_type} {feat_type} vectorizing started...")
    with open(tfidf_path, 'rb') as f:
        _tfidf = dill.load(f)
    _vectors = _tfidf.transform(
        df[feat_type].map(str) 
        if feat_type == 'url' 
        else df[feat_type].map(str).map(lambda x: BeautifulSoup(x, 'html.parser').text if x else ' ')
    )
    sha256s = df['sha256'].tolist()
    _vectors = {sha256s[i]: _vectors[i, :] for i in range(_vectors.shape[0])}
    end_time = time.time()
    print(f"  For {class_type} {feat_type} vectorizing time: {end_time - start_time : .4f} secs")
    
    with open(fv_path, 'wb') as f:
        dill.dump(_vectors, f)
    del _vectors
    print(f"Writing feature vectors complete!")



from collections import defaultdict
import numpy as np

def train_lsh(feature_vectors, n_vectors, seed=None):
    if seed is not None: np.random.seed(seed)

    sha256s, vectors = [], []
    for sha256, vector in feature_vectors.items():
        sha256s.append(sha256)
        vectors.append(vector)
    X_tfidf = sp.vstack(vectors)

    dim = X_tfidf.shape[1]
    random_vectors = np.random.randn(dim, n_vectors)

    # partition data points into bins,
    # and encode bin index bits into integers
    bin_indices_bits = X_tfidf.dot(random_vectors) >= 0
    powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)
    bin_indices = bin_indices_bits.dot(powers_of_two)

    # update table so that table[i] is the list of document ids with bin index equal to i
    table = defaultdict(list)
    for sha256, bin_index in zip(sha256s, bin_indices):
        table[bin_index].append(sha256)

    # note that we're storing the bin_indices here
    # so we can do some ad-hoc checking with it,
    # this isn't actually required
    model = {'table': table,
             'random_vectors': random_vectors,
             'bin_indices': bin_indices,
             'bin_indices_bits': bin_indices_bits}
    return model

import scipy.sparse as sp

def save_lsh(class_type, feat_type, fv_path, n_vectors, lsh_path, filtered=None):

    print(f"  For {class_type} {feat_type} loading vectors started...")
    with open(fv_path, 'rb') as f:
        feature_vectors = dill.load(f)

    if filtered is not None:
        print(f"  For {class_type} {feat_type} train size before filtering: {len(feature_vectors)}")
        feature_vectors = {k: v for k, v in feature_vectors.items() if k not in filtered}
        print(f"  For {class_type} {feat_type} train size after filtering: {len(feature_vectors)}")

    print(f"  For {class_type} {feat_type} train size: {len(feature_vectors)}")

    start_time = time.time()
    _model = train_lsh(feature_vectors, n_vectors, seed=42)
    end_time = time.time()
    print(f"  {class_type} {feat_type} LSH binning time: {end_time - start_time : .4f} secs")

    with open(lsh_path, 'wb') as f:
        dill.dump(_model, f)
    del _model
    print(f"Writing the LSH model details complete!")



from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

def search_nearby_bins(query_bin_bits, table, search_radius=3, candidate_set=None):
    """
    For a given query vector and trained LSH model's table
    return all candidate neighbors with the specified search radius.
    
    Example
    -------
    model = train_lsh(X_tfidf, n_vectors=16, seed=143)
    query = model['bin_index_bits'][0]  # vector for the first document
    candidates = search_nearby_bins(query, model['table'])
    """
    if candidate_set is None:
        candidate_set = set()

    n_vectors = query_bin_bits.shape[0]
    powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)

    for different_bits in combinations(range(n_vectors), search_radius):
        # flip the bits (n_1, n_2, ..., n_r) of the query bin to produce a new bit vector
        index = list(different_bits)
        alternate_bits = query_bin_bits.copy()
        alternate_bits[index] = np.logical_not(alternate_bits[index])

        # convert the new bit vector to an integer index
        nearby_bin = alternate_bits.dot(powers_of_two)

        # fetch the list of documents belonging to
        # the bin indexed by the new bit vector,
        # then add those documents to candidate_set;
        # make sure that the bin exists in the table
        if nearby_bin in table:
            candidate_set.update(table[nearby_bin])

    return candidate_set

from sklearn.metrics.pairwise import pairwise_distances

# Global variables for multiprocessing workers
_worker_train_fv = None
_worker_model = None
_worker_max_search_radius = None

def _init_worker(train_fv_path, lsh_path, max_search_radius, filtered=set()):
    """Initializer function for multiprocessing workers to load data once per process"""
    global _worker_train_fv, _worker_model, _worker_max_search_radius
    with open(train_fv_path, 'rb') as f:
        _worker_train_fv = dill.load(f)
    if filtered is not None:
        _worker_train_fv = {k: v for k, v in _worker_train_fv.items() if k not in filtered}
    with open(lsh_path, 'rb') as f:
        _worker_model = dill.load(f)
    # filtering out candidates that may have been filtered under diversity filter 
    # while using the code to genereate neighbors for post filter analysis
    _worker_model['table'] = {k: [x for x in lst if x not in filtered] for k, lst in _worker_model['table'].items()}
    _worker_max_search_radius = max_search_radius

def get_nearest_neighbors(query_item):
    """Multiprocessing version that uses global worker variables"""
    query_sha256, query_vector = query_item
    table = _worker_model['table']
    random_vectors = _worker_model['random_vectors']

    # compute bin index for the query vector, in bit representation.
    bin_index_bits = np.ravel(query_vector.dot(random_vectors) >= 0)

    # search nearby bins and collect candidates
    candidate_set = set()
    for search_radius in range(_worker_max_search_radius + 1):
        candidate_set = search_nearby_bins(bin_index_bits, table, search_radius, candidate_set)

    # sort candidates by their true distances from the query
    candidate_list = list(candidate_set)
    if len(candidate_list) > 0:
        candidates = sp.vstack([_worker_train_fv[candi] for candi in candidate_list])
        distance = pairwise_distances(candidates, query_vector, metric='cosine').flatten()

        nearest_neighbors = {
            candi: dist for candi, dist in zip(candidate_list, distance)
        }

        return (query_sha256, nearest_neighbors)
    else:
        return (query_sha256, {None: float('inf')})

from datasets import load_from_disk
from pathlib import Path

def get_predictions(prediction_dir: Path, class_type: str, num_workers: int =4, return_type: str ="dict"):
    predictions = load_from_disk(prediction_dir)
    print(f"  Loaded {len(predictions)} GTE prediction scores from {prediction_dir}")
    label = 1 if class_type == 'phish' else 0
    predictions = predictions.filter(
        lambda x: x['label'] == label, num_proc=num_workers, 
        desc=f"Filtering predictions for label {label} ({class_type})"
    )
    print(f"{len(predictions)} samples with label {label} ({class_type})")
    predictions = predictions.select_columns(['sha256', 'label', 'score'])
    if return_type == "dict":
        return predictions.to_dict()
    elif return_type == "dataset":
        return predictions


def save_nn(class_type, feat_type, train_fv_path, test_fv_path, lsh_path, max_search_radius, 
            nn_path, filtered=None, prediction_dir=None, cosine_distance_threshold=0.2, num_workers=4):
    """Multiprocessing version of save_nn using worker initializer pattern"""
    with open(test_fv_path, 'rb') as f:
        test_feature_vectors = dill.load(f)
    if filtered is not None:
        print(f"  For {class_type} {feat_type} test size before filtering: {len(test_feature_vectors)}")
        test_feature_vectors = {k: v for k, v in test_feature_vectors.items() if k not in filtered}
        print(f"  For {class_type} {feat_type} test size after filtering: {len(test_feature_vectors)}")
    else:
        filtered = set()

    start_time = time.time()
    print(f"Searching nearest neighbors of {len(test_feature_vectors)} test datapoints using multiprocessing")

    # # Select 10000 test_feature_vectors for testing
    # test_feature_vectors = dict(list(test_feature_vectors.items())[:10000])
    # print(f"  For {class_type} {feat_type} testing size for nearest neighbor search: {len(test_feature_vectors)}")
    
    # Create pool with initializer that loads data once per worker
    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(train_fv_path, lsh_path, max_search_radius, filtered)
    ) as pool:
        # Process all test items
        results = pool.map(
            get_nearest_neighbors,
            tqdm(
                test_feature_vectors.items(), 
                desc=f"  Searching {class_type} {feat_type} nearest neighbors ({num_workers} workers):"
            ),
            chunksize=10
        )
    
    # Convert results list to dictionary
    _test_near_neighbors = {sha256: neighbors for sha256, neighbors in results}
    
    end_time = time.time()
    print(f"Searching {class_type} {feat_type} nearby bins time: {end_time - start_time : .4f} secs")

    if prediction_dir is None:
        if cosine_distance_threshold is None:
            # No filtering, just take all nearest neighbors
            _test_min_dist = {
                sha256: [(k, v) for k, v in _test_near_neighbors[sha256].items() if k != sha256]
                for sha256 in _test_near_neighbors.keys()
            }
        else:
            # Apply cosine distance threshold filtering
            print(f"  Applying cosine distance threshold of {cosine_distance_threshold} for nearest neighbors")
            _test_min_dist = {
                sha256: [(k, v) for k, v in _test_near_neighbors[sha256].items() if k != sha256 and v <= cosine_distance_threshold]
                for sha256 in _test_near_neighbors.keys()
            }
    else:
        # Load GTE prediction scores to ensure hard samples are retained after diversity filtering
        print("  Loading test data after applying leakage filter...")
        predictions = get_predictions(prediction_dir, class_type, num_workers, return_type="dataset")
        predictions.set_format("numpy")
        print(f"  {len(predictions['sha256'])} GTE prediction scores for class type '{class_type}'")

        # Filter out leakage filtered SHA256s
        predictions = predictions.filter(
            lambda x: x['sha256'] not in filtered, num_proc=num_workers,
            desc="Applying leakage filter to GTE predictions"
        )
        print(f"  {len(predictions['sha256'])} samples remaining after applying leakage filter for class type '{class_type}'")
        predictions = predictions.to_dict()

        _test_min_dist = {
            sha256: (
                min([(k, v) for k, v in _test_near_neighbors[sha256].items() if k != sha256], key=lambda x: x[1]), 
                predictions['score'][predictions['sha256'].index(sha256)]
            )
            for sha256 in _test_near_neighbors.keys()
        }

    with open(nn_path, 'wb') as f:
        dill.dump(_test_min_dist, f)
    del _test_near_neighbors, _test_min_dist

    print(f"Writing nearest neighbors file complete!")

def save_splits(df, class_type, train_ratio, nn_paths, dist_thresholds, splits):
    print(f"Splitting data for class type '{class_type}' with train ratio {train_ratio}")

    n_total = len(df)
    train_size = int(n_total * train_ratio)
    train_df = df.iloc[:train_size, :]
    test_df = df.iloc[train_size:, :]
    test_size = len(test_df)
    print(f"{class_type} data split into train: {train_size} and test: {test_size}")
    
    print(f"Fetching nearest {class_type} train neighbors of test datapoints")
    nearby_df = get_distances(nn_paths, class_type)
    url_thresh = dist_thresholds['url'][class_type]
    html_thresh = dist_thresholds['html'][class_type]
    filter_df = nearby_df.loc[
        (nearby_df[f'distance_url'] > url_thresh) & 
        (nearby_df[f'distance_html'] > html_thresh)
    ]
    test_df = test_df.iloc[filter_df.idx, :]
    print(
        f"Based on thresholds, url: {url_thresh} & html: {html_thresh}, " +
        f"{class_type} test filtered down from {test_size} to {len(test_df)}"
    )

    train_dir = splits['train'][class_type]
    os.makedirs(train_dir, exist_ok=True)
    i = write_jsons(train_df, train_dir)
    print(f"{i}/{len(train_df)} trains written into {train_dir}")
    test_dir = splits['test'][class_type]
    os.makedirs(test_dir, exist_ok=True)
    i = write_jsons(test_df, test_dir)
    print(f"{i}/{len(test_df)} tests written into {test_dir}")


def load_filtered_sha256s(dir: Path, class_type: str):
    path = dir / f"{class_type}.pkl"
    if not path.exists():
        return set()
    else:
        with open(path, "rb") as f:
            sha256s = dill.load(f)
        return sha256s

def load_file(file_path: Path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

from datasets import Dataset
def load_test_data(dir: Path, class_type: str, filtered: list, num_workers: int = 4):
    class_type = 'phishing' if class_type == 'phish' else class_type
    dir = dir / class_type
    filenames = list(dir.glob("**/*.json"))
    print(f"  Total {len(filenames)} files found in {dir} for class type '{class_type}'")
    filenames = [f for f in filenames if f.stem not in filtered]
    print(f"  Total {len(filenames)} files remaining after applying leakage filter for class type '{class_type}'")
    
    # filenames = random.sample(filenames, 1000) # For testing purposes, limit samples
    
    with mp.Pool(num_workers) as pool:
        data = list(tqdm(pool.imap(load_file, filenames), total=len(filenames), desc=f"Loading test data ({num_workers} workers)"))
    return Dataset.from_list(data)