import json

def load_sample(filename):
    try:
        """Loads a sample from a JSON file"""
        with open(filename, 'r') as fp:
            sample = json.load(fp)
        match = next((key for key in sample.keys() if 'content' in key), None)

        sha256 = filename.split('/')[-1].split('.json')[0]
        url = sample['url']
        html = sample[match] if match else ''
        target = sample.get('target', None)
        ip = sample.get('ip', [])
        if type(ip) == str: ip = [ip]
        collect_date = sample.get('submission_time', None)
        collect_date = collect_date[:10] if collect_date is not None else collect_date

        return {
            'sha256': sha256, 'url': url, 'html': html, 
            'target': target, 'ip': ip, 'collect_date': collect_date
        }
    except Exception as e:
        fn = filename.split('/')[-1]
        print(f"Error loading file {fn}. Error: {e}")
        return fn

import os
import glob
import random
import multiprocessing as mp
import time

def create_single(dir, n, seed, single_input):
    nprocs = int(os.cpu_count() * 0.8)

    try:
        path = os.path.join(dir, '*.json')
        data_files = glob.glob(path)
        if (n > 0) & (n <= len(data_files)):
            random.seed(seed)
            data_files = random.sample(data_files, n)

        start_time = time.time()
        with mp.Pool(nprocs) as pool:
            data = pool.map(load_sample, data_files, chunksize=100)
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
from datetime import timedelta

def load_json(filename):
    """ 
    - Load
    - Replace missing collect_date with date of a day before min date
    - Change all the targets to lower case
    - Sort by collect_date
    """
    df = pd.read_json(filename)
    df.collect_date = pd.to_datetime(df.collect_date)
    zero_day = df.collect_date.min() - timedelta(days=1)
    df.loc[df.collect_date.isna(), 'collect_date'] = zero_day
    df.target = df.target.str.lower()
    df = df.sort_values(['collect_date']).reset_index(drop=True)
    return df, zero_day

def load_df(dir):
    nprocs = 60
    path = os.path.join(dir, '*.json')
    data_files = glob.glob(path)
    with mp.Pool(nprocs) as pool:
        data = pool.map(load_sample, data_files, chunksize=100)
    df = pd.DataFrame(data)
    df.collect_date = pd.to_datetime(df.collect_date)
    zero_day = df.collect_date.min() - timedelta(days=1)
    df.loc[df.collect_date.isna(), 'collect_date'] = zero_day
    df.target = df.target.str.lower()
    df = df.sort_values(['collect_date']).reset_index(drop=True)

    return df, zero_day

import matplotlib.pyplot as plt
import seaborn as sns

def time_analysis(df, zero_day, class_type, exclude_zero_day=False):
    sns.set_theme()
    n_before_day_zero = len(df[df.collect_date == zero_day])
    r_before_day_zero = n_before_day_zero / len(df)

    print(f"{class_type} history starts on: {zero_day}")
    print(
        f"Number of prehistoric {class_type}: {n_before_day_zero},", 
        f"which is {r_before_day_zero : .2f} of all {class_type}"
    )
    plt.figure(figsize=(10,5))
    x = (
        df[df.collect_date > zero_day]['collect_date'] 
        if exclude_zero_day else df.collect_date
    )
    sns.histplot(x)
    plt.xticks(rotation=90)
    plt.show()


from sklearn.feature_extraction.text import TfidfVectorizer

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


from collections import defaultdict
import numpy as np

def train_lsh(X_tfidf, n_vectors, seed=None):
    if seed is not None: np.random.seed(seed)

    dim = X_tfidf.shape[1]
    random_vectors = np.random.randn(dim, n_vectors)

    # partition data points into bins,
    # and encode bin index bits into integers
    bin_indices_bits = X_tfidf.dot(random_vectors) >= 0
    powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)
    bin_indices = bin_indices_bits.dot(powers_of_two)

    # update table so that table[i] is the list of document ids with bin index equal to i
    table = defaultdict(list)
    for idx, bin_index in enumerate(bin_indices):
        table[bin_index].append(idx)

    # note that we're storing the bin_indices here
    # so we can do some ad-hoc checking with it,
    # this isn't actually required
    model = {'table': table,
             'random_vectors': random_vectors,
             'bin_indices': bin_indices,
             'bin_indices_bits': bin_indices_bits}
    return model


def split_by_time(df, train_ratio=0.8):
    """
    - df is already sorted by collect_date
    - Split by train_ratio
    """
    train_size = int(len(df) * train_ratio)
    return (df.iloc[:train_size,], df.iloc[train_size:,])

from itertools import combinations

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

def get_nearest_neighbors(X_tfidf, query_vector, model, max_search_radius=3):
    table = model['table']
    random_vectors = model['random_vectors']

    # compute bin index for the query vector, in bit representation.
    bin_index_bits = np.ravel(query_vector.dot(random_vectors) >= 0)

    # search nearby bins and collect candidates
    candidate_set = set()
    for search_radius in range(max_search_radius + 1):
        candidate_set = search_nearby_bins(bin_index_bits, table, search_radius, candidate_set)

    # sort candidates by their true distances from the query
    candidate_list = list(candidate_set)
    candidates = X_tfidf[candidate_list]
    distance = pairwise_distances(candidates, query_vector, metric='cosine').flatten()

    nearest_neighbors = {
        candi: dist for candi, dist in zip(candidate_list, distance)
    }

    return nearest_neighbors

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
    sns.histplot(df.distance_html, ax=axes[1])
    axes[1].set_title(f"html")
    axes[1].set_xlim(0.0, 1.0)

    plt.show()

def show_distributions(class_type, n_total, train_ratio, nn_paths, dist_thresholds):
    sns.set_theme()

    df = get_distances(nn_paths, class_type)

    n_train = int(n_total * train_ratio)
    n_test = len(df)
    suptitle = f"Nearest train distance distribution of {n_test} test data before filtering. Train: {n_train}"
    plot_dist_dist(df, suptitle)

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

    return df

from tqdm import tqdm
def write_jsons(df, dir):
    """
    """
    i = 0
    for row in tqdm(df.iterrows()):
        data = row[1].to_dict()
        filename = os.path.join(dir, data['sha256']+'.json')
        data['collect_date'] = str(data['collect_date'])
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            i += 1
        except Exception as e:
            print(f"Couldn't write {filename} because of {e}")
    return i
