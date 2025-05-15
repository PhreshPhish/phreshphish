import time
import os
import pandas as pd
import dill
from source.splitutils import (
    create_single, load_json, vectorize,
    train_lsh, get_nearest_neighbors, get_distances, write_jsons
)
import logging
from tqdm import tqdm


def save_singles(dir, class_type, n, seed, single_input):
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    logger.debug(f"Creating single {class_type} json...")
    create_single(dir, n, seed, single_input)
    end_time = time.time()
    logger.debug(f"Created single json for {class_type} at {single_input} in {end_time - start_time : .4f} secs")


def save_vectors(df, class_type, feat_type, tfidf_path):
    logger = logging.getLogger(__name__)

    start_time = time.time()
    logger.debug(f"  For {class_type} {feat_type} vectorizing started...")
    _tfidf = vectorize(feat_type, df[feat_type])
    end_time = time.time()
    logger.debug(f"  For {class_type} {feat_type} vectorizing time: {end_time - start_time : .4f} secs")
    
    with open(tfidf_path, 'wb') as f:
        dill.dump(_tfidf, f)
    del _tfidf
    logger.debug(f"Writing tfidf features complete!")

def save_lsh(class_type, feat_type, tfidf_path, train_ratio, n_vectors, seed, lsh_path):
    logger = logging.getLogger(__name__)

    logger.debug(f"  For {class_type} {feat_type} loading vectors started...")
    with open(tfidf_path, 'rb') as f:
        _tfidf = dill.load(f)
    
    train_size = int(_tfidf.shape[0] * train_ratio)
    train_tfidf = _tfidf[:train_size, :]
    logger.debug(f"  For {class_type} {feat_type} train size: {train_size}")

    start_time = time.time()
    _model = train_lsh(train_tfidf, n_vectors, seed)
    end_time = time.time()
    logger.debug(f"  {class_type} {feat_type} LSH binning time: {end_time - start_time : .4f} secs")

    with open(lsh_path, 'wb') as f:
        dill.dump(_model, f)
    del _model
    logger.debug(f"Writing the LSH model details complete!")

def save_nn(class_type, feat_type, tfidf_path, train_ratio, lsh_path, max_search_radius, nn_path):
    logger = logging.getLogger(__name__)

    with open(tfidf_path, 'rb') as f:
        _tfidf = dill.load(f)
    
    train_size = int(_tfidf.shape[0] * train_ratio)
    train_tfidf = _tfidf[:train_size, :]
    test_tfidf = _tfidf[train_size:, :]

    with open(lsh_path, 'rb') as f:
        _model = dill.load(f)

    start_time = time.time()
    _test_near_neighbors = []
    logger.debug(f"Searching nearest neighbors of {test_tfidf.shape[0]} test datapoints")
    i = 0
    last_time = start_time
    for query_vector in tqdm(test_tfidf):
        _test_near_neighbors.append(
            get_nearest_neighbors(train_tfidf, query_vector, _model, max_search_radius)
        )
        i += 1
        if i % 10 == 0: 
            elapsed_time = time.time() - start_time
            logger.debug(f"  complete searching for {i} in {elapsed_time:.4f}")


    end_time = time.time()
    logger.debug(f"Searching {class_type} {feat_type} nearby bins time: {end_time - start_time : .4f} secs")

    _test_min_dist = {
        i: min(_test_near_neighbors[i].items(), key=lambda x: x[1])
        for i in range(len(_test_near_neighbors))
    }

    with open(nn_path, 'wb') as f:
        dill.dump(_test_min_dist, f)
    del _test_near_neighbors, _test_min_dist

    logger.debug(f"Writing nearest neighbors file complete!")

def save_splits(df, class_type, train_ratio, nn_paths, dist_thresholds, splits):
    logger = logging.getLogger(__name__)

    n_total = len(df)
    train_size = int(n_total * train_ratio)
    train_df = df.iloc[:train_size, :]
    test_df = df.iloc[train_size:, :]
    test_size = len(test_df)
    logger.debug(f"{class_type} data split into train: {train_size} and test: {test_size}")
    
    logger.debug(f"Fetching nearest {class_type} train neighbors of test datapoints")
    nearby_df = get_distances(nn_paths, class_type)
    url_thresh = dist_thresholds['url'][class_type]
    html_thresh = dist_thresholds['html'][class_type]
    filter_df = nearby_df.loc[
        (nearby_df[f'distance_url'] > url_thresh) & 
        (nearby_df[f'distance_html'] > html_thresh)
    ]
    test_df = test_df.iloc[filter_df.idx, :]
    logger.debug(
        f"Based on thresholds, url: {url_thresh} & html: {html_thresh}, " +
        f"{class_type} test filtered down from {test_size} to {len(test_df)}"
    )

    train_dir = splits['train'][class_type]
    i = write_jsons(train_df, train_dir)
    logger.debug(f"{i}/{len(train_df)} trains written into {train_dir}")
    test_dir = splits['test'][class_type]
    i = write_jsons(test_df, test_dir)
    logger.debug(f"{i}/{len(test_df)} tests written into {test_dir}")
    



def main(files, seed, train_ratio):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename='train_test_split.log', 
        encoding='utf-8', level=logging.DEBUG
    )

    if files['create_single_input']:
        print(f"Creating single input...")
        # phishes
        class_type = 'phishes'
        dir = files['json_dir'][class_type]
        n = files['n']
        single_input = files['single_input'][class_type]
        save_singles(dir, class_type, n, seed, single_input)

        # benigns
        class_type = 'benigns'
        dir = files['json_dir'][class_type]
        n = files['n']
        single_input = files['single_input'][class_type]
        save_singles(dir, class_type, n, seed, single_input)
    
    if files['create_fv_files']:
        print(f"Creating tfidf feature vector files...")
        # phishes
        class_type = 'phishes'
        filename = files['single_input'][class_type]
        df, zero_day = load_json(filename)
        print(f"  For {class_type}, date used to replace the missing dates: {zero_day}")        

        feat_type = 'url'
        tfidf_path = files[f'{feat_type}_fv'][class_type]
        save_vectors(df, class_type, feat_type, tfidf_path)

        feat_type = 'html'
        tfidf_path = files[f'{feat_type}_fv'][class_type]
        save_vectors(df, class_type, feat_type, tfidf_path)

        # benigns
        class_type = 'benigns'
        filename = files['single_input'][class_type]
        df, zero_day = load_json(filename)
        print(f"  For {class_type}, date used to replace the missing dates: {zero_day}")

        feat_type = 'url'
        tfidf_path = files[f'{feat_type}_fv'][class_type]
        save_vectors(df, class_type, feat_type, tfidf_path)

        feat_type = 'html'
        tfidf_path = files[f'{feat_type}_fv'][class_type]
        save_vectors(df, class_type, feat_type, tfidf_path)
    
    if files['create_lsh_models']:
        print(f"Creating LSH models...")
        # phishes
        class_type = 'phishes'
        
        feat_type = 'url'
        tfidf_path = files[f'{feat_type}_fv'][class_type]
        n_vectors = files['n_vectors'][feat_type]
        lsh_path = files[f'{feat_type}_lsh'][class_type]
        save_lsh(class_type, feat_type, tfidf_path, train_ratio, n_vectors, seed, lsh_path)

        feat_type = 'html'
        tfidf_path = files[f'{feat_type}_fv'][class_type]
        n_vectors = files['n_vectors'][feat_type]
        lsh_path = files[f'{feat_type}_lsh'][class_type]
        save_lsh(class_type, feat_type, tfidf_path, train_ratio, n_vectors, seed, lsh_path)
        
        # benigns
        class_type = 'benigns'

        feat_type = 'url'
        tfidf_path = files[f'{feat_type}_fv'][class_type]
        n_vectors = files['n_vectors'][feat_type]
        lsh_path = files[f'{feat_type}_lsh'][class_type]
        save_lsh(class_type, feat_type, tfidf_path, train_ratio, n_vectors, seed, lsh_path)
        
        feat_type = 'html'
        tfidf_path = files[f'{feat_type}_fv'][class_type]
        n_vectors = files['n_vectors'][feat_type]
        lsh_path = files[f'{feat_type}_lsh'][class_type]
        save_lsh(class_type, feat_type, tfidf_path, train_ratio, n_vectors, seed, lsh_path)

    if files['create_nearest_neighbors']:
        print(f"Creating nearest neighbors of test...")
        # phishes
        class_type = 'phishes'

        feat_type = 'url'
        tfidf_path = files[f'{feat_type}_fv'][class_type]
        lsh_path = files[f'{feat_type}_lsh'][class_type]
        max_search_radius = files['max_search_radius']
        nn_path = files[f'{feat_type}_nn'][class_type]
        save_nn(class_type, feat_type, tfidf_path, train_ratio, lsh_path, max_search_radius, nn_path)

        feat_type = 'html'
        tfidf_path = files[f'{feat_type}_fv'][class_type]
        lsh_path = files[f'{feat_type}_lsh'][class_type]
        max_search_radius = files['max_search_radius']
        nn_path = files[f'{feat_type}_nn'][class_type]
        save_nn(class_type, feat_type, tfidf_path, train_ratio, lsh_path, max_search_radius, nn_path)

        # benigns
        class_type = 'benigns'

        feat_type = 'url'
        tfidf_path = files[f'{feat_type}_fv'][class_type]
        lsh_path = files[f'{feat_type}_lsh'][class_type]
        max_search_radius = files['max_search_radius']
        nn_path = files[f'{feat_type}_nn'][class_type]
        save_nn(class_type, feat_type, tfidf_path, train_ratio, lsh_path, max_search_radius, nn_path)

        feat_type = 'html'
        tfidf_path = files[f'{feat_type}_fv'][class_type]
        lsh_path = files[f'{feat_type}_lsh'][class_type]
        max_search_radius = files['max_search_radius']
        nn_path = files[f'{feat_type}_nn'][class_type]
        save_nn(class_type, feat_type, tfidf_path, train_ratio, lsh_path, max_search_radius, nn_path)

    if files['split_train_test']:
        print(f"Creating train and test datasets...")
        nn_paths = {
            'url': files['url_nn'], 'html': files['html_nn']
        }
        dist_thresholds = files['dist_thresholds']
        splits = files['splits']
        
        # phishes
        class_type = 'phishes'
        filename = files['single_input'][class_type]
        df, zero_day = load_json(filename)
        print(f"  For {class_type}, date used to replace the missing dates: {zero_day}")
        save_splits(df, class_type, train_ratio, nn_paths, dist_thresholds, splits)

        # benigns
        class_type = 'benigns'
        filename = files['single_input'][class_type]
        df, zero_day = load_json(filename)
        print(f"  For {class_type}, date used to replace the missing dates: {zero_day}")
        save_splits(df, class_type, train_ratio, nn_paths, dist_thresholds, splits)

        




