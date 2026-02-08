import argparse
from pathlib import Path
from datasets import disable_caching, Dataset
disable_caching()
import json
import dill
import Levenshtein
import time
from utils import load_filtered_sha256s, load_test_data, save_lsh, save_nn, get_predictions


def lenmatch(row, comprow, matching_ratios):
    row_urllen = len(row['url'])
    row_htmllen = len(row['html'])
    comprow_urllen = len(comprow['url'][0])
    comprow_htmllen = len(comprow['html'][0])
    url_lenlim = (row_urllen + comprow_urllen) * (1 - matching_ratios['url'])
    html_lenlim = (row_htmllen + comprow_htmllen) * (1 - matching_ratios['html'])
    url_lendiff = abs(row_urllen - comprow_urllen)
    html_lendiff = abs(row_htmllen - comprow_htmllen)

    return (
        url_lendiff <= url_lenlim,
        html_lendiff <= html_lenlim
    )

def compute_lev_dist(row, idx, ds, thresholds):
    # ret_val = {'lev_ratio': '', 'lev_sha256s': '', 'lev_feat_type': ''}
    distances = []
    for i in range(ds.num_rows-1, idx, -1):
        comprow = ds.select([i]).to_dict()
        compute_url, compute_html = lenmatch(row, comprow, thresholds)
        if compute_url:
            lev_ratio = Levenshtein.ratio(row['url'], comprow['url'][0])
            if lev_ratio >= thresholds['url']:
                # ret_val['lev_ratio'] += f'{lev_ratio},'
                # ret_val['lev_sha256s'] += f'{comprow["sha256"][0]},'
                # ret_val['lev_feat_type'] += 'url,'
                distances.append(f"url,{comprow['sha256'][0]},{lev_ratio}")
                # break

        if compute_html:
            lev_ratio = Levenshtein.ratio(row['html'], comprow['html'][0])
            if lev_ratio >= thresholds['html']:
                # ret_val['lev_ratio'] += f'{lev_ratio},'
                # ret_val['lev_sha256s'] += f'{comprow["sha256"][0]},'
                # ret_val['lev_feat_type'] += 'html,'
                distances.append(f"html,{comprow['sha256'][0]},{lev_ratio}")
                # break

    return {'distances': '|'.join(str(d) for d in distances)}
    # return ret_val

from sklearn.metrics.pairwise import pairwise_distances
import scipy.sparse as sp
import numpy as np

def compute_cos_dist(row, idx, ds):
    # Leave last row as there are no more rows to compare
    if idx == ds.num_rows - 1: return {'distances': '|'}

    query = row['fv']
    remaining = ds.skip(idx + 1)
    candidates = np.vstack([remaining['fv']])
    dists = pairwise_distances(candidates, query.reshape(1, -1), metric='cosine').flatten()

    distances = []
    for sha256, dist in zip(remaining.select_columns('sha256').to_list(), dists):
        distances.append(f"{sha256['sha256']},{dist}")
    
    return {'distances': '|'.join(str(d) for d in distances)}


def diversity_filter_using_lev_ratio(ds: Dataset, prediction_dir: Path, thresholds: dict, 
                                     results_dir: Path, class_type: str, num_workers: int =4):
    # Get GTE prediction scores to ensure hard samples are retained after diversity filtering
    predictions = get_predictions(prediction_dir, class_type, num_workers)
    print(f"  {len(predictions['sha256'])} GTE prediction scores for class type '{class_type}'")

    # Map score to dataset
    ds = ds.map(
        lambda x: {
            'score': predictions['score'][predictions['sha256'].index(x['sha256'])]
        }, 
        desc="Mapping GTE prediction scores to dataset"
    )

    # Sort dataset by score in descending order if phish else ascending order if benign so that the hard samples are at the bottom and retained
    print("  Sorting dataset by GTE prediction scores...")
    ds = ds.sort('score', reverse=(class_type == 'phish'))
    print(f" Sorted dataset by GTE prediction scores in {'descending' if class_type == 'phish' else 'ascending'} order")

    # Compute Levenshtein distances
    ds = ds.map(
        compute_lev_dist, with_indices=True, 
        # num_proc=num_workers,
        fn_kwargs={'ds': ds, 'thresholds': thresholds},
        desc="Computing Levenshtein distances for URL and HTML features"
    )
    # print("Computed Levenshtein distances for URL and HTML features")

    # Save levenshtein distance results for analysis
    ds = ds.select_columns(['sha256', 'lev_ratio', 'lev_sha256s', 'lev_feat_type'])
    ds.to_csv(str(results_dir / f"{class_type}_levenshtein_ratio_results.tsv"), index=False, sep='\t')

    return


def diversity_filter_using_cos_dist(prediction_dir: Path, filtered: list, tfidf_fv_dir: Path,
                                    results_dir: Path, class_type: str, num_workers: int =4):
    # Load GTE prediction scores to ensure hard samples are retained after diversity filtering
    print("  Loading test data after applying leakage filter...")
    ds = get_predictions(prediction_dir, class_type, num_workers, return_type="dataset")
    ds.set_format("numpy")
    print(f"  {len(ds['sha256'])} GTE prediction scores for class type '{class_type}'")

    # Filter out leakage filtered SHA256s
    ds = ds.filter(lambda x: x['sha256'] not in filtered, num_proc=num_workers, desc="Applying leakage filter to GTE predictions")
    print(f"  {len(ds['sha256'])} samples remaining after applying leakage filter for class type '{class_type}'")

    # ds = ds.skip(random.randint(0, ds.num_rows - 100)).take(100) # For testing purposes, limit samples
        
    # Sort dataset by score in descending order if phish else ascending order if benign so that the hard samples are at the bottom and retained
    print("  Sorting dataset by GTE prediction scores...")
    ds = ds.sort('score', reverse=(class_type == 'phish'))
    print(f" Sorted dataset by GTE prediction scores in {'descending' if class_type == 'phish' else 'ascending'} order")

    # Load TF-IDF feature vectors
    print("  Loading TF-IDF feature vectors...")
    for feat_type in ['url', 'html']:
        # Load feature vectors
        if class_type == "phish":
            fv_path = tfidf_fv_dir / f"{feat_type}_phishes.pkl"
        elif class_type == "benign":
            fv_path = tfidf_fv_dir / f"{feat_type}_benigns.pkl"
        with open(fv_path, "rb") as f:
            fv = dill.load(f)
        print(f"  Loaded TF-IDF feature vectors from {fv_path} for class type '{class_type}'")
        # Map feature vectors to dataset
        ds = ds.map(
            lambda x: {'fv': fv[x['sha256']].toarray().flatten()}, 
            desc=f"Mapping {feat_type} TF-IDF feature vectors to dataset"
        )
        # Compute cosine distances
        ds = ds.map(
            compute_cos_dist, with_indices=True, fn_kwargs={'ds': ds},
            num_proc=num_workers,
            desc=f"Computing cosine distances for {feat_type} feature"
        )
        # Save cosine distance results for analysis
        ds = ds.select_columns(['sha256', 'distances'])
        ds.to_csv(str(results_dir / f"{class_type}_cosine_distance_results_{feat_type}.tsv"), index=False, sep='\t')

    return 



def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Apply leakage, diversity & difficulty filters on test data")
    parser.add_argument("--input_test_data_dir", type=Path, required=True, help="Path to the input test data directory containing phishes & benigns")
    parser.add_argument("--class_type", type=str, choices=["phish", "benign"], required=True, help="Class type: phish or benign")
    parser.add_argument("--leakage_filtered_dir", type=Path, required=True, help="Path to the directory containing SHA256 hashes of leakage filtered data of phishes & benigns")
    parser.add_argument("--gte_prediction_scores_dir", type=Path, help="Path to the directory containing GTE prediction scores")
    parser.add_argument("--diversity_filter_metric", type=str, choices=["levenshtein", "cosine", "cosine-in-nearby-lsh-bins"], default="levenshtein", help="Diversity filter metric to use: levenshtein or cosine")
    parser.add_argument("--levenshtein_ratio_thresholds", type=json.loads, default={"url": 0.95, "html": 0.95}, help="Levenshtein ratio thresholds for URL and HTML distance filtering")
    parser.add_argument("--results_dir", type=Path, required=False, default="./data/diversity_filtering", help="Path to save the Levenshtein ratio or cosine distance results for analysis")
    parser.add_argument("--tfidf_fv_dir", type=Path, required=True, help="Path to the directory containing TF-IDF feature vectors for cosine distance computation")
    parser.add_argument("--lsh_feat_type", type=str, choices=["url", "html", "both"], default="url", help="Feature type to use for cosine-in-nearby-lsh-bins method")
    parser.add_argument("--lsh_n_vectors", type=json.loads, default={"url": 4, "html": 8}, help="Number of LSH vectors for each feature type")
    parser.add_argument("--lsh_dir", type=Path, required=False, default="/home/hgowda/data/mlteamshare/phreshphish2.0/splitting/lsh", help="Path to save/load LSH models for cosine-in-nearby-lsh-bins method")
    parser.add_argument("--max_search_radius", type=int, default=10, help="Maximum search radius for nearest neighbors in cosine-in-nearby-lsh-bins method")
    parser.add_argument("--nearest_neighbors_dir", type=Path, required=False, default="/home/hgowda/data/mlteamshare/phreshphish2.0/splitting/nn", help="Path to save/load nearest neighbors for cosine-in-nearby-lsh-bins method")
    parser.add_argument("--nearest_neighbors_after_diversity_filter", action="store_true", help="Whether to compute nearest neighbors after diversity filtering")
    parser.add_argument("--diversity_filtered_dir", type=Path, required=False, default="/home/hgowda/data/mlteamshare/phreshphish2.0/splitting/diversity_filtered", help="Path to the directory containing SHA256 hashes of diversity filtered data of phishes & benigns")
    parser.add_argument("--cosine_distance_threshold", type=float, default=0.2, help="Cosine distance threshold for diversity filtering")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for parallel processing")

    args = parser.parse_args()
    assert args.leakage_filtered_dir.exists(), f"Leakage filtered directory does not exist: {args.leakage_filtered_dir}"
    
    # Load leakage filtered SHA256 hashes
    print("Loading leakage filtered SHA256 hashes...")
    leakage_filtered_sha256 = load_filtered_sha256s(args.leakage_filtered_dir, args.class_type)
    print(f"Loaded {len(leakage_filtered_sha256)} SHA256 hashes for leakage filtering of class type '{args.class_type}'")
    print("-"*100)
    
    start_diversity_time = time.time()
    if args.diversity_filter_metric == "levenshtein":
        # Load test data applying the leakage filter
        assert args.input_test_data_dir.exists(), f"Input test data directory does not exist: {args.input_test_data_dir}"
        print("Loading test data after applying leakage filter...")
        ds = load_test_data(args.input_test_data_dir, args.class_type, leakage_filtered_sha256, args.num_workers)
        print(f"Loaded {len(ds)} test data samples after applying leakage filter for class type '{args.class_type}'")
        print("-"*100)

        # Collect levenshtein ratios
        print(f"Collecting levenshtein ratios above thresholds {args.levenshtein_ratio_thresholds}...")
        assert args.gte_prediction_scores_dir.exists(), f"GTE prediction scores directory does not exist: {args.gte_prediction_scores_dir}"
        args.results_dir.mkdir(parents=True, exist_ok=True)
        diversity_filter_using_lev_ratio(
            ds, args.gte_prediction_scores_dir, args.levenshtein_ratio_thresholds, 
            args.results_dir, args.class_type, args.num_workers
        )
    elif args.diversity_filter_metric == "cosine":
        # Collect cosine distances
        print(f"Collecting cosine distances...")
        assert args.gte_prediction_scores_dir.exists(), f"GTE prediction scores directory does not exist: {args.gte_prediction_scores_dir}"
        assert args.tfidf_fv_dir.exists(), f"TF-IDF feature vector directory does not exist: {args.tfidf_fv_dir}"
        args.results_dir.mkdir(parents=True, exist_ok=True)
        diversity_filter_using_cos_dist(
            args.gte_prediction_scores_dir, leakage_filtered_sha256, args.tfidf_fv_dir,
            args.results_dir, args.class_type, args.num_workers
        )
    elif args.diversity_filter_metric == "cosine-in-nearby-lsh-bins":  
        assert args.tfidf_fv_dir.exists(), f"TF-IDF feature vector directory does not exist: {args.tfidf_fv_dir}"
        args.lsh_dir.mkdir(parents=True, exist_ok=True)
        args.nearest_neighbors_dir.mkdir(parents=True, exist_ok=True)

        lsh_feat_types = ['url', 'html'] if args.lsh_feat_type == 'both' else [args.lsh_feat_type]
        for feat_type in lsh_feat_types:
            if args.class_type == "phish":
                fv_path = args.tfidf_fv_dir / f"{feat_type}_phishes.pkl"
            elif args.class_type == "benign":
                fv_path = args.tfidf_fv_dir / f"{feat_type}_benigns.pkl"
            assert fv_path.exists(), f"TF-IDF feature vector file does not exist: {fv_path}"
            n_vectors = args.lsh_n_vectors[feat_type]
            lsh_path = args.lsh_dir / f"test_{feat_type}_{args.class_type}_{n_vectors}.pkl"
            if not lsh_path.exists():
                print("Creating LSH index and bins...")
                save_lsh(args.class_type, feat_type, fv_path, n_vectors, lsh_path, leakage_filtered_sha256)
            else:
                print(f"LSH model already exists at {lsh_path}, skipping creation")
            if args.nearest_neighbors_after_diversity_filter:
                assert args.diversity_filtered_dir.exists(), f"Diversity filtered directory does not exist: {args.diversity_filtered_dir}"
                # Load diversity filtered SHA256 hashes
                print("Loading diversity filtered SHA256 hashes...")
                diversity_filtered_sha256_path = args.diversity_filtered_dir / f"{args.class_type}.pkl"
                with open(diversity_filtered_sha256_path, "rb") as f:
                    diversity_filtered_sha256 = dill.load(f)
                print(f"Loaded {len(diversity_filtered_sha256)} SHA256 hashes for diversity filtering of class type '{args.class_type}'")
                # Add diversity_filtered_sha256 to leakage_filtered_sha256 for nearest neighbors after diversity filtering
                leakage_filtered_sha256 = list(leakage_filtered_sha256)  # Convert to list if not already
                leakage_filtered_sha256.extend(diversity_filtered_sha256)
                nn_path = args.nearest_neighbors_dir / f"post_diversity_test_{feat_type}_{args.class_type}_{n_vectors}_{args.max_search_radius}.pkl"
            else:
                nn_path = args.nearest_neighbors_dir / f"diversity_test_{feat_type}_{args.class_type}_{n_vectors}_{args.max_search_radius}.pkl"
            
            # Collect cosine distances in nearby LSH bins
            print(f"Collecting cosine distances in nearby LSH bins...")
            save_nn(
                args.class_type, 
                feat_type, 
                fv_path, 
                fv_path, 
                lsh_path, 
                args.max_search_radius, 
                nn_path, 
                leakage_filtered_sha256, 
                prediction_dir=args.gte_prediction_scores_dir,
                cosine_distance_threshold=args.cosine_distance_threshold,
                num_workers=args.num_workers
            )
            print(f"Saved nearest neighbors at {nn_path}")

    
    end_diversity_time = time.time()
    print(f"Collecting metrics for diversity filtering took {end_diversity_time - start_diversity_time:.2f} seconds for class type '{args.class_type}'")
    print("-"*100)
    

if __name__ == "__main__":
    main()
