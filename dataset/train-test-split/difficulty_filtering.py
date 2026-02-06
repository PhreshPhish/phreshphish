import argparse
from pathlib import Path
import dill
from utils import load_filtered_sha256s, get_predictions

def filter_by_difficulty(df, class_type, delta=0.15, phishfrac=0.1, benignfrac=0):
    
    assert delta > phishfrac and delta > benignfrac, "delta must be greater than frac"
    # easy phish points
    if class_type == "phish":
        thresh = df.score.quantile(1-delta)
        candidates = df.loc[df.score >= thresh, 'sha256']

        # Randomly sample rows to drop
        n_drop = int(phishfrac * len(df))
        to_drop = candidates.sample(n = n_drop, replace = False)
    # easy benign points
    elif class_type == "benign":
        thresh = df.score.quantile(delta)
        candidates = df.loc[df.score <= thresh, 'sha256']

        # Randomly sample rows to drop
        n_drop = int(benignfrac * len(df))
        to_drop = candidates.sample(n = n_drop, replace = False)
        
    return to_drop.tolist()


def main():
    parser = argparse.ArgumentParser(description="Filter items based on difficulty level.")
    parser.add_argument("--input_test_data_dir", type=Path, required=True, help="Path to the input test data directory containing phishes & benigns")
    parser.add_argument("--class_type", type=str, choices=["phish", "benign"], required=True, help="Class type: phish or benign")
    parser.add_argument("--leakage_filtered_dir", type=Path, required=True, help="Path to the directory containing SHA256 hashes of leakage filtered data of phishes & benigns")
    parser.add_argument("--diversity_filtered_dir", type=Path, required=True, help="Path to the directory containing SHA256 hashes of diversity filtered data of phishes & benigns")
    parser.add_argument("--gte_prediction_scores_dir", type=Path, required=True, help="Path to the directory containing GTE prediction scores")
    parser.add_argument("--svm_prediction_scores_dir", type=Path, required=True, help="Path to the directory containing SVM prediction scores")
    parser.add_argument("--dnn_prediction_scores_dir", type=Path, required=True, help="Path to the directory containing DNN prediction scores")
    parser.add_argument("--phishfrac", type=float, default=0.1, help="Fraction of easy phish samples to drop")
    parser.add_argument("--benignfrac", type=float, default=0.0, help="Fraction of easy benign samples to drop")
    parser.add_argument("--difficulty_filtered_dir", type=Path, required=True, help="Path to the directory to save difficulty filtered SHA256 hashes")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for parallel processing")
    args = parser.parse_args()

    assert args.input_test_data_dir.exists(), f"Input test data directory does not exist: {args.input_test_data_dir}"
    assert args.leakage_filtered_dir.exists(), f"Leakage filtered directory does not exist: {args.leakage_filtered_dir}"
    assert args.diversity_filtered_dir.exists(), f"Diversity filtered directory does not exist: {args.diversity_filtered_dir}"
    assert args.gte_prediction_scores_dir.exists(), f"GTE prediction scores directory does not exist: {args.gte_prediction_scores_dir}"
    assert args.svm_prediction_scores_dir.exists(), f"SVM prediction scores directory does not exist: {args.svm_prediction_scores_dir}"
    assert args.dnn_prediction_scores_dir.exists(), f"DNN prediction scores directory does not exist: {args.dnn_prediction_scores_dir}"
    assert 0 <= args.phishfrac < 1, "phishfrac must be between 0 and 1"
    assert 0 <= args.benignfrac < 1, "benignfrac must be between 0 and 1"
    args.difficulty_filtered_dir.mkdir(parents=True, exist_ok=True)
    
    # Load leakage filtered SHA256 hashes
    leakage_filtered_sha256 = load_filtered_sha256s(args.leakage_filtered_dir, args.class_type)
    print(f"Loaded {len(leakage_filtered_sha256)} SHA256 hashes for leakage filtering of class type '{args.class_type}'")
    # Load diversity filtered SHA256 hashes
    diversity_filtered_sha256 = load_filtered_sha256s(args.diversity_filtered_dir, args.class_type)
    print(f"Loaded {len(diversity_filtered_sha256)} SHA256 hashes for diversity filtering of class type '{args.class_type}'")
    # Combine both filtered SHA256 hashes
    leakage_filtered_sha256 = set(leakage_filtered_sha256)
    diversity_filtered_sha256 = set(diversity_filtered_sha256)
    filtered = leakage_filtered_sha256.union(diversity_filtered_sha256)
    print(f"Total filtered SHA256 hashes for class type '{args.class_type}': {len(filtered)}")
    
    # Fetch sha256's applying leakage and diversity filters
    input_dir = args.input_test_data_dir / ('phishing' if args.class_type == 'phish' else args.class_type)
    filenames = list(input_dir.glob("**/*.json"))
    print(f"Total {len(filenames)} files found in {input_dir} for class type '{args.class_type}'")
    remaining_sha256 = [f.stem for f in filenames if f.stem not in filtered]
    print(f"Total {len(remaining_sha256)} files remaining after applying leakage filter for class type '{args.class_type}'")

    # Load GTE prediction scores
    print("Loading GTE prediction scores...")
    gte_scores = get_predictions(args.gte_prediction_scores_dir, args.class_type, args.num_workers, return_type="dataset")
    print(f"Loaded GTE prediction scores for {len(gte_scores)} items of class type '{args.class_type}'")
    gte_scores = gte_scores.to_pandas()
    gte_scores = gte_scores[['sha256', 'score']]
    gte_scores = gte_scores.rename(columns={'score': 'gte_score'})
    gte_scores = gte_scores.loc[gte_scores['sha256'].isin(remaining_sha256)]
    print(f"Total {len(gte_scores)} GTE prediction scores remaining after applying leakage & diversity filters for class type '{args.class_type}'")

    # Load SVM prediction scores
    print("Loading SVM prediction scores...")
    svm_scores = get_predictions(args.svm_prediction_scores_dir, args.class_type, args.num_workers, return_type="dataset")
    print(f"Loaded SVM prediction scores for {len(svm_scores)} items of class type '{args.class_type}'")
    svm_scores = svm_scores.to_pandas()
    svm_scores = svm_scores[['sha256', 'score']]
    svm_scores = svm_scores.rename(columns={'score': 'svm_score'})
    svm_scores = svm_scores.loc[svm_scores['sha256'].isin(remaining_sha256)]
    print(f"Total {len(svm_scores)} SVM prediction scores remaining after applying leakage & diversity filters for class type '{args.class_type}'")

    # Load DNN prediction scores
    print("Loading DNN prediction scores...")
    dnn_scores = get_predictions(args.dnn_prediction_scores_dir, args.class_type, args.num_workers, return_type="dataset")
    print(f"Loaded DNN prediction scores for {len(dnn_scores)} items of class type '{args.class_type}'")
    dnn_scores = dnn_scores.to_pandas()
    dnn_scores = dnn_scores[['sha256', 'score']]
    dnn_scores = dnn_scores.rename(columns={'score': 'dnn_score'})
    dnn_scores = dnn_scores.loc[dnn_scores['sha256'].isin(remaining_sha256)]
    print(f"Total {len(dnn_scores)} DNN prediction scores remaining after applying leakage & diversity filters for class type '{args.class_type}'")

    # Merge all prediction scores
    merged_scores = gte_scores.merge(svm_scores, on='sha256', how='inner').merge(dnn_scores, on='sha256', how='inner')
    merged_scores["score"] = (merged_scores['gte_score'] + merged_scores['svm_score'] + merged_scores['dnn_score']) / 3
    print(f"Total {len(merged_scores)} items with all three prediction scores available for class type '{args.class_type}'")


    # Get sha256's to drop
    to_drop = filter_by_difficulty(merged_scores, args.class_type, delta=0.15, phishfrac=args.phishfrac, benignfrac=args.benignfrac)
    print(f"Total {len(to_drop)} items to drop based on difficulty filtering for class type '{args.class_type}'")

    # Save difficulty filtered SHA256 hashes
    if len(to_drop) == 0:
        print("No items to drop based on difficulty filtering. Exiting without saving.")
        return
    else:
        args.difficulty_filtered_dir.mkdir(parents=True, exist_ok=True)
        difficulty_filtered_path = args.difficulty_filtered_dir / f"{args.class_type}.pkl"
        with open(difficulty_filtered_path, "wb") as f:
            dill.dump(to_drop, f)
        print(f"Saved difficulty filtered SHA256 hashes to {difficulty_filtered_path} for class type '{args.class_type}'")

if __name__ == "__main__":
    main()