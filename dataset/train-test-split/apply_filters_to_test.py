import argparse
from pathlib import Path
from utils import load_filtered_sha256s
import shutil
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Apply filters to test data.")
    parser.add_argument("--input_test_data_dir", type=Path, required=True, help="Path to the input test data directory containing phishes & benigns")
    parser.add_argument("--class_type", type=str, choices=["phish", "benign"], required=True, help="Class type: phish or benign")
    parser.add_argument("--leakage_filtered_dir", type=Path, help="Path to the directory containing SHA256 hashes of leakage filtered data of phishes & benigns")
    parser.add_argument("--diversity_filtered_dir", type=Path, help="Path to the directory containing SHA256 hashes of diversity filtered data of phishes & benigns")
    parser.add_argument("--difficulty_filtered_dir", type=Path, help="Path to the directory containing SHA256 hashes of difficulty filtered data of phishes & benigns")
    parser.add_argument("--output_test_data_dir", type=Path, required=True, help="Path to the output test data directory after applying all filters")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for parallel processing")
    args = parser.parse_args()

    args.output_test_data_dir.mkdir(parents=True, exist_ok=True)

    # Load filtered SHA256 hashes
    filtered_sha256 = set()
    if args.leakage_filtered_dir and args.leakage_filtered_dir.exists():
        leakage_filtered_sha256 = load_filtered_sha256s(args.leakage_filtered_dir, args.class_type)
        filtered_sha256 = filtered_sha256.union(set(leakage_filtered_sha256))
        print(f"Loaded {len(leakage_filtered_sha256)} leakage filtered SHA256 hashes for class type '{args.class_type}'")
    if args.diversity_filtered_dir and args.diversity_filtered_dir.exists():
        diversity_filtered_sha256 = load_filtered_sha256s(args.diversity_filtered_dir, args.class_type)
        filtered_sha256 = filtered_sha256.union(set(diversity_filtered_sha256))
        print(f"Loaded {len(diversity_filtered_sha256)} diversity filtered SHA256 hashes for class type '{args.class_type}'")
    if args.difficulty_filtered_dir and args.difficulty_filtered_dir.exists():
        difficulty_filtered_sha256 = load_filtered_sha256s(args.difficulty_filtered_dir, args.class_type)
        filtered_sha256 = filtered_sha256.union(set(difficulty_filtered_sha256))
        print(f"Loaded {len(difficulty_filtered_sha256)} difficulty filtered SHA256 hashes for class type '{args.class_type}'")
    print(f"Total filtered SHA256 hashes for class type '{args.class_type}': {len(filtered_sha256)}")

    # Apply filters to input test data and save to output directory
    input_dir = args.input_test_data_dir / ('phishing' if args.class_type == 'phish' else args.class_type)
    output_dir = args.output_test_data_dir / ('phishing' if args.class_type == 'phish' else args.class_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in tqdm(input_dir.iterdir(), desc=f"Copying {args.class_type} test data"):
        sha256 = item.stem
        if sha256 not in filtered_sha256:
            shutil.copy(item, output_dir / item.name)
    print(f"{len(list(output_dir.iterdir()))} test data saved to {output_dir} for class type '{args.class_type}'")

if __name__ == "__main__":
    main()