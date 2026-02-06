import argparse
import pandas as pd
import json
import os
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm

def read_json_file(filepath):
    """Read a single JSON file and extract filename and target."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {
                'filename': os.path.basename(filepath),
                'target': data.get('target', None)
            }
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Fetch and process targets from JSON files.")
    parser.add_argument(
        '--input_dir', type=str, required=True, 
        help='Directory containing JSON files to process.'
    )
    parser.add_argument(
        '--output_file', type=str, required=True, help='Output JSON file for frequent targets.'
    )
    parser.add_argument(
        '--min_count', type=int, required=True, default=100, 
        help='Minimum count threshold for frequent targets.'
    )
    parser.add_argument(
        '--filter_list', type=str, required=True, default='config/filter_list.json', 
        help='JSON file containing list of targets to filter out.'
    )
    parser.add_argument(
        '--replace_dict', type=str, required=True, default='config/replace_dict.json',
        help='JSON file containing dictionary of target replacements.'
    )
    args = parser.parse_args()
    
    # Load filter list from JSON file
    with open(args.filter_list, 'r', encoding='utf-8') as f:
        filter_list = json.load(f)
    
    # Load replace dict from JSON file
    with open(args.replace_dict, 'r', encoding='utf-8') as f:
        replace_dict = json.load(f)
    
    # Get all JSON files
    json_dir = args.input_dir
    json_files = list(Path(json_dir).glob('*.json'))
    
    # Read files using multiprocessing with progress bar
    with Pool() as pool:
        results = list(tqdm(pool.imap(read_json_file, json_files), total=len(json_files), desc="Processing files"))
    
    # Filter out None results and create DataFrame
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    
    # Fill NaN values in target
    df['target'] = df['target'].fillna('n/a')
    
    # Convert target to lowercase
    df['target'] = df['target'].str.lower()
    
    # Filter out specified targets
    df = df[~df['target'].isin(filter_list)]
    
    # Replace '&amp;' with '&' in target strings
    for old, new in replace_dict.items():
        if old in new:
            df.loc[df['target'] == old, 'target'] = new
        else:
            df['target'] = df['target'].str.replace(old, new, regex=False)
    
    # Count occurrences of each target
    targets = df['target'].value_counts().reset_index()
    targets.columns = ['target', 'count']
    frequent_targets = targets.loc[targets['count'] >= args.min_count]
    if len(frequent_targets) > 0:
        if args.output_file:
            # Convert DataFrame to list of dictionaries and write to JSON
            frequent_targets_list = frequent_targets.to_dict('records')
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(frequent_targets_list, f, indent=2, ensure_ascii=False)
    print(f"Found {len(frequent_targets)} frequent targets with at least {args.min_count} occurrences")

if __name__ == "__main__":
    main()