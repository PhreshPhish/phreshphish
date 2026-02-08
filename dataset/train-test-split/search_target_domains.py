import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv
from tqdm import tqdm


def load_quota_config(config_file):
    """Load or initialize the quota tracking config."""
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {
            "last_reset_date": datetime.now().strftime("%Y-%m-%d"),
            "api_keys": {
                "search_engine_1": {"requests_today": 0},
                "search_engine_2": {"requests_today": 0}
            }
        }


def save_quota_config(config_file, config):
    """Save the quota tracking config."""
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def reset_quota_if_new_day(config):
    """Reset quota counters if it's a new day."""
    today = datetime.now().strftime("%Y-%m-%d")
    if config.get("last_reset_date") != today:
        config["last_reset_date"] = today
        for key_id in config.get("api_keys", {}):
            config["api_keys"][key_id]["requests_today"] = 0
    return config


def get_available_api_key(config, api_keys_info, daily_limit=100):
    """Get an available API key that hasn't reached the daily limit."""
    for key_id, key_info in api_keys_info.items():
        if key_id not in config["api_keys"]:
            config["api_keys"][key_id] = {
                "requests_today": 0
            }
        
        if config["api_keys"][key_id]["requests_today"] < daily_limit:
            return key_id, key_info
    
    return None, None


def extract_domain(url):
    """Extract domain from URL."""
    parsed = urlparse(url)
    return parsed.netloc


def google_search(query, developer_key, cx):
    """Perform a Google Custom Search and return the first result."""
    try:
        service = build("customsearch", "v1", developerKey=developer_key)
        result = service.cse().list(q=query, cx=cx, num=1).execute()
        
        if 'items' in result and len(result['items']) > 0:
            first_result = result['items'][0]
            url = first_result.get('link', '')
            domain = extract_domain(url) if url else ''
            return url, domain
        else:
            return None, None
    except Exception as e:
        print(f"Error during search for '{query}': {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Fetch target domains using Google Custom Search.")
    parser.add_argument('--input_json', type=str, 
                        default='frequent_targets.json',
                        help='Input/Output JSON file with frequent targets (will be updated in place).')
    parser.add_argument('--config_file', type=str,
                        default='config/search_quota.json',
                        help='JSON file to track daily search quota.')
    parser.add_argument('--daily_limit', type=int, default=100,
                        help='Daily search limit per API key.')
    parser.add_argument('--env_file', type=str, default='.env',
                        help='Path to .env file with API keys.')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv(args.env_file)
    
    # Get API keys from environment
    api_keys_info = {
        "search_engine_1": {
            "developer_key": os.getenv("GOOGLE_API_KEY_1"),
            "cx": os.getenv("GOOGLE_CX_1")
        },
        "search_engine_2": {
            "developer_key": os.getenv("GOOGLE_API_KEY_2"),
            "cx": os.getenv("GOOGLE_CX_2")
        }
    }
    
    # Validate API keys
    for key_id, key_info in api_keys_info.items():
        if not key_info["developer_key"] or not key_info["cx"]:
            print(f"Warning: {key_id} is missing developer_key or cx in .env file")
    
    # Load and reset quota config if needed
    quota_config = load_quota_config(args.config_file)
    quota_config = reset_quota_if_new_day(quota_config)
    
    # Read input JSON
    print(f"Reading input JSON: {args.input_json}")
    if not os.path.exists(args.input_json):
        print(f"Error: Input file {args.input_json} does not exist.")
        return
    
    with open(args.input_json, 'r', encoding='utf-8') as f:
        targets_data = json.load(f)
    
    # Sort by count in descending order
    targets_data = sorted(targets_data, key=lambda x: x.get('count', 0), reverse=True)
    
    # Start timing
    start_time = time.time()
    
    processed_count = 0
    skipped_count = 0
    already_has_url_count = 0
    
    # Count targets that need searching for progress bar
    targets_needing_search = sum(1 for record in targets_data if 'website_url' not in record or not record['website_url'])
    
    # Process each target
    total_targets = len(targets_data)
    with tqdm(total=targets_needing_search, desc="Searching targets", unit="target") as pbar:
        for idx, record in enumerate(targets_data):
            target = record.get('target', '')
            
            # Skip if already has website_url
            if 'website_url' in record and record['website_url']:
                already_has_url_count += 1
                continue
            
            # Check if we can get an available API key
            key_id, key_info = get_available_api_key(quota_config, api_keys_info, args.daily_limit)
            
            if not key_id:
                pbar.write(f"\nINFO: Daily quota exhausted for all API keys ({args.daily_limit} requests per key).")
                pbar.write(f"Processed {processed_count} new targets. Remaining {total_targets - already_has_url_count - processed_count} targets will be searched next run.")
                pbar.write(f"Already had URLs for {already_has_url_count} targets.")
                break
            
            # Create search query
            search_query = f"{target} website"
            
            # Perform search
            pbar.set_description(f"Searching: {target[:30]}..." if len(target) > 30 else f"Searching: {target}")
            website_url, domain = google_search(
                search_query, 
                key_info["developer_key"], 
                key_info["cx"]
            )
            
            # Update quota
            quota_config["api_keys"][key_id]["requests_today"] += 1
            
            if website_url and domain:
                record['website_url'] = website_url
                record['domain'] = domain
                processed_count += 1
            else:
                pbar.write(f"  -> No results found for '{target}'")
                record['website_url'] = None
                record['domain'] = None
                skipped_count += 1
                processed_count += 1
            
            pbar.update(1)
            
            # Save quota config and updated data after each request
            save_quota_config(args.config_file, quota_config)
        with open(args.input_json, 'w', encoding='utf-8') as f:
            json.dump(targets_data, f, indent=2, ensure_ascii=False)
    
    # Write final output (in case loop exited without completing all targets)
    with open(args.input_json, 'w', encoding='utf-8') as f:
        json.dump(targets_data, f, indent=2, ensure_ascii=False)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    
    print(f"\n{'='*50}")
    print(f"Processing Complete")
    print(f"{'='*50}")
    print(f"Total targets: {total_targets}")
    print(f"Already had URLs: {already_has_url_count}")
    print(f"Newly searched: {processed_count}")
    print(f"No results found: {skipped_count}")
    remaining = total_targets - already_has_url_count - processed_count
    if remaining > 0:
        print(f"Remaining to search: {remaining}")
    print(f"File updated: {args.input_json}")
    if hours > 0:
        print(f"Time taken: {hours}h {minutes}m {seconds:.1f}s")
    elif minutes > 0:
        print(f"Time taken: {minutes}m {seconds:.1f}s")
    else:
        print(f"Time taken: {seconds:.1f}s")
    print(f"{'='*50}")
    
    # Print quota status
    print("\nQuota status:")
    for key_id, key_data in quota_config["api_keys"].items():
        remaining = args.daily_limit - key_data["requests_today"]
        print(f"  {key_id}: {key_data['requests_today']}/{args.daily_limit} requests used, {remaining} remaining")


if __name__ == "__main__":
    main()
