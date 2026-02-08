import argparse
from pathlib import Path
from ftlangdetect import detect
import os
import time
import multiprocessing as mp
import json
from bs4 import BeautifulSoup
from tqdm import tqdm

def update_language(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        html = data.get('html', '')
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        # to avoid the error: predict processes one line at a time (remove '\n')
        text = text.replace('\n', ' ').replace('\r', ' ').strip()
        if len(text) > 0:
            result = detect(text=text, low_memory=False)
            data['lang'] = str(result['lang'])
            data['lang_score'] = float(result['score'])
        else:
            data['lang'] = None
            data['lang_score'] = -1.0

        # print(f"Updated {filename}: lang={data['lang']}, lang_score={data['lang_score']}")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return None
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return filename


def main():
    parser = argparse.ArgumentParser(description="Language Detection on Dataset")
    parser.add_argument("--dir", type=Path, required=True, help="Directory of jsons with html.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers to use. Defaults to 90% of CPU cores.")
    args = parser.parse_args()

    assert args.dir.exists(), f"Dataset directory {args.dir} does not exist."
    num_proc = args.num_workers if args.num_workers is not None else int(os.cpu_count() * 0.9)
    
    print(f"Starting language detection with {num_proc} processes...")
    start_time = time.time()

    for class_type in ['phishing', 'benign']:
        print(f"Processing {class_type}...")
        datapath = args.dir / class_type
        assert datapath.exists(), f"{class_type} path {datapath} does not exist."
        filenames = list(datapath.glob("*.json"))

        # for testing purposes, limit to first 100 files
        # filenames = filenames[:100]

        with mp.Pool(processes=num_proc) as pool:
            results = pool.map(update_language, tqdm(filenames))

        failed_files = [f for f in results if f is not None]
        if failed_files:
            print(f"Failed to process {len(failed_files)} files in {class_type}:")
            for f in failed_files:
                print(f"  {f}")
        else:
            print(f"All files in {class_type} processed successfully.")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Language detection completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("Done!")

if __name__ == "__main__":
    main()