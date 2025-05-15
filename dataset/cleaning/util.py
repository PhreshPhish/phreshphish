from bs4 import BeautifulSoup
import os
import numpy as np
import polars as pl
from pathlib import Path
import json
import re
from tqdm import tqdm
import scipy
import multiprocessing as mp
from scipy.sparse import vstack
from typing import Dict, Any, List, Tuple
import csv
import random
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt

from collections import Counter

from IPython.display import clear_output

import concurrent.futures

from itertools import combinations
from collections import defaultdict
from sklearn.metrics.pairwise import pairwise_distances


def load_our_sample(filename: str) -> dict[str, Any]:
    label = 1 if "phish" in filename else -1

    with open(filename, "r") as fp:
        data = json.load(fp)

    url = normalize_url(data["url"])
    html = data["html"]

    if not re.search(r"<\s*\w+[^>]*>", html):
        soup = None
    else:
        soup = BeautifulSoup(html, "lxml")

    if soup:
        html = soup.prettify()
        title = str(soup.title.string).strip().lower() if soup.title else None
        text = soup.get_text(separator=" ", strip=True)
    else:
        html = None
        title = None
        text = None

    return {"url": url, "html": html, "text": text, "title": title, "label": label}


def load_aljofey_labels(label_path: str):
    labels = []
    dirname = os.path.dirname(label_path)
    with open(label_path, "r") as fp:
        for row in fp.readlines():
            idx, url, label = row.split(" ")
            url = normalize_url(url)
            html_path = f"{dirname}/{idx}.txt"
            labels.append((html_path, url, int(label.strip())))

    return labels


def load_sample(sample):
    filename, url, label = sample

    try:
        with open(filename, "r") as fp:
            html = fp.read()
    except Exception:
        html = ""

    if not re.search(r"<\s*\w+[^>]*>", html):
        soup = None
    else:
        soup = BeautifulSoup(html, "lxml")

    if soup:
        html = soup.prettify()
        title = str(soup.title.string).strip().lower() if soup.title else None
        text = soup.get_text(separator=" ", strip=True)
    else:
        html = None
        title = None
        text = None

    return {"url": url, "html": html, "text": text, "title": title, "label": label}


def generate_random_vectors(dim, n_vectors):
    """
    generate random projection vectors
    the dims comes first in the matrix's shape,
    so we can use it for matrix multiplication.
    """
    return np.random.randn(dim, n_vectors)


def train_lsh(X_tfidf, n_vectors, seed=None):
    if seed is not None:
        np.random.seed(seed)

    dim = X_tfidf.shape[1]
    random_vectors = generate_random_vectors(dim, n_vectors)

    # partition data points into bins,
    # and encode bin index bits into integers
    bin_indices_bits = X_tfidf.dot(random_vectors) >= 0
    powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)
    bin_indices = bin_indices_bits.dot(powers_of_two)

    # update `table` so that `table[i]` is the list of document ids with bin index equal to i
    table = defaultdict(list)
    for idx, bin_index in enumerate(bin_indices):
        table[bin_index].append(idx)

    model = {
        "table": table,
        "random_vectors": random_vectors,
        "bin_indices": bin_indices,
        "bin_indices_bits": bin_indices_bits,
    }
    return model


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


def get_nearest_neighbors(
    X_tfidf: "scipy.sparse.csr_matrix",
    query_vector: np.ndarray,
    model: Dict[str, Any],
    max_search_radius: int = 3,
) -> pl.DataFrame:
    """
    Returns a Polars DataFrame with columns ['id', 'distance'], sorted by distance.
    """
    table = model["table"]
    random_vectors = model["random_vectors"]

    # 1) compute the bit‐vector for this query
    bin_index_bits = np.ravel(query_vector.dot(random_vectors) >= 0)

    # 2) gather candidate indices from nearby bins
    candidate_set = set()
    for r in range(max_search_radius + 1):
        candidate_set = search_nearby_bins(bin_index_bits, table, r, candidate_set)

    # 3) compute true cosine distances
    candidate_list = list(candidate_set)
    candidates = X_tfidf[candidate_list]
    distances = pairwise_distances(candidates, query_vector, metric="cosine").ravel()

    # 4) build a Polars DataFrame and sort by distance
    nn_df = pl.DataFrame({"id": candidate_list, "distance": distances}).sort(
        "distance", descending=True
    )

    return nn_df


# compile once at import time
TAG_RE = re.compile(r"<\s*\w+[^>]*>")


def _parse_sample(sample: tuple[str, str, str]) -> dict[str, pl.Any]:
    """Given (filename, url, label), read & parse HTML into fields."""
    filename, url, label = sample
    try:
        html_raw = Path(filename).read_text()
    except Exception:
        return {"url": url, "html": None, "text": None, "title": None, "label": label}

    # skip expensive BS4 parse if no HTML tags
    if not TAG_RE.search(html_raw):
        return {"url": url, "html": None, "text": None, "title": None, "label": label}

    soup = BeautifulSoup(html_raw, "lxml")
    pretty = soup.prettify()
    title = (
        soup.title.string.strip().lower() if soup.title and soup.title.string else None
    )
    text = soup.get_text(separator=" ", strip=True)

    return {"url": url, "html": pretty, "text": text, "title": title, "label": label}


def load_crawling2024_labels(
    phish_index_csv: str, benign_index_csv: str
) -> List[Tuple[str, str, int]]:
    """
    Read the two index CSVs and return a list of (url, after_content_path, label)
      where label is 1 for phish and -1 for benign.
    """
    entries: List[Tuple[str, str, int]] = []

    def build_num_dir_map(index_csv: str) -> dict[str, str]:
        base_dir = os.path.dirname(index_csv)
        num_dir: dict[str, str] = {}
        for name in os.listdir(base_dir):
            full = os.path.join(base_dir, name)
            if os.path.isdir(full) and "-" in name:
                num, _ = name.split("-", 1)
                num_dir.setdefault(num, full)
        return num_dir

    # build once per folder
    phish_map = build_num_dir_map(phish_index_csv)
    benign_map = build_num_dir_map(benign_index_csv)

    def process(index_csv: str, label: int, num_map: dict[str, str]):
        with open(index_csv, newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                num = row["num"]
                url = normalize_url(row["request_url"])
                data_dir = num_map.get(num)
                if not data_dir:
                    continue
                content_path = os.path.join(data_dir, "after_content.txt")
                if os.path.isfile(content_path):
                    entries.append((content_path, url, label))

    # process phish then benign
    process(phish_index_csv, 1, phish_map)
    process(benign_index_csv, -1, benign_map)

    return entries


def normalize_url(url: str):
    """Normalizes a URL to a canonical form.
    It might not be the 'best' canonical form, but it's ours <3.
    """
    url = url.lower()
    url = url.rstrip("/")
    return url


def load_dataset(
    labels: [], max_rows: int | None = None, num_workers: int | None = None
) -> pl.DataFrame:
    """
    Load labels, parse each HTML file in parallel, and return a Polars DataFrame.
    """
    import warnings

    warnings.filterwarnings("ignore")

    if max_rows is not None:
        labels = labels[:max_rows]

    num_workers = num_workers or mp.cpu_count()
    chunksize = 10

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
        parsed = pool.map(_parse_sample, labels, chunksize=chunksize)
        rows = []
        with tqdm(total=len(labels), desc="Loading dataset") as pbar:
            for row in parsed:
                rows.append(row)
                pbar.update()
        df = pl.from_dicts(rows)

    return df


def find_duplicate_urls(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds a 'duplicate_url' column that is True only for subsequent
    occurrences of the same 'url'. First instance is False.
    Prints how many duplicates were found.
    """
    df = df.with_row_count("row_idx")

    df = df.with_columns(
        [
            # Rank the occurrence index of each URL using dense_rank per group
            pl.col("row_idx")
            .rank("dense")
            .over("url")
            .alias("dup_index")
        ]
    )

    # Mark rows as duplicate if their dup_index > 1 (i.e. not the first appearance)
    df = df.with_columns((pl.col("dup_index") > 1).alias("duplicate_url"))

    n_dups = df.filter(pl.col("duplicate_url")).height
    print(f"Found {n_dups} subsequent duplicate URLs")

    return df.drop(["row_idx", "dup_index"])


def find_bad_titles(df: pl.DataFrame) -> pl.DataFrame:
    """
    Finds rows whose `title` matches any of a set of "bad" patterns.
    """
    _bad_titles = [
        "400",
        "404",
        "410",
        "403",
        "found",
        "encontrada",
        "forbidden",
        "error",
        "suspended",
        "bad request",
        "cloudflare",
        "just a moment…",
        "warning! | there might be a problem with the requested link",
        "url shortener, branded short links & analytics | tinyurl",
        "denied",
    ]

    _BAD_PATTERN = "(?i)" + "|".join(map(re.escape, _bad_titles))
    df = df.with_columns(
        pl.col("title").str.contains(_BAD_PATTERN).fill_null(False).alias("bad_title")
    )
    print(f"Found {df.filter(pl.col('bad_title')).height} bad titles")
    return df


def find_empty_html(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds an `empty_html` column, True if the `html` field is null or empty.
    """
    df = df.with_columns(
        (pl.col("html").is_null() | pl.col("html").eq("")).alias("empty_html")
    )
    print(f"Found {df.filter(pl.col('empty_html')).height} empty HTML documents")
    return df


def drop_bad_rows(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(
        ~(pl.col("duplicate_url") | pl.col("bad_title") | pl.col("empty_html"))
    ).drop(["duplicate_url", "bad_title", "empty_html"])

    return df


def _init_worker(vectorizer):
    global _tfidf
    _tfidf = vectorizer


def _transform_chunk(chunk):
    # replace None/empty with empty string
    clean = [t or "" for t in chunk]
    return _tfidf.transform(clean)


def build_tfidf(
    df: pl.DataFrame,
    max_features: int | None = None,
    sample_frac: float = 1.0,
    random_state: int = 42,
    n_jobs: int | None = None,
    chunk_size: int | None = None,
) -> scipy.sparse.csr_matrix:
    """
    Fit a TF-IDF on a random subset of documents, then transform the full corpus using process-based parallelism.

    Args:
        df: Polars DataFrame with column "html" containing text.
        max_features: Maximum number of features (vocabulary size).
        sample_frac: Fraction of documents to use for fitting (0 < sample_frac <= 1).
        random_state: Seed for reproducible sampling.
        n_jobs: Number of worker processes. Defaults to os.cpu_count().
        chunk_size: Number of docs per transform chunk. Defaults to len(texts)//(n_jobs*4).

    Returns:
        scipy.sparse.csr_matrix: TF-IDF feature matrix for all documents.
    """
    texts = df["html"].fill_null("").to_list()
    n_docs = len(texts)

    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be in (0, 1]")

    if sample_frac < 1.0:
        k = int(sample_frac * n_docs)
        rng = random.Random(random_state)
        fit_texts = rng.sample(texts, k)
    else:
        fit_texts = texts

    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=0.0,
        max_features=max_features,
        stop_words="english",
    )
    print(f"Fitting TF-IDF on {len(fit_texts)} / {n_docs} docs...")
    tfidf.fit(fit_texts)

    n_jobs = n_jobs or os.cpu_count() or 1
    # chunk_size = chunk_size or max(1, n_docs // (n_jobs * 4))
    chunk_size = 10
    chunks = [texts[i : i + chunk_size] for i in range(0, n_docs, chunk_size)]

    print(
        f"Transforming all {n_docs} documents in {len(chunks)} chunks on {n_jobs} processes..."
    )
    with mp.Pool(processes=n_jobs, initializer=_init_worker, initargs=(tfidf,)) as pool:
        mats = list(
            tqdm(
                pool.imap(_transform_chunk, chunks),
                total=len(chunks),
                desc="Transform",
                ncols=100,
            )
        )

    X_tfidf = vstack(mats, format="csr")
    return X_tfidf


def print_progress(df: pl.DataFrame) -> None:
    """
    Print counts and percentages of processed, kept, and rejected rows.
    """
    total = df.height
    assert total > 0, "DataFrame is empty"

    processed = df.filter(pl.col("keep").is_not_null()).height
    keep_count = df.filter(pl.col("keep") == True).height
    reject_count = df.filter(pl.col("keep") == False).height

    pct = lambda part, whole: (part / whole * 100) if whole else 0.0

    print(f"Processed: {processed} / {total} ({pct(processed, total):.2f}%)")
    if processed:
        print(
            f"  Keep:   {keep_count} / {processed} ({pct(keep_count, processed):.2f}%)"
        )
        print(
            f"  Reject: {reject_count} / {processed} ({pct(reject_count, processed):.2f}%)"
        )
    print("-" * 40)


def plot_distance_histogram(
    neighbors: pl.DataFrame, bins: int = 10, max_x: float = 1.0
) -> None:
    """
    Show a log-scaled histogram of the 'distance' column.
    """
    distances = neighbors["distance"].to_list()
    plt.figure(figsize=(6, 4))
    plt.hist(distances, bins=bins, log=True)
    plt.xlabel("Distance")
    plt.xlim(0, max_x)
    plt.ylabel("Frequency")
    plt.title("Histogram of Neighbor Distances")
    plt.show()


def update_keep_column(
    df: pl.DataFrame, indices: List[int], keep_value: bool
) -> pl.DataFrame:
    """
    Mark the given row-indices as keep/reject in the 'keep' column.
    """
    return df.with_columns(
        pl.when(pl.col("index").is_in(indices))
        .then(keep_value)
        .otherwise(pl.col("keep"))
        .alias("keep")
    )


def run_cleaning(
    X_tfidf: "scipy.sparse.csr_matrix",
    df: pl.DataFrame,
    lsh_model: Dict[str, Any],
    budget: int = 25,
    group_col: str = "bin",
    distance_cutoff: float = 0.2,
) -> pl.DataFrame:
    """
    Iterate over bins (largest first), sample one row per bin, fetch neighbors,
    show diagnostics, and let the user keep/reject clusters until budget is used.
    """
    # Ensure index & keep columns exist
    if "index" not in df.columns:
        df = df.with_row_count("index")
    if "keep" not in df.columns:
        df = df.with_columns(pl.lit(None).alias("keep"))

    # ── get bins sorted by size desc via group_by/count ───────────────────────
    bin_counts = df.group_by(group_col).count().sort("count", descending=True)
    bins = bin_counts[group_col].to_list()

    spent = 0
    for bin_id in bins:
        if spent >= budget:
            break

        # only un-annotated rows in this bin
        mask = (pl.col(group_col) == bin_id) & pl.col("keep").is_null()
        candidates = df.filter(mask)
        if candidates.height == 0:
            continue

        # sample one row
        sample = candidates.sample(n=1)
        idx = sample["index"][0]

        clear_output(wait=True)

        # fetch & filter neighbors
        neighbors = get_nearest_neighbors(
            X_tfidf, X_tfidf[idx], lsh_model, max_search_radius=4
        ).filter(pl.col("distance") <= distance_cutoff)
        neighbor_ids = neighbors["id"].to_list()
        if len(neighbor_ids) == 0:
            continue

        # build title/label counters
        titles = df.filter(pl.col("index").is_in(neighbor_ids))["title"].to_list()
        labels = df.filter(pl.col("index").is_in(neighbor_ids))["label"].to_list()
        top_titles = Counter(titles).most_common(10)
        top_labels = Counter(labels).most_common(2)

        print_progress(df)
        print(f"Bin: {bin_id}")
        print(f"Bin members: {df.filter(pl.col(group_col) == bin_id).height}")
        print(f"URL: {sample['url'][0]}")
        print(f"Neighbors: {neighbors.height}")
        print(f"Title: {sample['title'][0]}")
        print(f"Labels: {top_labels}")
        print(f"Top titles:\n{top_titles}")
        print("HTML:")
        print("-" * 10 + "\n\n")
        print(sample["html"][0][:1000])
        print("\n\n" + "-" * 10)
        print("Text:")
        print("-" * 10 + "\n\n")
        print(sample["text"][0][:1000])
        print("\n\n" + "-" * 10)

        plot_distance_histogram(neighbors)

        choice = input("Keep (k), remove (r), skip (s): ").strip().lower()
        if choice == "k":
            df = update_keep_column(df, neighbor_ids + [idx], True)
            spent += 1
        elif choice == "r":
            df = update_keep_column(df, neighbor_ids + [idx], False)
            spent += 1
        elif choice == "s":
            continue
        else:
            break

    clear_output(wait=True)
    print("Budget exhausted. Finalizing...")
    print_progress(df)
    return df
