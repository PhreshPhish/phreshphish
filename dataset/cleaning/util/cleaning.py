import glob
import os
import json
import re
import random
import shutil
import multiprocessing as mp
import gc
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl
import scipy
import joblib
from tqdm import tqdm
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from itertools import combinations
from IPython.display import clear_output
from selectolax.parser import HTMLParser
from bs4 import BeautifulSoup

# compile once at import time
HTML_TAG_PATTERN = re.compile(r"<\s*\w+[^>]*>")


def load_sample(filename: str) -> dict[str, Any]:
    """
    Load a sample JSON file.
    """
    try:
        label = "phishing" if "phishing" in filename else "benign"
        with open(filename, "r") as fp:
            data = json.load(fp)

        sha256 = data["sha256"]
        url = normalize_url(data["url"])
        html = data.get("html", data.get("html_content", ""))

        # Parse title if possible, but do NOT parse text
        if html and HTML_TAG_PATTERN.search(html):
            parser = HTMLParser(html)
            title_tag = parser.css_first("title")
            title = title_tag.text().strip().lower() if title_tag else None
        else:
            title = None

        return {
            "sha256": sha256,
            "url": url,
            "html": html,
            "title": title,
            "label": label,
            "filename": filename,
        }
    except Exception as exp:
        raise RuntimeError(f"Failed to process {filename}") from exp


def load_batch(filenames_batch):
    results = []
    errors = []

    for filename in filenames_batch:
        try:
            results.append(load_sample(filename))
        except Exception as e:
            errors.append((filename, str(e)))

    return results, errors


def load_sample_dir(
    base_dir: str,
    max_samples: int = -1,
    n_jobs: int = -1,
    batch_size: int = 32,
) -> pl.DataFrame:
    """
    Load a dataset from JSON files in the specified directory using multiprocessing.
    """
    filenames = glob.glob(os.path.join(base_dir, "*.json"))
    # filenames = sorted(filenames, key=lambda x: os.stat(x).st_ino)
    filenames = sorted(filenames)
    if max_samples > 0: filenames = filenames[:max_samples]
    print(f"Total files found: {len(filenames):,}")

    batches = [
        filenames[i : i + batch_size] for i in range(0, len(filenames), batch_size)
    ]

    schema = {
        "sha256": pl.Utf8,
        "url": pl.Utf8,
        "html": pl.Utf8,
        "title": pl.Utf8,
        "label": pl.Utf8,
        "filename": pl.Utf8,
    }

    dfs = []
    all_errors = []
    success_count = 0
    fail_count = 0

    with tqdm(total=len(filenames), desc="Loading data points") as pbar:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(load_batch, batch): len(batch) for batch in batches
            }
            for future in as_completed(futures):
                try:
                    batch_results, batch_errors = future.result(timeout=5)
                    batch_size = futures[future]

                    # Convert batch to DataFrame immediately
                    if batch_results:
                        dfs.append(pl.DataFrame(batch_results, schema=schema))

                    all_errors.extend(batch_errors)
                    success_count += len(batch_results)
                    fail_count += len(batch_errors)
                    pbar.set_postfix({"success": success_count, "fail": fail_count})
                    pbar.update(batch_size)
                except Exception as e:
                    print(f"\nBatch failed: {e}")
                    batch_size = futures[future]
                    fail_count += batch_size
                    pbar.set_postfix({"success": success_count, "fail": fail_count})
                    pbar.update(batch_size)

    # Concatenate all DataFrames at once
    df = pl.concat(dfs) if dfs else pl.DataFrame(schema=schema)
    return df


def generate_random_vectors(dim, n_vectors):
    """
    generate random projection vectors
    the dims comes first in the matrix's shape,
    so we can use it for matrix multiplication.
    """
    return np.random.randn(dim, n_vectors)


def _build_table_chunk(args):
    """Helper function to build table for a chunk of bin indices."""
    indices, bin_indices_chunk = args
    local_table = defaultdict(list)
    for local_idx, bin_index in enumerate(bin_indices_chunk):
        global_idx = indices[local_idx]
        local_table[bin_index].append(global_idx)
    return dict(local_table)


def train_lsh(X_tfidf, n_vectors, seed=None, n_jobs=None, model_path=None):
    """
    Train an LSH model on TF-IDF vectors with parallel table building.
    
    Args:
        X_tfidf: Sparse TF-IDF matrix.
        n_vectors: Number of random projection vectors.
        seed: Random seed for reproducibility.
        n_jobs: Number of processes for parallel table building. Defaults to os.cpu_count().
        model_path: Path to save/load the LSH model (e.g., 'lsh_model.joblib').
                   If file exists, it will be loaded instead of training.
    
    Returns:
        Dictionary containing LSH model components.
    """
    # Check if model already exists and load it
    if model_path and os.path.exists(model_path):
        print(f"[train_lsh] Loading existing LSH model from {model_path}...")
        with tqdm(total=1, desc="Loading LSH model", unit="step") as pbar:
            model = joblib.load(model_path)
            pbar.update(1)
        print(f"[train_lsh] Loaded LSH model with {len(model['table'])} bins")
        return model
    
    print(f"[train_lsh] Training LSH with {n_vectors} projection vectors...")
    
    if seed is not None:
        np.random.seed(seed)

    n_jobs = n_jobs or os.cpu_count() or 1
    n_docs = X_tfidf.shape[0]
    dim = X_tfidf.shape[1]
    
    print(f"[train_lsh] Documents: {n_docs:,}, Features: {dim:,}")
    
    # Generate random projection vectors
    with tqdm(total=1, desc="Generating random vectors", unit="step") as pbar:
        random_vectors = generate_random_vectors(dim, n_vectors)
        pbar.update(1)

    # Partition data points into bins (matrix multiplication is already parallel via BLAS)
    print(f"[train_lsh] Computing bin assignments via matrix multiplication...")
    with tqdm(total=1, desc="Computing hash bits", unit="step") as pbar:
        bin_indices_bits = X_tfidf.dot(random_vectors) >= 0
        pbar.update(1)
    
    # Encode bin index bits into integers
    with tqdm(total=1, desc="Converting bits to bin indices", unit="step") as pbar:
        powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)
        bin_indices = bin_indices_bits.dot(powers_of_two)
        pbar.update(1)
    
    del powers_of_two
    gc.collect()
    
    # Build the hash table in parallel
    print(f"[train_lsh] Building hash table with {n_jobs} processes...")
    chunk_size = max(1, n_docs // (n_jobs * 4))
    chunks = []
    
    for i in range(0, n_docs, chunk_size):
        end_idx = min(i + chunk_size, n_docs)
        indices = list(range(i, end_idx))
        bin_indices_chunk = bin_indices[i:end_idx]
        chunks.append((indices, bin_indices_chunk))
    
    # Process chunks in parallel
    table = defaultdict(list)
    with tqdm(total=len(chunks), desc="Building hash table", unit="chunks") as pbar:
        with mp.Pool(processes=n_jobs) as pool:
            for partial_table in pool.imap_unordered(_build_table_chunk, chunks):
                # Merge partial table into main table
                for bin_idx, doc_list in partial_table.items():
                    table[bin_idx].extend(doc_list)
                pbar.update(1)
    
    print(f"[train_lsh] Created {len(table)} bins")
    
    # note that we're storing the bin_indices here
    # so we can do some ad-hoc checking with it,
    # this isn't actually required
    model = {
        "table": table,
        "random_vectors": random_vectors,
        "bin_indices": bin_indices,
        "bin_indices_bits": bin_indices_bits,
    }
    
    # Save model to disk if path provided
    if model_path:
        print(f"[train_lsh] Saving LSH model to {model_path}...")
        joblib.dump(model, model_path, compress=3)
        print(f"[train_lsh] LSH model saved successfully")
    
    gc.collect()
    print(f"[train_lsh] Complete!")
    
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
    max_neighbors: int | None = None,
) -> pl.DataFrame:
    """
    Returns a Polars DataFrame with columns ['index', 'distance'], sorted by
    ascending distance (true nearest neighbors first).

    - 'index' matches df['index'] (row_index) type-wise and name-wise.
    - 'distance' is stored as float32 for efficiency.
    """
    table = model["table"]
    random_vectors = model["random_vectors"]

    # 1) compute the bit‐vector for this query
    bin_index_bits = np.ravel(query_vector.dot(random_vectors) >= 0)

    # 2) gather candidate indices from nearby bins
    candidate_set: set[int] = set()
    for r in range(max_search_radius + 1):
        candidate_set = search_nearby_bins(bin_index_bits, table, r, candidate_set)

    if not candidate_set:
        return pl.DataFrame({"index": [], "distance": []})

    # 3) compute true cosine distances
    candidate_list = list(candidate_set)
    candidates = X_tfidf[candidate_list]
    distances = pairwise_distances(candidates, query_vector, metric="cosine").ravel()

    # enforce dtypes & ascending sort
    nn_df = pl.DataFrame(
        {
            "index": pl.Series(candidate_list, dtype=pl.Int64),
            "distance": pl.Series(distances.astype("float32"), dtype=pl.Float32),
        }
    ).sort("distance", descending=False)

    if max_neighbors is not None:
        nn_df = nn_df.head(max_neighbors)

    return nn_df


def normalize_url(url: str):
    """Normalizes a URL to a canonical form.
    It might not be the 'best' canonical form, but it's ours <3.
    """
    url = url.lower()
    url = url.rstrip("/")
    return url


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
    print(f"Found {n_dups:,} subsequent duplicate URLs")

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
        "just a moment...",
        "warning! | there might be a problem with the requested link",
        "url shortener, branded short links & analytics | tinyurl",
        "denied",
    ]

    _BAD_PATTERN = "(?i)" + "|".join(map(re.escape, _bad_titles))
    df = df.with_columns(
        pl.col("title").str.contains(_BAD_PATTERN).fill_null(False).alias("bad_title")
    )
    print(f"Found {df.filter(pl.col('bad_title')).height:,} bad titles")
    return df


def find_empty_html(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds an `empty_html` column, True if the `html` field is null or empty.
    """
    df = df.with_columns(
        (
            pl.col("html").is_null() | pl.col("html").eq("")  # also flag empty strings
        ).alias("empty_html")
    )
    print(f"Found {df.filter(pl.col('empty_html')).height:,} empty HTML documents")
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
    batch_size: int | None = None,
    vectorizer_path: str | None = None,
) -> scipy.sparse.csr_matrix:
    """
    Fit a TF-IDF on a random subset of documents, then transform the full corpus using process-based parallelism.
    Optimized for memory efficiency with large datasets. Can load a previously saved vectorizer.

    Args:
        df: Polars DataFrame with column "html" containing text.
        max_features: Maximum number of features (vocabulary size).
        sample_frac: Fraction of documents to use for fitting (0 < sample_frac <= 1).
        random_state: Seed for reproducible sampling.
        n_jobs: Number of worker processes. Defaults to os.cpu_count().
        batch_size: Number of docs per transform chunk. Defaults to 1000.
        vectorizer_path: Path to save/load the fitted vectorizer (e.g., 'tfidf_vectorizer.joblib').
                        If file exists, it will be loaded instead of fitting.
                        If None, vectorizer will not be saved to disk.

    Returns:
        scipy.sparse.csr_matrix: TF-IDF feature matrix for all documents.
    """
    print(f"[build_tfidf] Starting with {df.height:,} documents")
    
    # 1. Validate inputs
    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be in (0, 1]")
    
    n_docs = df.height
    n_jobs = n_jobs or os.cpu_count() or 1
    # Use smaller default batch size for better memory management
    batch_size = batch_size or 1000
    
    print(f"[build_tfidf] Using {n_jobs} processes with batch size {batch_size}")
    
    # 2. Check if vectorizer already exists and load it
    if vectorizer_path and os.path.exists(vectorizer_path):
        print(f"[build_tfidf] Loading existing vectorizer from {vectorizer_path}...")
        with tqdm(total=1, desc="Loading vectorizer", unit="step") as pbar:
            tfidf = joblib.load(vectorizer_path)
            pbar.update(1)
        vocab_size = len(tfidf.vocabulary_)
        print(f"[build_tfidf] Loaded vectorizer with {vocab_size:,} features")
    else:
        # 3. Sample subset for fitting (without loading all texts into memory yet)
        if sample_frac < 1.0:
            k = int(sample_frac * n_docs)
            print(f"[build_tfidf] Sampling {k:,} / {n_docs:,} documents for fitting...")
            
            # Use polars sampling to avoid loading all data
            fit_df = df.sample(n=k, seed=random_state)
            fit_texts = fit_df["html"].fill_null("").to_list()
            del fit_df
            gc.collect()
        else:
            print(f"[build_tfidf] Using all {n_docs:,} documents for fitting...")
            fit_texts = df["html"].fill_null("").to_list()
        
        # 4. Initialize and fit vectorizer on subset
        print(f"[build_tfidf] Fitting TF-IDF vectorizer on {len(fit_texts):,} documents...")
        tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=0.0,
            max_features=max_features,
            stop_words="english",
        )
        
        with tqdm(total=1, desc="Fitting TF-IDF", unit="step") as pbar:
            tfidf.fit(fit_texts)
            pbar.update(1)
        
        vocab_size = len(tfidf.vocabulary_)
        print(f"[build_tfidf] Fitted vocabulary size: {vocab_size:,} features")
        
        # Free memory immediately after fitting
        del fit_texts
        gc.collect()
        
        # 5. Save vectorizer to disk if path provided
        if vectorizer_path:
            print(f"[build_tfidf] Saving vectorizer to {vectorizer_path}...")
            joblib.dump(tfidf, vectorizer_path, compress=3)
            print(f"[build_tfidf] Vectorizer saved successfully")
    
    # 6. Define generator to yield chunks lazily (memory-efficient)
    def chunk_generator():
        """Generator that yields text chunks on-demand to avoid loading all at once."""
        for i in range(0, n_docs, batch_size):
            end_idx = min(i + batch_size, n_docs)
            chunk_texts = df[i:end_idx]["html"].fill_null("").to_list()
            yield chunk_texts
            # Explicit cleanup after yielding
            del chunk_texts
    
    # 7. Transform using streaming approach
    num_chunks = (n_docs + batch_size - 1) // batch_size
    print(f"[build_tfidf] Transforming {n_docs:,} documents in {num_chunks} chunks...")
    print(f"[build_tfidf] Memory-efficient streaming mode: processing {n_jobs} chunks at a time")
    
    mats = []
    with tqdm(total=n_docs, desc="TF-IDF transform", unit="docs") as pbar:
        with mp.Pool(processes=n_jobs, initializer=_init_worker, initargs=(tfidf,)) as pool:
            # Use imap to process chunks in a streaming fashion
            # This ensures only n_jobs chunks are in memory at once
            for mat in pool.imap(_transform_chunk, chunk_generator(), chunksize=1):
                mats.append(mat)
                pbar.update(mat.shape[0])
                
                # Periodically run garbage collection and free memory
                if len(mats) % 10 == 0:
                    gc.collect()
    
    # 8. Stitch all chunks into one CSR matrix
    print(f"[build_tfidf] Combining {len(mats)} sparse matrices...")
    with tqdm(total=1, desc="Combining matrices", unit="step") as pbar:
        X_tfidf = vstack(mats, format="csr")
        pbar.update(1)
    
    print(f"[build_tfidf] Final matrix shape: {X_tfidf.shape}")
    print(f"[build_tfidf] Matrix sparsity: {1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]):.4%}")
    
    # Final cleanup
    del mats
    gc.collect()
    
    print(f"[build_tfidf] Complete!")
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


def extract_domain(url):
    try:
        return url.split("//")[-1].split("/")[0].split(":")[0]
    except Exception:
        return None


def prettify_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    pretty_html = soup.prettify()
    return pretty_html


def get_text_from_html(html: str) -> str:
    if html and HTML_TAG_PATTERN.search(html):
        # parser = HTMLParser(html)
        # text = parser.text(separator=" ", strip=True)
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text
    return ""


def load_sample_html(filename: str) -> str:
    with open(filename, "r") as fp:
        return json.load(fp)["html"]


def plot_distance_in_bin(
    neighbors: pl.DataFrame,
    df: pl.DataFrame,
    bin_id: Any,
    distance_cutoff: float,
    group_col: str = "bin",
) -> None:
    """
    Plot the distance between the sampled point and neighbors
    that belong to the same bin. Distance is assumed to be (1 - cosine similarity)
    or any other metric produced by get_nearest_neighbors.

    Also prints the fraction of neighbors in the bin that fall under distance_cutoff.
    """
    # Attach bin info to neighbors
    neighbors_with_bin = neighbors.join(
        df.select(["index", group_col]),
        left_on="index",
        right_on="index",
        how="inner",
    )

    neighbors_in_bin = neighbors_with_bin.filter(pl.col(group_col) == bin_id)

    if neighbors_in_bin.height == 0:
        print(f"No neighbors from bin {bin_id} to plot.")
        return

    distances = neighbors_in_bin["distance"].to_numpy()
    frac_under_cutoff = (distances <= distance_cutoff).mean()

    plt.figure(figsize=(6, 4))
    plt.hist(distances, bins=20, alpha=0.7)
    plt.axvline(distance_cutoff, linestyle="--")
    plt.title(f"Distance to neighbors within grouping '{bin_id}'")
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    print(
        f"Fraction of neighbors in grouping '{bin_id}' with distance <= {distance_cutoff}: "
        f"{frac_under_cutoff:.2%}"
    )


def _initialize_cleaning_df(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure DataFrame has required 'index' and 'keep' columns."""
    if "index" not in df.columns:
        df = df.with_row_index()
    if "keep" not in df.columns:
        df = df.with_columns(pl.lit(None).alias("keep"))
    return df


def _validate_cleaning_inputs(grouping_method: str, lsh_model: Dict[str, Any] | None) -> None:
    """Validate run_cleaning input parameters."""
    if grouping_method not in ["bin", "title"]:
        raise ValueError("grouping_method must be 'bin' or 'title'")
    if grouping_method == "bin" and lsh_model is None:
        raise ValueError("lsh_model is required when grouping_method='bin'")


def _find_neighbors_by_lsh(
    X_tfidf: "scipy.sparse.csr_matrix",
    idx: int,
    lsh_model: Dict[str, Any],
) -> pl.DataFrame:
    """Find neighbors using LSH-based search."""
    return get_nearest_neighbors(
        X_tfidf, X_tfidf[idx], lsh_model, max_search_radius=4
    )


def _find_neighbors_by_title(
    X_tfidf: "scipy.sparse.csr_matrix",
    df: pl.DataFrame,
    idx: int,
    title_value: Any,
) -> pl.DataFrame:
    """Find neighbors by matching title and computing cosine distances."""
    group_indices = df.filter(pl.col("title") == title_value)["index"].to_list()
    candidate_indices = [i for i in group_indices if i != idx]
    if not candidate_indices:
        return pl.DataFrame({"index": [], "distance": []})
    
    candidates = X_tfidf[candidate_indices]
    distances = pairwise_distances(candidates, X_tfidf[idx], metric="cosine").ravel()
    return pl.DataFrame(
        {"index": candidate_indices, "distance": distances}
    ).sort("distance", descending=False)


def _display_sample_info(
    df: pl.DataFrame,
    sample: pl.DataFrame,
    neighbor_rows: pl.DataFrame,
    neighbors: pl.DataFrame,
    group_label: str,
    group_id: Any,
    total_group_size: int,
    coverage: float,
) -> None:
    """Display information about the sampled page and its neighbors."""
    titles = neighbor_rows["title"].to_list()
    labels = neighbor_rows["label"].to_list()
    urls = neighbor_rows["url"].to_list()
    domains = [extract_domain(u) for u in urls if u]
    
    print_progress(df)
    print("")
    print(f"{group_label}: {group_id!r}")
    print(f"  - Num group members: {total_group_size}")
    print(f"  - Group coverage: {coverage:.2%} annotated")
    print("")
    print("-" * 12)
    print("Sampled page")
    print("-" * 12)
    print("")
    print(f"URL: {sample['url'][0]}")
    print(f"Title: {sample['title'][0]}")
    print(f"Num neighbors (<= cutoff): {neighbors.height}")
    print(f"  - Neighbor labels: {Counter(labels).most_common(2)}")
    print(f"  - Neighbor titles: {Counter(titles).most_common(10)}")
    print(f"  - Neighbor domains: {Counter(domains).most_common(10)}")
    print("")
    print("-" * 4)
    print("HTML")
    print("-" * 4)
    print("")
    # html = load_sample_html(sample["filename"][0])
    html = sample["html"][0]
    print(prettify_html(html)[:1000])
    print("")
    print("-" * 4)
    print("Text")
    print("-" * 4)
    print("")
    print(get_text_from_html(html)[:1000])


def _get_user_decision(
    df: pl.DataFrame,
    neighbor_ids: List[int],
    idx: int,
) -> tuple[pl.DataFrame, bool | None]:
    """
    Get user decision on whether to keep or remove the sample and neighbors.
    
    Returns:
        (updated_df, should_increment_spent)
        should_increment_spent is True/False for valid choices, None to force exit
    """
    choice = input("Keep (k), remove (r), skip (s): ").strip().lower()
    if choice == "k":
        return update_keep_column(df, neighbor_ids + [idx], True), True
    elif choice == "r":
        return update_keep_column(df, neighbor_ids, False), True
    elif choice == "s":
        return df, False
    else:
        # Invalid input, force exit
        return df, None


def run_cleaning(
    X_tfidf: "scipy.sparse.csr_matrix",
    df: pl.DataFrame,
    grouping_method: str = "bin",
    lsh_model: Dict[str, Any] | None = None,
    budget: int = 25,
    distance_cutoff: float = 0.2,
    coverage_threshold: float = 0.8,
    exclude_groups: List[Any] | None = None,
    max_samples_per_group: int = 3,
) -> pl.DataFrame:
    """
    Interactive cleaning loop for phishing/benign data.
    
    Args:
        X_tfidf: TF-IDF feature matrix
        df: DataFrame with columns including 'index', 'title', 'url', 'label', 'filename'
        grouping_method: Either "bin" (LSH-based) or "title" (title-based grouping)
        lsh_model: Required when grouping_method="bin", unused otherwise
        budget: Maximum number of samples to manually review
        distance_cutoff: Maximum distance to consider items as neighbors
        coverage_threshold: Fraction of group that must be annotated before moving on
        exclude_groups: List of group values to skip (e.g., ["", None] to skip empty titles)
        max_samples_per_group: Maximum samples to draw from each group (prevents infinite loops)
    
    Returns:
        DataFrame with 'keep' column updated based on user decisions
    """
    df = _initialize_cleaning_df(df)
    _validate_cleaning_inputs(grouping_method, lsh_model)
    
    group_col = grouping_method
    group_label = "Bin ID" if grouping_method == "bin" else "Title"
    
    # Process groups from largest to smallest
    group_counts = df.group_by(group_col).len().sort("len", descending=True)
    groups = group_counts[group_col].to_list()
    
    # Filter out excluded groups
    if exclude_groups is not None:
        groups = [g for g in groups if g not in exclude_groups]
    
    spent = 0
    
    for group_id in groups:
        total_group_size = df.filter(pl.col(group_col) == group_id).height
        if total_group_size == 0:
            continue
        
        group_samples = 0  # Track samples from this group
        
        while spent < budget and group_samples < max_samples_per_group:
            # Get unannotated rows in this group
            mask = (pl.col(group_col) == group_id) & pl.col("keep").is_null()
            candidates = df.filter(mask)
            
            num_unannotated = candidates.height
            if num_unannotated == 0:
                break
            
            # Check if we've reached coverage threshold
            coverage = (total_group_size - num_unannotated) / total_group_size
            if coverage >= coverage_threshold:
                print(
                    f"{group_label} {group_id!r}: coverage {coverage:.2%} >= "
                    f"threshold {coverage_threshold:.2%}, moving on."
                )
                break
            
            # Sample one row and find its neighbors
            sample = candidates.sample(n=1)
            idx = int(sample["index"][0])
            clear_output(wait=True)
            
            # Find neighbors based on grouping method
            if grouping_method == "bin":
                neighbors_all = _find_neighbors_by_lsh(X_tfidf, idx, lsh_model)
            else:
                neighbors_all = _find_neighbors_by_title(X_tfidf, df, idx, group_id)
            
            if neighbors_all.is_empty():
                print(f"No neighbors found for sample in {group_label} {group_id!r}, skipping.")
                continue
            
            # Filter neighbors by distance cutoff
            neighbors = neighbors_all.filter(pl.col("distance") <= distance_cutoff)
            neighbor_ids = neighbors["index"].to_list()
            if len(neighbor_ids) == 0:
                print(
                    f"No neighbors within distance_cutoff={distance_cutoff} "
                    f"for sample in {group_label} {group_id!r}, skipping."
                )
                continue
            
            # Display information and get user decision
            neighbor_rows = df.filter(pl.col("index").is_in(neighbor_ids))
            _display_sample_info(
                df, sample, neighbor_rows, neighbors,
                group_label, group_id, total_group_size, coverage
            )
            plot_distance_in_bin(
                neighbors_all, df, group_id, distance_cutoff, group_col
            )
            
            df, should_increment = _get_user_decision(df, neighbor_ids, idx)
            if should_increment is None:
                spent = budget  # Force exit
                break
            elif should_increment:
                spent += 1
                group_samples += 1
        
        # Check if we hit the per-group limit
        if group_samples >= max_samples_per_group:
            print(
                f"{group_label} {group_id!r}: reached max samples per group "
                f"({max_samples_per_group}), moving on."
            )
        
        if spent >= budget:
            break
    
    clear_output(wait=True)
    print("Budget exhausted or coverage thresholds met. Finalizing...")
    print_progress(df)
    return df

def copy_file(row: tuple, new_dir: Path) -> tuple[bool, str]:
    """Copy a single file to its destination."""
    filename, label, sha256 = row
    try:
        src = Path(filename)
        dst = Path(new_dir) / label / f"{sha256}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True, filename
    except Exception as e:
        return False, f"{filename}: {e}"
