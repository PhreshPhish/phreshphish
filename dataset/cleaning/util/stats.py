from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Iterable, Optional, Tuple

import polars as pl
from lxml import etree, html as lxml_html
from urllib.parse import urlparse
import tldextract
from tqdm import tqdm


def compute_feature_statistics(
    df: pl.DataFrame,
    feature_cols: list[str],
    label_col: str = "label"
) -> pl.DataFrame:
    """
    Compute summary statistics (min, max, 25th, 50th, 75th percentiles) 
    for specified features grouped by label.
    
    Args:
        df: DataFrame containing features and labels
        feature_cols: List of feature column names to compute statistics for
        label_col: Name of the label column (default: "label")
    
    Returns:
        DataFrame with statistics for each feature grouped by label
    """
    stats_list = []
    
    for label in df[label_col].unique().sort():
        label_df = df.filter(pl.col(label_col) == label)
        
        for feature in feature_cols:
            # Compute statistics, excluding nulls
            feature_data = label_df[feature].drop_nulls()
            
            if len(feature_data) > 0:
                stats_list.append({
                    "label": label,
                    "feature": feature,
                    "min": feature_data.min(),
                    "p25": feature_data.quantile(0.25),
                    "p50": feature_data.quantile(0.50),
                    "p75": feature_data.quantile(0.75),
                    "max": feature_data.max(),
                    "count": len(feature_data),
                })
            else:
                # Handle case where all values are null
                stats_list.append({
                    "label": label,
                    "feature": feature,
                    "min": None,
                    "p25": None,
                    "p50": None,
                    "p75": None,
                    "max": None,
                    "count": 0,
                })
    
    return pl.DataFrame(stats_list)


def _html_feats_worker(
    s: Optional[str],
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Returns: (head_len_sum, img_len_sum, script_len_sum)
    """
    if s is None:
        return (None, None, None)

    try:
        root = lxml_html.fromstring(s)

        # serialize each matched element once; len() counts characters
        # (encoding="unicode" returns str, avoiding bytes->str overhead)
        def sum_serialized(xpath: str) -> int:
            nodes = root.xpath(xpath)
            # etree.tostring(node, encoding="unicode") includes the tag itself + contents
            return sum(len(etree.tostring(n, encoding="unicode")) for n in nodes)

        head_len_sum = sum_serialized(".//head")
        img_len_sum = sum_serialized(".//img")
        script_len_sum = sum_serialized(".//script")

        return (head_len_sum, img_len_sum, script_len_sum)
    except Exception:
        return (None, None, None)


def extract_url_features(url):
    try:
        parsed = urlparse(url)
        ext = tldextract.extract(url)
        return {
            "url_len": len(url),
            "domain_len": len(ext.domain + "." + ext.suffix),
            "subdomain_len": len(ext.subdomain),
            "path_len": len(parsed.path),
        }
    except Exception:
        return {
            "url_len": None,
            "domain_len": None,
            "subdomain_len": None,
            "path_len": None,
        }


def _calculate_html_lengths(html_iterable: Iterable[Optional[str]]) -> list[Optional[int]]:
    """Calculate the byte length of each HTML string."""
    lengths = []
    for html in html_iterable:
        if html is None:
            lengths.append(None)
        else:
            lengths.append(len(html.encode('utf-8')) if isinstance(html, str) else None)
    return lengths


def _create_features_dataframe(
    results: list[Tuple[Optional[int], Optional[int], Optional[int]]],
    total_html_lengths: list[Optional[int]]
) -> pl.DataFrame:
    """Create a DataFrame from the extracted HTML features."""
    head_len_sum, img_len_sum, script_len_sum = (
        zip(*results) if results else ([], [], [])
    )
    
    return pl.DataFrame(
        {
            "total_html_len": total_html_lengths,
            "head_len_sum": head_len_sum,
            "img_len_sum": img_len_sum,
            "script_len_sum": script_len_sum,
        },
        schema={
            "total_html_len": pl.Int64,
            "head_len_sum": pl.Int64,
            "img_len_sum": pl.Int64,
            "script_len_sum": pl.Int64,
        },
    )


def _process_html_parallel(
    html_list: list[Optional[str]], 
    n_jobs: int,
    chunksize: Optional[int] = None
) -> list[Tuple[Optional[int], Optional[int], Optional[int]]]:
    """Process HTML strings in parallel using ProcessPoolExecutor."""
    if chunksize is None:
        chunksize = 256 if len(html_list) >= 50_000 else 4
    
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        it = ex.map(_html_feats_worker, html_list, chunksize=chunksize)
        results = list(tqdm(it, total=len(html_list), desc="Extracting HTML features"))
    
    return results


def extract_html_features(
    html_iterable: Iterable[Optional[str]], 
    n_jobs: Optional[int] = None
) -> pl.DataFrame:
    """
    Extract HTML features from an iterable of HTML strings.
    
    Args:
        html_iterable: An iterable of HTML strings (or None values)
        n_jobs: Number of parallel workers. If None, uses (cpu_count - 1)
    
    Returns:
        DataFrame with columns: total_html_len, head_len_sum, img_len_sum, script_len_sum
    """
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    
    # Convert iterable to list to support multiple iterations
    html_list = list(html_iterable)
    
    # Calculate total HTML lengths
    total_html_lengths = _calculate_html_lengths(html_list)
    
    # Process HTML in parallel
    results = _process_html_parallel(html_list, n_jobs)
    
    # Create and return DataFrame
    return _create_features_dataframe(results, total_html_lengths)

def _worker_chunk(html_chunk: list[str | None]):
    """Process a chunk of HTML strings."""
    out = []
    for s in html_chunk:
        out.append(_html_feats_worker(s))
    return out


def _process_html_chunked(
    html_list: list[Optional[str]], 
    n_jobs: int,
    chunk_rows: int
) -> list[Tuple[Optional[int], Optional[int], Optional[int]]]:
    """Process HTML strings in parallel with manual chunking for better progress tracking."""
    # Partition into chunks
    chunks = [html_list[i:i+chunk_rows] for i in range(0, len(html_list), chunk_rows)]
    
    results = [None] * len(chunks)
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(_worker_chunk, ch): idx for idx, ch in enumerate(chunks)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting HTML features"):
            idx = futures[fut]
            results[idx] = fut.result()
    
    # Flatten in original order
    flat = [t for chunk in results for t in chunk]
    return flat


def extract_html_features_fast_progress(
    html_iterable: Iterable[Optional[str]], 
    n_jobs: Optional[int] = None, 
    chunk_rows: int = 2048
) -> pl.DataFrame:
    """
    Extract HTML features from an iterable with better progress tracking.
    
    This version provides more granular progress updates by processing in larger chunks.
    
    Args:
        html_iterable: An iterable of HTML strings (or None values)
        n_jobs: Number of parallel workers. If None, uses (cpu_count - 1)
        chunk_rows: Number of rows per chunk for processing
    
    Returns:
        DataFrame with columns: total_html_len, head_len_sum, img_len_sum, script_len_sum
    """
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    
    # Convert iterable to list
    html_list = list(html_iterable)
    
    # Calculate total HTML lengths
    total_html_lengths = _calculate_html_lengths(html_list)
    
    # Process HTML in parallel with chunking
    results = _process_html_chunked(html_list, n_jobs, chunk_rows)
    
    # Create and return DataFrame
    return _create_features_dataframe(results, total_html_lengths)