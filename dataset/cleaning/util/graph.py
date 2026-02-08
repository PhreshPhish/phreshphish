import matplotlib as mpl
import matplotlib.dates as mdates
import seaborn as sns
import polars as pl
import pandas as pd
import numpy as np
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import tldextract
from tqdm import tqdm
from .stats import extract_html_features, extract_html_features_fast_progress
from .stats import extract_url_features, compute_feature_statistics


LABEL_PALETTE = {
    "benign": sns.color_palette()[0],
    "phish": sns.color_palette()[1],
}

SPLIT_ORDER = ["train", "test"]

sns.set_theme(font_scale=1.5)
sns.set_palette(list(LABEL_PALETTE.values()))


def _extract_domain(url: Optional[str]) -> Optional[str]:
    """Extract registered domain from URL."""
    if url is None:
        return None
    ext = tldextract.extract(url)
    return ext.domain + "." + ext.suffix if ext.suffix else ext.domain


def _process_url_chunk(url_chunk: list[Optional[str]]) -> list[dict]:
    """Process a chunk of URLs for parallel extraction."""
    return [extract_url_features(url) for url in url_chunk]


def _remove_outliers(
    df: pl.DataFrame, value_col: str, group_cols: list[str], quantile: float = 0.99
) -> pl.DataFrame:
    """Remove outliers beyond the specified quantile for each group."""
    return (
        df.with_columns(
            pl.col(value_col).quantile(quantile).over(group_cols).alias("_q")
        )
        .filter(pl.col(value_col) <= pl.col("_q"))
        .drop("_q")
    )


def _prepare_melted_data(
    df: pl.DataFrame, feature_cols: list[str], remove_outliers: bool = True
) -> pl.DataFrame:
    """Melt DataFrame and optionally remove outliers for plotting."""
    melted = df.select(["label"] + feature_cols).unpivot(
        index="label", on=feature_cols, variable_name="Metric", value_name="Value"
    )

    if remove_outliers:
        melted = _remove_outliers(melted, "Value", ["label", "Metric"])

    return melted


def _create_stripplot(
    melted_pd: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[int, int] = (14, 6),
) -> mpl.figure.Figure:
    """Create a standardized stripplot."""
    fig, ax = mpl.pyplot.subplots(figsize=figsize)
    sns.stripplot(
        data=melted_pd,
        x="Metric",
        y="Value",
        hue="label",
        dodge=True,
        jitter=0.2,
        alpha=0.05,
        size=1,
        linewidth=0,
        palette=LABEL_PALETTE,
        ax=ax,
    )

    labels = list(melted_pd["label"].unique())
    handles = [
        mpl.patches.Patch(
            facecolor=LABEL_PALETTE[label], edgecolor="white", label=label
        )
        for label in labels
    ]

    ax.legend(handles=handles, title="Label", loc="upper left")
    ax.grid(True, which="major", axis="both")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    mpl.pyplot.tight_layout()

    return fig


def _extract_url_features_parallel(
    urls: list[Optional[str]],
    n_jobs: Optional[int] = None,
    chunk_size: int = 1000,
    parallel_threshold: int = 50000,
) -> list[dict]:
    """
    Extract URL features in parallel for maximum performance.

    For small datasets (< parallel_threshold), uses serial processing to avoid overhead.
    For large datasets, uses parallel processing across all cores.

    Args:
        urls: List of URLs to process
        n_jobs: Number of parallel workers (defaults to all available cores)
        chunk_size: Number of URLs per chunk for better progress tracking
        parallel_threshold: Minimum number of rows to use parallel processing

    Returns:
        List of feature dictionaries
    """
    n = len(urls)

    # For small datasets, serial processing is faster (no process overhead)
    if n < parallel_threshold:
        results = []
        for url in tqdm(urls, desc="Extracting URL features", unit="url"):
            results.append(extract_url_features(url))
        return results

    # For large datasets, use parallel processing
    if n_jobs is None:
        n_jobs = cpu_count()  # Use ALL cores for maximum performance

    # Process in chunks for better progress tracking
    chunks = [urls[i : i + chunk_size] for i in range(0, len(urls), chunk_size)]

    # Track chunk index to preserve order
    chunk_results = [None] * len(chunks)
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_idx = {
            executor.submit(_process_url_chunk, chunk): idx
            for idx, chunk in enumerate(chunks)
        }

        with tqdm(
            total=len(future_to_idx), desc="Extracting URL features", unit="chunk"
        ) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                chunk_results[idx] = future.result()
                pbar.update(1)

    # Flatten results in original order
    results = []
    for chunk_result in chunk_results:
        results.extend(chunk_result)

    return results


def class_balance(
    df: pl.DataFrame, by="split"
) -> tuple[pl.DataFrame, mpl.figure.Figure]:
    """Generate a class balance plot and table for the given DataFrame.

    Args:
        df (pl.DataFrame): Input DataFrame containing a 'label' column and a grouping column.
        by (str): Column name to group by (default is "split").
    """
    class_bal = (
        df.group_by([by, "label"])
        .agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / pl.col("n").sum().over(by)).alias("frac"))
        .sort([by, "n"], descending=[False, True])
    )

    fig, ax = mpl.pyplot.subplots(figsize=(8, 6))
    sns.barplot(
        data=class_bal,
        x=by,
        y="n",
        hue="label",
        order=SPLIT_ORDER if by == "split" else None,
        palette=LABEL_PALETTE,
        ax=ax,
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.0f",
            label_type="center",
            color="white",
            weight="bold",
        )

    ax.set_ylabel("Count")
    ax.set_xlabel(by.capitalize())
    ax.set_title("Class Balance by " + by.capitalize())
    ax.legend(title="Label", loc="upper right")
    mpl.pyplot.tight_layout()
    return class_bal, fig


def plot_topk_domains(
    df: pl.DataFrame, k: int = 10, by_label: bool = False
) -> tuple[pl.DataFrame, mpl.figure.Figure]:
    """Generate a plot of the top K domains by frequency.

    Args:
        df (pl.DataFrame): Input DataFrame containing a 'url' column.
        k (int): Number of top domains to display (default is 10).
        by_label (bool): Whether to split counts by label (default is False).
    """
    df = df.with_columns(
        pl.col("url")
        .map_elements(_extract_domain, return_dtype=pl.Utf8)
        .alias("domain")
    )
    
    group_bys = ["domain", "label"] if by_label else ["domain"]
    domain_counts = (
        df.group_by(group_bys)
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )

    fig, ax = mpl.pyplot.subplots(figsize=(10, 6))
    sns.barplot(
        data=domain_counts.head(k),
        x="n",
        y="domain",
        hue="label" if by_label else None,
        palette=LABEL_PALETTE if by_label else None,
        ax=ax,
    )

    ax.set_xscale("log")
    ax.grid(True, which="major", axis="both")
    ax.grid(True, which="minor", axis="x", color="gray", linestyle="--", alpha=0.3)

    ax.set_xlabel("Count")
    ax.set_ylabel("Domain")
    ax.set_title("Domain Distribution (Top " + str(k) + ")")
    ax.legend(title="Label", loc="lower right") if by_label else None
    mpl.pyplot.tight_layout()
    return domain_counts.head(k), fig


def plot_daily_collection(df: pl.DataFrame) -> tuple[pl.DataFrame, mpl.figure.Figure]:

    daily_counts = (
        df.group_by(["date", "split", "label"]).len().rename({"len": "count"})
    )

    grouped = (
        daily_counts.pivot(
            values="count",
            index="date",
            on=["split", "label"],
            aggregate_function="first",
        )
        .fill_null(0)
        .sort("date")
    )

    # Labels to plot
    labels = df["label"].unique().to_list()

    # Min test day per label
    test_start_dates = {}
    for label in labels:
        s = daily_counts.filter(
            (pl.col("split") == "test") & (pl.col("label") == label)
        ).select(pl.col("date").min())
        test_start_dates[label] = s.item() if s.height > 0 else None

    x_dates = grouped["date"].to_list()
    x_indices = list(range(len(x_dates)))

    split_colors = {"train": "steelblue", "test": "darkorange"}
    bar_width = 1.0

    fig, axes = mpl.pyplot.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)

    for i, label in enumerate(labels):
        ax = axes[i]

        for j, split in enumerate(["train", "test"]):
            # Polars pivot produced columns like: {"train","phish"}
            col = f'{{"{split}","{label}"}}'
            if col in grouped.columns:
                values = grouped[col].to_numpy()
                offset = (j - 0.5) * bar_width
                positions = [x + offset for x in x_indices]
                ax.bar(
                    positions,
                    values,
                    width=bar_width,
                    label=split,
                    color=split_colors[split],
                    alpha=0.9,
                )

        # Add test split line
        test_date = test_start_dates[label]
        if test_date is not None and test_date in x_dates:
            split_idx = x_dates.index(test_date)
            ax.axvline(split_idx, color="red", linestyle="--", linewidth=1)

        # Axis formatting
        ax.set_title(f"{label.capitalize()} — Train/Test Daily Counts")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        # ax.grid(True, axis="y")
        ax.grid(True, which="both", axis="both")

        # Format x-axis
        tick_freq = max(1, len(x_dates) // 10)
        ax.set_xticks(x_indices[::tick_freq])
        ax.set_xticklabels(
            [d.strftime("%Y-%m-%d") for d in x_dates[::tick_freq]],
            rotation=45,
            ha="right",
        )

    axes[-1].set_xlabel("Date")

    # Add a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, title="Split", loc="upper right", bbox_to_anchor=(0.98, 0.98)
    )

    mpl.pyplot.tight_layout()
    return daily_counts, fig


def plot_url_characteristics(
    df: pl.DataFrame,
    n_jobs: Optional[int] = None,
    parallel_threshold: int = 50000,
) -> tuple[pl.DataFrame, mpl.figure.Figure]:
    """
    Plot URL characteristics with intelligent parallelization.

    For small datasets (< parallel_threshold), uses serial processing to avoid overhead.
    For large datasets, uses parallel processing across all cores.

    Args:
        df: DataFrame with 'url' and 'label' columns
        n_jobs: Number of parallel workers (None = auto-detect based on dataset size)
        parallel_threshold: Minimum number of rows to use parallel processing (default: 50,000)
    
    Returns:
        Tuple of (summary statistics DataFrame, figure)
    """
    n_rows = len(df)
    print(f"Extracting URL features from {n_rows:,} rows...")

    # Extract features with intelligent parallelization
    urls = df["url"].to_list()
    features_list = _extract_url_features_parallel(
        urls, n_jobs=n_jobs, parallel_threshold=parallel_threshold
    )

    # Create features DataFrame
    features_df = pl.DataFrame(features_list)

    # Combine with original DataFrame
    features_df = df.select(["label"]).hstack(features_df)

    feature_cols = ["url_len", "domain_len", "subdomain_len", "path_len"]
    features_df = features_df.drop_nulls(subset=feature_cols + ["label"])

    print("Preparing plot data...")
    # Prepare melted data with outlier removal
    melted = _prepare_melted_data(features_df, feature_cols, remove_outliers=True)

    # Convert to pandas and create categorical for proper ordering
    melted_pd = melted.to_pandas()
    melted_pd["Metric"] = pd.Categorical(melted_pd["Metric"], ordered=True)
    melted_pd.rename(columns={"Value": "Length"}, inplace=True)

    print("Creating plot...")
    # Create stripplot
    fig, ax = mpl.pyplot.subplots(figsize=(14, 6))
    sns.stripplot(
        data=melted_pd,
        x="Metric",
        y="Length",
        hue="label",
        dodge=True,
        jitter=0.2,
        alpha=0.05,
        size=1,
        linewidth=0,
        palette=LABEL_PALETTE,
        ax=ax,
    )

    labels = list(melted_pd["label"].unique())
    handles = [
        mpl.patches.Patch(
            facecolor=LABEL_PALETTE[label], edgecolor="white", label=label
        )
        for label in labels
    ]

    ax.legend(handles=handles, title="Label", loc="upper left")
    ax.grid(True, which="major", axis="both")
    ax.set_title("URL Characteristics")
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Length (Characters)")
    mpl.pyplot.tight_layout()

    # Compute summary statistics
    print("Computing summary statistics...")
    summary_stats = compute_feature_statistics(
        features_df, feature_cols
    )

    return summary_stats, fig


def plot_html_characteristics(
    df: pl.DataFrame,
    n_jobs: Optional[int] = None,
    use_fast_progress: bool = True,
    parallel_threshold: int = 50000,
) -> tuple[pl.DataFrame, mpl.figure.Figure]:
    """
    Plot HTML characteristics with parallel processing and progress tracking.

    For small datasets (< parallel_threshold), uses serial processing to avoid overhead.
    For large datasets, uses parallel processing across all cores.

    Args:
        df: DataFrame with 'html' and 'label' columns
        n_jobs: Number of parallel workers (None = use all cores)
        use_fast_progress: Use the fast progress version for better tracking on large datasets
        parallel_threshold: Minimum number of rows to use parallel processing (default: 50,000)
    
    Returns:
        Tuple of (summary statistics DataFrame, figure)
    """
    feature_cols = [
        "total_html_len",
        "head_len_sum",
        "img_len_sum",
        "script_len_sum",
    ]

    n_rows = len(df)

    # For small datasets, use serial or minimal parallelism to avoid overhead
    if n_rows < parallel_threshold:
        if n_jobs is None:
            n_jobs = min(4, cpu_count())  # Use only a few cores for small datasets
        print(
            f"Extracting HTML features from {n_rows:,} rows (small dataset, using {n_jobs} workers)..."
        )
        html_features = extract_html_features(df["html"], n_jobs=n_jobs)
    else:
        # For large datasets, use all cores
        if n_jobs is None:
            n_jobs = cpu_count()  # Use ALL cores for maximum performance

        print(
            f"Extracting HTML features from {n_rows:,} rows (large dataset, using {n_jobs} workers)..."
        )

        if use_fast_progress:
            # Use fast progress version for better progress tracking on large datasets
            html_features = extract_html_features_fast_progress(
                df["html"],
                n_jobs=n_jobs,
                chunk_rows=2048,  # Optimize chunk size for large datasets
            )
        else:
            html_features = extract_html_features(df["html"], n_jobs=n_jobs)

    df_with_features = df.hstack(html_features)
    df_with_features = df_with_features.drop_nulls(subset=feature_cols + ["label"])

    print("Preparing plot data...")
    # Prepare melted data with outlier removal
    melted = _prepare_melted_data(df_with_features, feature_cols, remove_outliers=True)

    # Convert to pandas and create categorical for proper ordering
    melted_pd = melted.to_pandas()
    melted_pd["Metric"] = pd.Categorical(melted_pd["Metric"], ordered=True)

    print("Creating plot...")
    # Create the plot using helper function
    fig = _create_stripplot(
        melted_pd, title="HTML Characteristics", xlabel="Quantity", ylabel="Value"
    )

    # Compute summary statistics
    print("Computing summary statistics...")
    summary_stats = compute_feature_statistics(
        df_with_features, feature_cols
    )

    return summary_stats, fig


def plot_topk_languages(
    df: pl.DataFrame, k: int = 10, by_label: bool = False
) -> tuple[pl.DataFrame, mpl.figure.Figure]:
    """Generate a plot of the top K languages by frequency.

    Args:
        df (pl.DataFrame): Input DataFrame containing a 'lang' column.
        k (int): Number of top languages to display (default is 10).
        by_label (bool): Whether to split counts by label (default is False).
    """
    group_bys = ["lang", "label"] if by_label else ["lang"]
    lang_counts = (
        df.group_by(group_bys)
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )

    fig, ax = mpl.pyplot.subplots(figsize=(10, 6))
    sns.barplot(
        data=lang_counts.head(k),
        x="n",
        y="lang",
        hue="label" if by_label else None,
        palette=LABEL_PALETTE if by_label else None,
        ax=ax,
    )

    ax.set_xscale("log")
    ax.grid(True, which="major", axis="both")
    ax.grid(True, which="minor", axis="x", color="gray", linestyle="--", alpha=0.3)

    ax.set_xlabel("Count")
    ax.set_ylabel("Language")
    ax.set_title("Language Distribution (Top " + str(k) + ")")
    ax.legend(title="Label", loc="lower right") if by_label else None
    mpl.pyplot.tight_layout()
    return lang_counts.head(k), fig


def plot_topk_targets(
    df: pl.DataFrame, k: int = 10, by_label: bool = False
) -> tuple[pl.DataFrame, mpl.figure.Figure]:
    """Generate a plot of the top K phishing targets by frequency.

    Args:
        df (pl.DataFrame): Input DataFrame containing a 'target' column.
        k (int): Number of top targets to display (default is 10).
        by_label (bool): Whether to split counts by label (default is False).
    """
    group_bys = ["target", "label"] if by_label else ["target"]
    target_counts = (
        df.group_by(group_bys)
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )

    fig, ax = mpl.pyplot.subplots(figsize=(10, 6))
    sns.barplot(
        data=target_counts.head(k),
        x="n",
        y="target",
        hue="label" if by_label else None,
        palette=LABEL_PALETTE if by_label else None,
        ax=ax,
    )

    ax.set_xscale("log")
    ax.grid(True, which="major", axis="both")
    ax.grid(True, which="minor", axis="x", color="gray", linestyle="--", alpha=0.3)

    ax.set_xlabel("Count")
    ax.set_ylabel("Target")
    ax.set_title("Target Distribution (Top " + str(k) + ")")
    ax.legend(title="Label", loc="lower right") if by_label else None
    mpl.pyplot.tight_layout()
    return target_counts.head(k), fig


def plot_daily_targets(
    df: pl.DataFrame, n: int = 9
) -> tuple[pl.DataFrame, mpl.figure.Figure]:
    """
    Plot top N phishing targets over time as a stacked bar chart.

    Args:
        df: DataFrame with 'label', 'target', and 'date' columns
        n: Number of top targets to show individually (default: 9)

    Returns:
        Tuple of (daily_counts DataFrame, matplotlib Figure)
    """

    # only looking at phishing rows with known targets
    df_phish = df.filter(
        (pl.col("label") == "phish")
        & (pl.col("target") != "unknown")
        & (pl.col("target") != "other")
    )

    # 2) Find the top-N targets by raw count
    top_targets = (
        df_phish.group_by("target")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(n)
        .select("target")
        .to_series()
        .to_list()
    )

    # 3) Create a new column that maps anything not in top_targets to "other"
    df_phish = df_phish.with_columns(
        pl.when(pl.col("target").is_in(top_targets))
        .then(pl.col("target"))
        .otherwise(pl.lit("other"))
        .alias("target_group")
    )

    # 4) Group by day + target_group to get daily counts
    daily_counts = (
        df_phish.group_by(["date", "target_group"])
        .agg(pl.len().alias("count"))
        .sort("date")
    )

    # 5) Pivot to get targets as columns
    daily_pivot = daily_counts.pivot(
        values="count", index="date", on="target_group", aggregate_function="sum"
    ).fill_null(0)

    # 6) Reorder columns so top targets come first, then 'other' last
    # Calculate total counts for each target across all days
    target_cols = [col for col in daily_pivot.columns if col != "date"]
    col_totals = {col: daily_pivot[col].sum() for col in target_cols}

    # Sort by total count descending, but ensure 'other' is last
    sorted_targets = sorted(
        [t for t in target_cols if t != "other"],
        key=lambda x: col_totals[x],
        reverse=True,
    )
    if "other" in target_cols:
        sorted_targets.append("other")

    # 7) Create the stacked bar chart with distinct colors
    fig, ax = mpl.pyplot.subplots(figsize=(12, 6), dpi=300)

    # Generate distinct colors for each target
    num_targets = len(sorted_targets)
    if num_targets <= 10:
        colors = sns.color_palette("tab10", n_colors=num_targets)
    elif num_targets <= 20:
        colors = sns.color_palette("tab20", n_colors=num_targets)
    else:
        colors = sns.color_palette("husl", n_colors=num_targets)

    # Convert to pandas for easier plotting
    dates = daily_pivot["date"].to_list()
    bottom = np.zeros(len(dates))
    bar_width = 0.8  # ~1 day

    for idx, target in enumerate(sorted_targets):
        vals = daily_pivot[target].to_numpy()
        ax.bar(
            dates,
            vals,
            bottom=bottom,
            width=bar_width,
            align="center",
            label=target,
            color=colors[idx],
        )
        bottom += vals

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(
        axis="x",
        which="major",
        bottom=True,
        rotation=45,
        labelsize=12,
        length=6,
        width=1,
    )
    ax.tick_params(
        axis="x", which="minor", bottom=True, rotation=0, length=3, width=0.5
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.set_title("Top Daily Phishing Targets")
    ax.legend(loc="best", ncol=3, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)

    mpl.pyplot.tight_layout()

    return daily_pivot, fig


def plot_domain_wordcloud(df: pl.DataFrame) -> tuple[pl.DataFrame, mpl.figure.Figure]:
    from wordcloud import WordCloud

    df = df.with_columns(
        (
            pl.col("url")
            .map_elements(
                lambda u: _extract_domain(u),
                return_dtype=pl.String,
            )
            .alias("domain")
        )
    )

    counts = (
        df.filter(pl.col("label") == "phish")["domain"]
        .value_counts()
        .sort(by="count", descending=True)
    )

    freqs = dict(zip(counts["domain"].to_list(), counts["count"].to_list()))

    wc = WordCloud(
        width=900,
        height=500,
        background_color="white",
        max_words=100,
        collocations=False,
    ).generate_from_frequencies(freqs)

    fig, ax = mpl.pyplot.subplots(figsize=(9, 5), dpi=300)
    ax.set_title("Top Phishing Domains")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    mpl.pyplot.tight_layout()

    return counts, fig
