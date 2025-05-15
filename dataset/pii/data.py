import polars as pl
from urllib.parse import urlparse
from datasets import Dataset
from torch.utils.data import DataLoader


def get_candidates(url: str) -> list[str]:
    """Parses a URL and returns a list of name-like candidate strings."""
    path = urlparse(url).path
    candidates = []
    for part in path.split("/"):
        tokens = [
            token for token in part.replace("-", "_").split("_") if token.isalpha()
        ]
        if len(tokens) >= 2:
            candidate = " ".join(t.capitalize() for t in tokens)
            candidates.append(candidate)
    return candidates


class PhreshPhishDataset:
    def __init__(self, path: str):
        print("Loading dataset...", end="")
        df = pl.read_ipc(path).filter(pl.col('label') == -1).select(["sha256", "url"]) # only care about customer data
        print("done")

        # Build a list of records, one per candidate
        records = []
        for row in df.iter_rows(named=True):
            sha256 = row["sha256"]
            url = row["url"]
            candidates = get_candidates(url)
            for candidate in candidates:
                records.append({
                    "sha256": str(sha256),
                    "url": str(url),
                    "candidate": str(candidate)
                })

        self.dataset = Dataset.from_list(records)

    def get_hf_dataset(self) -> Dataset:
        """Return the processed Hugging Face dataset."""
        return self.dataset

    def get_dataloader(self, batch_size=32, shuffle=False) -> DataLoader:
        """Return a PyTorch DataLoader over candidate strings."""
        candidates = [row["candidate"] for row in self.dataset]
        return DataLoader(
            candidates,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda x: x  # returns list[str]
        )