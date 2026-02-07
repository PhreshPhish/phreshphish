"""
Optimized LinearSVC training for large datasets.

Uses streaming data loading + limited vocabulary to fit in memory,
then trains LinearSVC on the full dataset at once (not incremental).

This replicates trainSVCmisc.py behavior but handles larger datasets.
"""

import csv
import os
import gc
import sys
import pickle
import logging
import numpy as np
import joblib
from datetime import datetime
from os.path import join, exists
from collections import defaultdict
from typing import Iterator, Tuple, List, Dict, Optional
from tqdm import tqdm
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, hinge_loss
import matplotlib.pyplot as plt

# Increase CSV field size limit
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2147483647)


class StreamingVectorizer:
    """
    Memory-efficient vectorizer that:
    1. Builds vocabulary by streaming through data (PASS 1)
    2. Transforms data in batches (PASS 2)
    """

    def __init__(self, max_features: int = 500000):
        self.max_features = max_features
        self.feature_to_idx: Dict[str, int] = {}
        self.feature_counts: Dict[str, int] = defaultdict(int)
        self.n_features = 0
        self.is_fitted = False

    def partial_fit_vocabulary(self, features_batch: List[Dict[str, float]]):
        """Update vocabulary with a batch of samples."""
        for features in features_batch:
            for feat_name in features.keys():
                if feat_name == 'sha256':
                    continue
                self.feature_counts[feat_name] += 1
                if feat_name not in self.feature_to_idx:
                    if len(self.feature_to_idx) < self.max_features:
                        self.feature_to_idx[feat_name] = len(self.feature_to_idx)

        self.n_features = len(self.feature_to_idx)

    def finalize_vocabulary(self):
        """Finalize vocabulary after all partial_fit calls."""
        self.is_fitted = True
        logging.info(f"Vocabulary finalized with {self.n_features} features")

    def transform_batch(self, features_batch: List[Dict[str, float]]) -> csr_matrix:
        """Transform a batch of feature dictionaries to sparse matrix."""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")

        rows, cols, data = [], [], []

        for row_idx, features in enumerate(features_batch):
            for feat_name, feat_val in features.items():
                if feat_name == 'sha256':
                    continue
                if feat_name not in self.feature_to_idx:
                    continue
                if feat_val is not None:
                    rows.append(row_idx)
                    cols.append(self.feature_to_idx[feat_name])
                    data.append(float(feat_val))

        # Use int64 indices to handle large datasets (>2B non-zeros)
        mat = csr_matrix(
            (np.array(data, dtype=np.float32),
             (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
            shape=(len(features_batch), self.n_features),
            dtype=np.float32
        )
        # Ensure indices are int64 for large matrices
        mat.indices = mat.indices.astype(np.int64)
        mat.indptr = mat.indptr.astype(np.int64)
        return mat

    def save(self, path: str):
        """Save vectorizer state."""
        state = {
            'max_features': self.max_features,
            'feature_to_idx': self.feature_to_idx,
            'n_features': self.n_features,
            'is_fitted': self.is_fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> 'StreamingVectorizer':
        """Load vectorizer from file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        vec = cls(max_features=state['max_features'])
        vec.feature_to_idx = state['feature_to_idx']
        vec.n_features = state['n_features']
        vec.is_fitted = state['is_fitted']
        return vec


def stream_feat_file(feat_file: str, batch_size: int = 10000) -> Iterator[Tuple[List[int], List[Dict], List[str]]]:
    """
    Stream a .feat file yielding batches of (labels, features, sha256s).
    Never loads entire file into memory.
    """
    labels_batch = []
    features_batch = []
    sha256_batch = []

    with open(feat_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                parts = line.split(' ')
                if len(parts) < 2:
                    continue

                label = int(parts[0])
                sha256 = parts[1] if len(parts) > 1 else ""

                features = {}
                for feat_str in parts[2:]:
                    if ':' in feat_str:
                        feat_name, feat_val = feat_str.split(':', 1)
                        try:
                            features[feat_name] = float(feat_val)
                        except ValueError:
                            continue

                labels_batch.append(label)
                features_batch.append(features)
                sha256_batch.append(sha256)

                if len(labels_batch) >= batch_size:
                    yield labels_batch, features_batch, sha256_batch
                    labels_batch = []
                    features_batch = []
                    sha256_batch = []

            except Exception as e:
                if line_num % 100000 == 0:
                    logging.warning(f"Error parsing line {line_num}: {e}")
                continue

    # Yield remaining
    if labels_batch:
        yield labels_batch, features_batch, sha256_batch


class LinearSVCTrainer:
    """
    Memory-efficient LinearSVC trainer.

    Strategy:
    1. PASS 1: Stream through data to build vocabulary
    2. PASS 2: Transform and save batches to disk as .npz files
    3. PASS 3: Load all batches, combine, train LinearSVC
    """

    def __init__(
        self,
        max_features: int = 5000000,
        batch_size: int = 10000,
        C: float = 1.0,
        max_iter: int = 100000,
        tol: float = 1e-3,
        class_weight: str = 'balanced',
        random_state: int = 42,
        test_size: float = 0.1
    ):
        self.max_features = max_features
        self.batch_size = batch_size
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.random_state = random_state
        self.test_size = test_size

        self.vectorizer = StreamingVectorizer(max_features=max_features)
        self.model = None

    def train(self, feat_files: List[str], output_dir: str):
        """Main training pipeline."""
        os.makedirs(output_dir, exist_ok=True)
        for subdir in ['models', 'batches', 'plots', 'metrics']:
            os.makedirs(join(output_dir, subdir), exist_ok=True)

        # Setup logging
        log_file = join(output_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ],
            force=True
        )

        logging.info("=" * 60)
        logging.info("LinearSVC Training Pipeline")
        logging.info("=" * 60)
        logging.info(f"Max features: {self.max_features}")
        logging.info(f"Batch size: {self.batch_size}")
        logging.info(f"C: {self.C}")
        logging.info(f"Test size: {self.test_size}")

        # Check if batches already exist (resume from PASS 3)
        batches_dir = join(output_dir, 'batches')
        vectorizer_path = join(output_dir, 'models', 'vectorizer.pkl')
        existing_batches = [f for f in os.listdir(batches_dir) if f.startswith('batch_') and f.endswith('.npz')] if exists(batches_dir) else []

        if existing_batches and exists(vectorizer_path):
            logging.info(f"Found {len(existing_batches)} existing batch files. Skipping PASS 1 & 2.")
            logging.info("Loading existing vectorizer...")
            self.vectorizer = StreamingVectorizer.load(vectorizer_path)

            # Get batch and label file paths
            batch_files = sorted([join(batches_dir, f) for f in existing_batches],
                                key=lambda x: int(x.split('_')[-1].replace('.npz', '')))
            label_files = [f.replace('batch_', 'labels_').replace('.npz', '.npy') for f in batch_files]
        else:
            # PASS 1: Build vocabulary
            logging.info("=" * 60)
            logging.info("PASS 1: Building vocabulary")
            logging.info("=" * 60)

            total_samples = 0
            class_counts = defaultdict(int)

            for feat_file in feat_files:
                logging.info(f"Processing vocabulary from: {feat_file}")
                file_samples = 0

                for labels, features, _ in tqdm(
                    stream_feat_file(feat_file, self.batch_size),
                    desc=f"Vocab: {os.path.basename(feat_file)}"
                ):
                    self.vectorizer.partial_fit_vocabulary(features)
                    for label in labels:
                        class_counts[label] += 1
                    file_samples += len(labels)

                total_samples += file_samples
                logging.info(f"  Processed {file_samples} samples, vocabulary size: {self.vectorizer.n_features}")

            self.vectorizer.finalize_vocabulary()
            logging.info(f"Total samples: {total_samples}")
            logging.info(f"Class distribution: {dict(class_counts)}")
            logging.info(f"Final vocabulary size: {self.vectorizer.n_features}")

            # Save vectorizer
            self.vectorizer.save(join(output_dir, 'models', 'vectorizer.pkl'))

            # PASS 2: Transform and save batches
            logging.info("=" * 60)
            logging.info("PASS 2: Transforming data to sparse matrices")
            logging.info("=" * 60)

            batch_files = []
            label_files = []
            batch_idx = 0

            for feat_file in feat_files:
                logging.info(f"Transforming: {feat_file}")

                for labels, features, _ in tqdm(
                    stream_feat_file(feat_file, self.batch_size),
                    desc=f"Transform: {os.path.basename(feat_file)}"
                ):
                    # Transform to sparse matrix
                    X_batch = self.vectorizer.transform_batch(features)

                    # Save batch
                    batch_path = join(output_dir, 'batches', f'batch_{batch_idx}.npz')
                    label_path = join(output_dir, 'batches', f'labels_{batch_idx}.npy')

                    save_npz(batch_path, X_batch)
                    np.save(label_path, np.array(labels, dtype=np.int8))

                    batch_files.append(batch_path)
                    label_files.append(label_path)
                    batch_idx += 1

                    del X_batch
                    gc.collect()

            logging.info(f"Created {len(batch_files)} batch files")

        # PASS 3: Load all data and train
        logging.info("=" * 60)
        logging.info("PASS 3: Loading data and training LinearSVC")
        logging.info("=" * 60)

        # Load all batches
        logging.info("Loading all batches into memory...")
        X_matrices = []
        y_arrays = []

        for batch_path, label_path in tqdm(
            zip(batch_files, label_files),
            total=len(batch_files),
            desc="Loading batches"
        ):
            mat = load_npz(batch_path)
            # Convert to int64 indices for large datasets
            mat.indices = mat.indices.astype(np.int64)
            mat.indptr = mat.indptr.astype(np.int64)
            X_matrices.append(mat)
            y_arrays.append(np.load(label_path))

        logging.info("Combining batches...")
        X = vstack(X_matrices, format='csr')
        y = np.concatenate(y_arrays)

        # Convert to int64 indices for large datasets
        X.indices = X.indices.astype(np.int64)
        X.indptr = X.indptr.astype(np.int64)

        # Free memory
        del X_matrices, y_arrays
        gc.collect()

        logging.info(f"Combined data shape: X={X.shape}, y={y.shape}")
        logging.info(f"Memory for X: {X.data.nbytes / 1e9:.2f} GB")

        # Manual train/test split to avoid scipy fancy indexing overflow
        logging.info(f"Splitting data: {1-self.test_size:.0%} train, {self.test_size:.0%} test")

        # Stratified split: get indices for each class
        np.random.seed(self.random_state)

        # Get indices for each class
        class_0_indices = np.where(y == -1)[0]
        class_1_indices = np.where(y == 1)[0]

        # Shuffle indices
        np.random.shuffle(class_0_indices)
        np.random.shuffle(class_1_indices)

        # Calculate split sizes for each class
        n_test_0 = int(len(class_0_indices) * self.test_size)
        n_test_1 = int(len(class_1_indices) * self.test_size)

        # Split indices
        test_indices = np.concatenate([class_0_indices[:n_test_0], class_1_indices[:n_test_1]])
        train_indices = np.concatenate([class_0_indices[n_test_0:], class_1_indices[n_test_1:]])

        # Sort indices for efficient sparse matrix slicing
        train_indices = np.sort(train_indices)
        test_indices = np.sort(test_indices)

        logging.info(f"Train indices: {len(train_indices)}, Test indices: {len(test_indices)}")

        # Extract train and test sets using row slicing (more memory efficient)
        # Convert to lil_matrix for efficient row slicing, then back to csr
        logging.info("Extracting train set...")
        X_train_rows = []
        y_train = y[train_indices]

        # Process in chunks to avoid memory issues
        chunk_size = 50000
        for i in tqdm(range(0, len(train_indices), chunk_size), desc="Building train set"):
            chunk_indices = train_indices[i:i+chunk_size]
            X_train_rows.append(X[chunk_indices])

        X_train = vstack(X_train_rows, format='csr')
        del X_train_rows
        gc.collect()

        logging.info("Extracting test set...")
        X_test_rows = []
        y_test = y[test_indices]

        for i in tqdm(range(0, len(test_indices), chunk_size), desc="Building test set"):
            chunk_indices = test_indices[i:i+chunk_size]
            X_test_rows.append(X[chunk_indices])

        X_test = vstack(X_test_rows, format='csr')
        del X_test_rows
        gc.collect()

        # Convert to int32 for sklearn compatibility
        X_train.indices = X_train.indices.astype(np.int32)
        X_train.indptr = X_train.indptr.astype(np.int32)
        X_test.indices = X_test.indices.astype(np.int32)
        X_test.indptr = X_test.indptr.astype(np.int32)

        logging.info(f"Train: {X_train.shape[0]} samples")
        logging.info(f"Test: {X_test.shape[0]} samples")

        # Free full data
        del X, y
        gc.collect()

        # Train LinearSVC
        logging.info("Training LinearSVC...")
        logging.info(f"Parameters: C={self.C}, max_iter={self.max_iter}, class_weight={self.class_weight}")

        self.model = LinearSVC(
            loss='squared_hinge',
            penalty='l2',
            dual=True,
            C=self.C,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            class_weight=self.class_weight,
            verbose=1
        )

        self.model.fit(X_train, y_train)
        logging.info("Training complete!")

        # Evaluate
        logging.info("=" * 60)
        logging.info("Evaluation")
        logging.info("=" * 60)

        # Train metrics
        y_train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        logging.info(f"Train Accuracy: {train_acc:.4f}")

        # Test metrics
        y_test_pred = self.model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='binary')
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

        logging.info(f"Test Accuracy: {test_acc:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")
        logging.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        # Save metrics
        metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }

        with open(join(output_dir, 'metrics', 'metrics.txt'), 'w') as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

        # Save model
        logging.info("Saving model...")
        joblib.dump(self.model, join(output_dir, 'models', 'model.joblib'))

        # Save in .svm format
        self._save_svm_format(join(output_dir, 'models', 'model.svm'))

        # Plot loss curve
        plot_loss_curve_linearsvc(
            X_train, y_train, X_test, y_test, output_dir,
            C=self.C, class_weight=self.class_weight, random_state=self.random_state
        )

        logging.info("=" * 60)
        logging.info("Training pipeline complete!")
        logging.info("=" * 60)

        return metrics

    def _save_svm_format(self, path: str):
        """Save model weights in .svm format."""
        weights = self.model.coef_.flatten()
        bias = self.model.intercept_[0]

        with open(path, 'w') as f:
            f.write(f"No of features {self.vectorizer.n_features}\n")
            f.write(f"weightvectorbias\t{bias:.6f}\n")

            for feat_name, idx in self.vectorizer.feature_to_idx.items():
                f.write(f"{feat_name}\t{weights[idx]:.6f}\n")

        logging.info(f"Saved SVM weights to {path}")


def plot_loss_curve_linearsvc(X_train, y_train, X_test, y_test, output_dir,
                               C=1.0, class_weight='balanced', random_state=42):
    """
    Plot loss curve by training LinearSVC at multiple iteration checkpoints.

    Since LinearSVC doesn't expose per-iteration loss, we train multiple models
    with increasing max_iter values and compute hinge loss for each.
    """
    iter_checkpoints = [2500, 5000, 10000, 25000, 50000]
    train_losses = []
    test_losses = []

    logging.info("=" * 60)
    logging.info("Generating Loss Curve (training at multiple checkpoints)")
    logging.info("=" * 60)

    for max_iter in tqdm(iter_checkpoints, desc="Loss curve checkpoints"):
        model = LinearSVC(
            loss='squared_hinge',
            penalty='l2',
            dual=True,
            C=C,
            max_iter=max_iter,
            tol=1e-10,  # Very small to force max_iter to be the stopping criterion
            class_weight=class_weight,
            random_state=random_state,
            verbose=0
        )
        model.fit(X_train, y_train)

        # Compute decision scores
        train_scores = model.decision_function(X_train)
        test_scores = model.decision_function(X_test)

        # Compute hinge loss
        train_loss = hinge_loss(y_train, train_scores)
        test_loss = hinge_loss(y_test, test_scores)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        logging.info(f"  max_iter={max_iter}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(iter_checkpoints, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    plt.plot(iter_checkpoints, test_losses, 'r-o', label='Test Loss', linewidth=2, markersize=6)
    plt.xlabel('Max Iterations', fontsize=12)
    plt.ylabel('Hinge Loss', fontsize=12)
    plt.title('LinearSVC Loss Curve', fontsize=14)
    plt.legend(fontsize=11)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plot_path = join(output_dir, 'plots', 'loss_curve.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logging.info(f"Loss curve saved to {plot_path}")

    # Save loss data
    loss_data_path = join(output_dir, 'metrics', 'loss_curve_data.txt')
    with open(loss_data_path, 'w') as f:
        f.write("max_iter,train_loss,test_loss\n")
        for i, max_iter in enumerate(iter_checkpoints):
            f.write(f"{max_iter},{train_losses[i]:.6f},{test_losses[i]:.6f}\n")
    logging.info(f"Loss data saved to {loss_data_path}")

    return iter_checkpoints, train_losses, test_losses


def main():
    # Configuration
    feat_files = [
        'traindata/benigns.feat',
        'traindata/phishes.feat'
    ]
    output_dir = 'model_outputs_linearsvc'

    # Initialize trainer
    trainer = LinearSVCTrainer(
        max_features=100000,    # 100K features (reduced to fit int32 sparse indices)
        batch_size=10000,
        C=1.0,                  # Same as trainSVCmisc.py
        max_iter=100000,
        tol=1e-3,
        class_weight='balanced',
        random_state=42,
        test_size=0.1           # 10% for testing
    )

    # Train
    trainer.train(feat_files=feat_files, output_dir=output_dir)


if __name__ == "__main__":
    main()
