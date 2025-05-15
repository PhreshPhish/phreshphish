"""
Memory-efficient SVM model training for large-scale datasets.

This module provides utilities for training LinearSVC models on large datasets
that may not fit in memory. It implements batch processing, sparse matrix representations,
and disk checkpointing to handle large feature sets efficiently.

Usage:
    trainer = MemoryEfficientTrainer(max_features=4000000, batch_size=1000)
    trainer.train_model(train_dir='path/to/training/data', output_dir='output_path')
"""

import csv
from sklearn.svm import LinearSVC
from scipy.sparse import vstack, csr_matrix, save_npz, load_npz
import os
import joblib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from tqdm import tqdm
from os.path import join, exists, splitext, isdir
import logging
import tempfile
import pickle
import multiprocessing


csv.field_size_limit(20000000)
class MemoryEfficientVectorizer:
    def __init__(self, max_features=None, batch_size=1000, handle_none='average'):
        self.feature_indices = {}
        self.feature_sums = {}
        self.feature_counts = {}
        self.feature_max = {}
        self.feature_replacement = {}
        self.n_features = 0
        self.max_features = max_features
        self.batch_size = batch_size
        self.handle_none = handle_none
    
    def update_feature_indices(self, dict_data):
        try:
            for d in dict_data:
                # Skip the sha256 feature if it exists in the data
                if 'sha256' in d:
                    d = {k: v for k, v in d.items() if k != 'sha256'}
                    
                for feat, val in d.items():
                    if feat not in self.feature_indices:
                        if self.max_features and len(self.feature_indices) >= self.max_features:
                            continue
                        self.feature_indices[feat] = len(self.feature_indices)
                        self.feature_sums[feat] = 0.0
                        self.feature_counts[feat] = 0
                        self.feature_max[feat] = float('-inf')
                    
                    if val is not None:
                        val = float(val)
                        self.feature_sums[feat] += val
                        self.feature_counts[feat] += 1
                        if val > self.feature_max[feat]:
                            self.feature_max[feat] = val
            
            self.n_features = len(self.feature_indices)
            if self.handle_none == 'average':
                self.feature_replacement = {
                    feat: (self.feature_sums[feat] / self.feature_counts[feat]) if self.feature_counts[feat] > 0 else 0.0
                    for feat in self.feature_sums
                }
            elif self.handle_none == 'max':
                self.feature_replacement = {
                    feat: self.feature_max[feat] if self.feature_counts[feat] > 0 else 0.0
                    for feat in self.feature_max
                }
        except Exception as e:
            logging.error(f"Error in update_feature_indices: {e}")
            raise

    def fit(self, dict_data):
        try:
            for i in tqdm(range(0, len(dict_data), self.batch_size), desc="Fitting Vectorizer"):
                batch = dict_data[i:i + self.batch_size]
                self.update_feature_indices(batch)
            return self
        except Exception as e:
            logging.error(f"Error in fit method: {e}")
            raise

    def _transform_batch(self, dict_data):
        try:
            rows, cols, data = [], [], []
            for row_idx, d in enumerate(dict_data):
                # Skip the sha256 feature during transformation
                for feat, val in d.items():
                    if feat != 'sha256' and feat in self.feature_indices:
                        if val is None:
                            val = self.feature_replacement.get(feat, 0.0)
                        rows.append(row_idx)
                        cols.append(self.feature_indices[feat])
                        data.append(float(val))
            # Explicitly specify dtype=np.int32 for the indices
            return csr_matrix(
                (data, (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))), 
                shape=(len(dict_data), self.n_features), 
                dtype=np.float32
            )
        except Exception as e:
            logging.error(f"Error in _transform_batch: {e}")
            raise

    def transform(self, dict_data):
        try:
            if not self.feature_indices:
                raise ValueError("Vectorizer must be fitted before transform")
            matrices = []
            for i in tqdm(range(0, len(dict_data), self.batch_size), desc="Transforming batches"):
                batch = dict_data[i:i + self.batch_size]
                batch_matrix = self._transform_batch(batch)
                matrices.append(batch_matrix)
            return vstack(matrices)
        except Exception as e:
            logging.error(f"Error in transform method: {e}")
            raise

    def fit_transform(self, dict_data):
        return self.fit(dict_data).transform(dict_data)


def load_fv_large_file_multithread(feat_file, chunk_size=1000, n_threads=4, max_rows=None, offset=0):
    try:
        # Use all available cores if not specified
        if n_threads is None:
            n_threads = multiprocessing.cpu_count()

        all_labels = []
        all_features = []

        def process_chunk_multithread(chunk):        
            local_labels, local_features = [], []
            for row in chunk:
                if len(row) > 0:
                    lbl = int(row[0].split(' ')[0])
                    sha256 = row[0].split(' ')[1]
                    # Create a dictionary of features, excluding 'sha256' as a feature key
                    dictRow = {feat.split(':')[0]: float(feat.split(':')[1]) for feat in row[0].split(' ')[2:]}
                    # Store sha256 separately from the feature vectors
                    dictRow['sha256'] = sha256  # Keep it as a metadata field but not as a feature
                    local_labels.append(lbl)
                    local_features.append(dictRow)
            return local_labels, local_features

        with open(feat_file, 'r') as fp:
            reader = csv.reader(fp, delimiter='\t')
            
            # Skip to the desired offset
            if offset > 0:
                for _ in range(offset):
                    try:
                        next(reader)
                    except StopIteration:
                        # No more rows to process
                        return all_labels, all_features
            
            chunk = []
            futures = []
            total_rows_processed = 0
            
            with ThreadPoolExecutor(max_workers=n_threads) as executor, \
                tqdm(total=max_rows or float('inf'), desc=f"Loading {feat_file} from offset {offset}") as pbar:
                for row in reader:
                    if max_rows and total_rows_processed >= max_rows:
                        break
                    chunk.append(row)
                    total_rows_processed += 1
                    pbar.update(1)
                    if len(chunk) >= chunk_size:
                        futures.append(executor.submit(process_chunk_multithread, chunk))
                        chunk = []
                if chunk:
                    futures.append(executor.submit(process_chunk_multithread, chunk))
                
                for future in futures:
                    local_labels, local_features = future.result()
                    all_labels.extend(local_labels)
                    all_features.extend(local_features)
                
                # Break if max_rows is reached
                if max_rows and total_rows_processed >= max_rows:
                    return all_labels, all_features

        return all_labels, all_features
    except Exception as e:
        logging.error(f"Error in load_fv_large_file_multithread: {e}")
        raise

def batch_predict(model, X, batch_size=1000):
    """Make predictions in batches"""
    predictions = []
    n_samples = X.shape[0]
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch_predictions = model.predict(X[i:end])
        predictions.append(batch_predictions)
    
    return np.concatenate(predictions)

def batch_decision_function(model, X, batch_size=1000):
    """Get decision function scores in batches"""
    scores = []
    n_samples = X.shape[0]
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch_scores = model.decision_function(X[i:end])
        scores.append(batch_scores)
    
    return np.concatenate(scores)



class MemoryEfficientTrainer:
    def __init__(self, max_features=300000, batch_size=1000, n_threads=4):
        self.vectorizer = MemoryEfficientVectorizer(max_features=max_features, batch_size=batch_size)
        # Use all available cores
        self.n_threads = n_threads if n_threads is not None else multiprocessing.cpu_count()
        logging.info(f"Using {self.n_threads} CPU cores for parallel processing")
    
    def find_all_feat_files(self, train_dir):
        """Find all feature files in the directory structure"""
        all_files = []
        
        # Check if the path is a directory
        if not isdir(train_dir):
            return all_files
            
        # Look for phish and benign subfolders
        phish_dir = join(train_dir, 'phish')
        benign_dir = join(train_dir, 'benign')
        
        # Process phish directory if it exists
        if isdir(phish_dir):
            phish_files = [join(phish_dir, f) for f in os.listdir(phish_dir) if f.endswith('.feat')]
            all_files.extend(phish_files)
            #all_files=all_files[:1]
            logging.info(f"Found {len(phish_files)} phishing feature files in {phish_dir}")
        
        # Process benign directory if it exists
        if isdir(benign_dir):
            benign_files = [join(benign_dir, f) for f in os.listdir(benign_dir) if f.endswith('.feat')]
            all_files.extend(benign_files)
            #all_files=all_files[:2]
            logging.info(f"Found {len(benign_files)} benign feature files in {benign_dir}")
        
        # Fallback to direct search in the given directory
        if not all_files:
            all_files = [join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.feat')]
            logging.info(f"Found {len(all_files)} feature files directly in {train_dir}")
            
        return all_files

    def process_training_directory(self, train_dir, output_dir):
        """
        Process all files in the training directory with improved batch loading.
        """
        all_labels = []
        batch_files = []

        feat_files = self.find_all_feat_files(train_dir)
        
        # Define the batch size for loading large files
        row_batch_size = 50000  # Process 50K rows at a time instead of 450K at once
        
        # Temporary file to store features for vectorizer fitting
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_name = temp_file.name

            # First pass: Write all features to the temporary file in smaller batches
            logging.info("Writing features to disk in batches...")
            for file_name in tqdm(feat_files, desc="Processing files for vectorizer fitting"):
                file_path = file_name
                if not file_path.endswith('.feat'):
                    continue
                    
                is_benign = 'benign' in file_path
                
                # Process large files in smaller batches
                offset = 0
                while True:
                    # For benign files, load in batches up to max_row_count
                    # For phishing files, load all content in batches
                    current_batch_size = row_batch_size
                    
                    logging.info(f"Loading batch from {file_path}: rows {offset} to {offset + current_batch_size}")
                    batch_labels, batch_features = load_fv_large_file_multithread(
                        file_path, 
                        n_threads=self.n_threads, 
                        max_rows=current_batch_size,
                        offset=offset  # We need to add an offset parameter to load_fv_large_file_multithread
                    )
                    
                    if not batch_features:  # No more data to read
                        logging.info(f"No more data in {file_path} after {offset} rows")
                        break
                        
                    # Append to temporary file
                    pickle.dump(batch_features, temp_file)
                    
                    # Update offset for next batch
                    offset += len(batch_features)
                    
                    # For benign files, check if we've reached max_row_count
                    if (is_benign or not is_benign) and offset >= 430000:
                        logging.info(f"Reached max row count (430,000) for benign file {file_path}")
                        break

            # Fit vectorizer on all features from the temporary file
            logging.info("Fitting vectorizer on features from disk...")
            with open(temp_file_name, 'rb') as temp_file:
                while True:
                    try:
                        file_features = pickle.load(temp_file)
                        self.vectorizer.fit(file_features)
                    except EOFError:
                        break

        checkpoint_dir = join(output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        vectorizer_file = join(checkpoint_dir, 'vectorizer_fitted.joblib')
        joblib.dump(self.vectorizer, vectorizer_file)
        logging.info(f"Saved fitted vectorizer checkpoint to {vectorizer_file}")

        #Second pass: Process files for transformation and saving
        logging.info("Processing and transforming all files...")
        for file_name in tqdm(feat_files, desc="Transforming and saving batches"):
            file_path = file_name
            if not file_path.endswith('.feat'):
                continue
                
            is_benign = 'benign' in file_path
            
            # Process each file in smaller batches
            offset = 0
            file_batch_parts = []
            
            while True:
                current_batch_size = row_batch_size
                
                logging.info(f"Loading batch for transformation from {file_path}: rows {offset} to {offset + current_batch_size}")
                batch_labels, batch_features = load_fv_large_file_multithread(
                    file_path, 
                    n_threads=self.n_threads, 
                    max_rows=current_batch_size,
                    offset=offset
                )
                
                if not batch_features:  # No more data to read
                    break
                    
                all_labels.extend(batch_labels)
                
                # Generate a unique filename for this batch part
                filename = os.path.basename(file_name)
                batch_part_prefix = f"{splitext(filename)[0]}_offset_{offset}"
                
                # Transform and save this batch part
                batch_part_files = self.transform_and_save(batch_features, batch_part_prefix, output_dir)
                file_batch_parts.extend(batch_part_files)
                
                # Update offset for next batch
                offset += len(batch_features)
                
                # For benign files, check if we've reached max_row_count
                if (is_benign or not is_benign) and offset >= 430000:
                    logging.info(f"Reached max row count (430,000) for benign file {file_path}")
                    break
                
            batch_files.extend(file_batch_parts)

        logging.info("Combining all training data batches...")
        combined_matrix = self.combine_batches(batch_files)

        # Clean up the temporary file
        os.remove(temp_file_name)

        return all_labels, combined_matrix
    
    def process_test_data(self, test_dir, output_dir):
        """
        Process test data with the same batched approach as training data.
        Uses disk checkpointing to handle large datasets efficiently.
        """
        all_labels = []
        batch_files = []
        
        feat_files = self.find_all_feat_files(test_dir)
        if not feat_files:
            logging.error(f"No test feature files found in {test_dir}")
            return [], None
            
        # Define the batch size for loading large files
        row_batch_size = 50000  # Process 50K rows at a time
        
        # Temporary file to store features for vectorizer fitting
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_name = temp_file.name
            
            # Process all test files in batches
            logging.info("Processing test files in batches...")
            for file_name in tqdm(feat_files, desc="Processing test files"):
                file_path = file_name
                if not file_path.endswith('.feat'):
                    continue
                    
                # Process large files in smaller batches
                offset = 0
                file_batch_parts = []
                
                while True:
                    logging.info(f"Loading batch for transformation from {file_path}: rows {offset} to {offset + row_batch_size}")
                    batch_labels, batch_features = load_fv_large_file_multithread(
                        file_path, 
                        n_threads=self.n_threads, 
                        max_rows=row_batch_size,
                        offset=offset
                    )
                    
                    if not batch_features:  # No more data to read
                        break
                        
                    all_labels.extend(batch_labels)
                    
                    # Generate a unique filename for this batch part
                    filename = os.path.basename(file_name)
                    batch_part_prefix = f"test_{splitext(filename)[0]}_offset_{offset}"
                    
                    # Transform and save this batch part
                    batch_part_files = self.transform_and_save(batch_features, batch_part_prefix, output_dir)
                    file_batch_parts.extend(batch_part_files)
                    
                    # Update offset for next batch
                    offset += len(batch_features)
                    if offset >= 430000:
                        logging.info(f"Reached max row count (430,000) for benign file {file_path}")
                        break
                    
                batch_files.extend(file_batch_parts)

        logging.info("Combining all test data batches...")
        combined_matrix = self.combine_batches(batch_files)
        
        # Clean up the temporary file
        os.remove(temp_file_name)
        
        # Create a checkpoint of the test data
        test_matrix_file = join(output_dir, 'X_test.npz')
        save_npz(test_matrix_file, combined_matrix)
        with open(join(output_dir, 'test_labels.pkl'), 'wb') as f:
            pickle.dump(all_labels, f)
        logging.info(f"Saved test data checkpoint to {test_matrix_file}")
            
        return all_labels, combined_matrix

    def transform_and_save(self, data, filename_prefix, output_dir, fit=False):
        """
        Transform data in batches, save each batch to a file, and return the list of batch file paths.
        """
        batch_files = []
        if fit:
            logging.info("Fitting vectorizer on the data...")
            self.vectorizer.fit(data)

        for i in tqdm(range(0, len(data), self.vectorizer.batch_size), desc="Transforming and saving batches"):
            batch = data[i:i + self.vectorizer.batch_size]
            batch_matrix = self.vectorizer.fit_transform(batch) if fit else self.vectorizer._transform_batch(batch)
            batch_filename = join(output_dir, f"{filename_prefix}_batch_{i // self.vectorizer.batch_size}.npz")
            save_npz(batch_filename, batch_matrix)
            batch_files.append(batch_filename)
        return batch_files

    def combine_batches(self, batch_files):
        """
        Combine all batch files into a single sparse matrix.
        """
        matrices = []
        for file in tqdm(batch_files, desc="Combining batches"):
            matrices.append(load_npz(file))
        return vstack(matrices)

    def save_model_weights(self, model, vectorizer, output_path):
        """
        Save model weights with feature names, excluding 'sha256'.
        
        Parameters:
        - model: Trained LinearSVC model
        - vectorizer: Fitted vectorizer with feature indices
        - output_path: Path to save the model weights
        """
        try:
            bias = model.intercept_[0] 
            weights = model.coef_.flatten()
            
            # Get feature names, excluding 'sha256'
            feature_names = [
                feature for feature in vectorizer.feature_indices.keys() 
                if feature != 'sha256'
            ]
            feature_indices = {
                feature: idx for feature, idx in vectorizer.feature_indices.items()
                if feature != 'sha256'
            }
            
            n_features = len(feature_names)
            
            with open(output_path, "w") as f:
                # Write number of features and bias
                f.write(f"No of features {n_features}\n")
                f.write(f"weightvectorbias\t{bias:.6f}\n")
            
                # Write weights with actual feature names
                for feature, idx in feature_indices.items():
                    f.write(f"{feature}\t{weights[idx]:.6f}\n")
                    
            logging.info(f"Saved model weights to {output_path} (excluding sha256 feature)")
                    
        except Exception as e:
            logging.error(f"Error saving model weights: {e}")
            raise

    def train_model(self, train_dir, output_dir):
        """
        Train a LinearSVC model using data from the training directory.
        """
        try:
            if not exists(train_dir):
                raise FileNotFoundError(f"Training directory {train_dir} not found")
            
            logging.info("Processing training directory...")
            train_labels, X_train = self.process_training_directory(train_dir, output_dir)

            logging.info(f"Training data shape: {X_train.shape}")
            logging.info("Training model...")

            model = LinearSVC(
                loss="squared_hinge",
                penalty='l2',
                dual=True,
                max_iter=100000,
                tol=1e-3,
                random_state=42,
                class_weight='balanced',
                C=1.0
            )

            model.fit(X_train, train_labels)

            logging.info("Saving model...")
            os.makedirs(join(output_dir, 'models'), exist_ok=True)
            joblib.dump(model, join(output_dir, 'models', 'model.joblib'))
            joblib.dump(self.vectorizer, join(output_dir, 'models', 'vectorizer.joblib'))
            
            # Use our new save function that explicitly excludes sha256
            self.save_model_weights(
                model=model,
                vectorizer=self.vectorizer,
                output_path=join(output_dir, 'models', 'model.svm')
            )

            logging.info("Training complete!")

            

        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise





if __name__ == "__main__":
    try:
        output_dir = f'model_outputs_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(output_dir, exist_ok=True)
        for subdir in ['models', 'plots', 'metrics']:
            os.makedirs(join(output_dir, subdir), exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(join(output_dir, f'model_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
                logging.StreamHandler()
            ]
        )

        # Calculate optimal batch size based on available memory and cores
        n_cores = multiprocessing.cpu_count()
        logging.info(f"Detected {n_cores} CPU cores, utilizing all for parallel processing")
        
        # Adjust batch size based on available cores to optimize memory usage
        batch_size = max(500, 1000 // n_cores * n_cores)
        
        trainer = MemoryEfficientTrainer(max_features=4000000, batch_size=batch_size, n_threads=n_cores)
        trainer.train_model(
            train_dir=join('traindata','splits','train'),
            output_dir=output_dir
        )
    except Exception as e:
        logging.critical(f"Critical error in main: {e}")
        raise
