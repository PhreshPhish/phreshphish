from huggingface_hub import list_repo_files
from datasets import load_dataset, load_from_disk, concatenate_datasets
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import multiprocessing as mp
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import seaborn as sns
sns.set_theme()


def get_ci(data: list) -> tuple[float, float, float]:
    """
    generate confidence interval @ 95% confidence level
    input: 
        - data in a list/array
    processing:
        - generate 10k bootstrap samples from the data assuming uniform distribution
    output:
        - mean of the data
        - 2.5 percentile of bootstrap samples as the lower limit of CI
        - 97.5 percentile of bootstrap samples as the upper limit of CI
    """
    
    n = len(data)
    n_bootstraps = 10000

    # Generate bootstrap resamples and compute means
    bootstrap_means = np.array([
        np.mean(np.random.choice(data, size=n, replace=True))
        for _ in range(n_bootstraps)
    ])

    # Compute 95% confidence interval
    lower = np.percentile(bootstrap_means, 2.5)
    upper = np.percentile(bootstrap_means, 97.5)
    
    return np.mean(data), lower, upper

def eval(params: tuple) -> tuple:
    """
    input:
        - params - a tuple of HF_PATH, eval_file, test_ds, benign_ds
        - HF_PATH -> 
        - eval_file -> benchmark file name in huggingface
        - test_ds -> test dataset with sha256, label and prediction score
        - benign_ds -> test dataset with sha256, label and prediction score with only benign
    processing:
        - read the sha256's in eval_file
        - fetch the test datapoints of the sha256's from test_ds
        - concatenate with benign datapoint from benign_ds
        - compute precision & recall
        - interpolate precision for a recall grid of 1000 points
        - compute average precision

    """
    HF_PATH, eval_file, test_ds, benign_ds = params

    ds = load_dataset(HF_PATH, data_files=eval_file, trust_remote_code=True)
    sha256s = ds['train']['text']
    sha256s = [x.split('.')[0] for x in sha256s]

    phish_ds = test_ds.filter(lambda x: x['sha256'] in sha256s)

    eval_ds = concatenate_datasets([benign_ds, phish_ds])

    y_true = np.array([
        1.0 if x == 'phish' else 0.0 for x in eval_ds['label']
    ])
    y_score = np.array(eval_ds['score'])

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    
    # Need to sort recall ascending for interpolation
    recall = recall[::-1]
    precision = precision[::-1]

    # Define a common recall grid for interpolation (important for averaging)
    recall_grid = np.linspace(0, 1, 1000)
    precision_interp = np.interp(recall_grid, recall, precision)
    avg_precs = average_precision_score(y_true, y_score)

    return precision_interp, precision, recall, avg_precs


class PhishBenchEvaluator:
    def __init__(self, pred_dir: str):
        """
        inputs:
            - prediction directory
        processing: 
            - fetch phreshphish/phreshphish/benchmark files from huggingface
            - get the base rates available from the file names
        """
        self.test_ds = load_from_disk(pred_dir)
        assert (
            'sha256' in self.test_ds.features.keys() and 
            'label' in self.test_ds.features.keys() and
            'score' in self.test_ds.features.keys()
        ), "Need sha256, label & score features in prediction"

        self.HF_PATH = 'phreshphish/phreshphish'

        try:
            # fetch the benchmark file names
            benchmark_files = list_repo_files(self.HF_PATH, repo_type='dataset')
            
            # fetch the benign benchmark file names
            self.ben_bench_files = [f for f in benchmark_files if f.startswith('benchmark/benign')]
            ds = load_dataset(self.HF_PATH, data_files=self.ben_bench_files, trust_remote_code=True)
            sha256s = ds['train']['text']
            sha256s = [x.split('.')[0] for x in sha256s]
            self.benign_ds = self.test_ds.filter(lambda x: x['sha256'] in sha256s)

            # fetch the phish benchmark file names
            self.phish_bench_files = [f for f in benchmark_files if f.startswith('benchmark/benchmark')]
            # get the base rates for which the benchmarks are available
            self.available_base_rates = set([float(x.split('-')[1]) for x in self.phish_bench_files])
        except Exception as e:
            print(f"{self.HF_PATH}/benchmark not loaded. Exception: {e}")

        
        
    def get_available_base_rates(self) -> list[float]:
        """
        inputs: none
        outputs: available base rates
        """
        return sorted(list(self.available_base_rates))
    
    def evaluate(self, base_rate: float):
        """
        inputs:
            - base_rate: float -> rate of positive samples (phish)
        processing:
            - assert the base rate in available base rates
            - 
        """
        assert base_rate in self.available_base_rates, f"{base_rate} is not available in benchmark. Available: {self.available_base_rates}"

        # filter benchmark_files for base_rate
        eval_files = [x for x in self.phish_bench_files if float(x.split('-')[1]) == base_rate]

        params = [(self.HF_PATH, eval_file, self.test_ds, self.benign_ds) for eval_file in eval_files]
        n_procs = int(mp.cpu_count() * 0.8)
        # with mp.Pool(n_procs) as pool:
        #     results = pool.map(eval, params)
        results = map(eval, params)
        
        df = pd.DataFrame(results, columns=['precision_interp', 'precision', 'recall', 'avg_precs'])

        return df.precision_interp.to_list(), df.avg_precs.to_list(), df.precision.to_list(), df.recall.to_list()


        

    def plot(self, plot_params, plot_save):

        mpl.rcParams.update({
            'font.size': 16,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
        })
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), tight_layout=True)

        table_rows = [
            ['Base Rate', 'Average Precision (CI)', '@Recall', 'Precision @Recall (CI)']
        ]
        for base_rate in plot_params.keys():
            precision_list, avg_precs, precisions, recalls = plot_params[base_rate]

            for precision, recall in zip(precisions, recalls):
                axs[1].plot(recall, precision, color='grey', alpha=1/10)

            # Average precision over all datasets
            mean_precision = np.mean(precision_list, axis=0)

            # compute precision @ 0.9 recall
            target_recall = 0.9

            recall_grid = np.linspace(0, 1, 1000)
            # Find the index of the recall value closest to 0.9
            differences = np.abs(recall_grid - target_recall)
            
            # Find the index of the smallest difference
            closest_index = np.argmin(differences)

            # Get the corresponding precision
            precisionAtR = [ x[closest_index] for x in precision_list ]    #mean_precision[closest_index]
            mean_precAtR, low_precAtR, hi_precAtR = get_ci(precisionAtR)
            closest_recall = recall_grid[closest_index]
            
            mean_ap, low_ap, hi_ap = get_ci(avg_precs)
            table_rows.append([
                base_rate, 
                f"{mean_ap:.4} ({low_ap:.4}, {hi_ap:.4})", 
                f"{closest_recall:.2}", 
                f"{mean_precAtR:.4} ({low_precAtR:.4}, {hi_precAtR:.4})"
            ])
        
            # Plot the mean Precision-Recall curve
            axs[1].plot(recall_grid, mean_precision, linewidth=2, label=f'{base_rate} (AP: {mean_ap:.4})')

        bbox_height = 0.1 + (0.1 * len(plot_params.keys()))
        bbox = Bbox.from_bounds(0.0, 0.0, 1.0, bbox_height)
        table = axs[0].table(cellText=table_rows[1:], colLabels=table_rows[0], cellLoc='center', loc='center', bbox=bbox)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        axs[0].set_xticklabels([])
        axs[0].set_yticklabels([])

        # Plot settings
        axs[1].set_xlabel('Recall')
        axs[1].set_ylabel('Precision')
        axs[1].set_title('Precision-Recall Curves with Average')
        axs[1].set_ylim([0, 1.09])
        axs[1].set_xlim([0, 1.09])
        axs[1].legend(title='Base Rates')
        plt.show()

        if plot_save is not None:
            fig.savefig(plot_save)



from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

from tqdm import tqdm
import time
import random

if __name__ == '__main__':
    params = {
        'pred_dir': 'data/predictions/gte/release/hf_test_pred',
        'plot_save': 'data/misc/phishbencheval/bencheval.png'
    }

    phish_bench_evaluator = PhishBenchEvaluator(params['pred_dir'])
    available_base_rates = phish_bench_evaluator.get_available_base_rates()

    plot_params = {}
    for base_rate in tqdm(available_base_rates):
        plot_params[base_rate] = phish_bench_evaluator.evaluate(base_rate)
        time.sleep(random.randint(1, 10))

    phish_bench_evaluator.plot(plot_params, params['plot_save'])