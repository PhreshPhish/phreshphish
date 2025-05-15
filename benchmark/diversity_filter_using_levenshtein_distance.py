import os
import json

def get_fnames(dir):
    """
    """
    fnames = [
        os.path.join(dir, x) for x in os.listdir(dir)
        if x.endswith('.json')
    ]    
    return fnames

def read_json(filenames):
    """
    """
    for filename, label in filenames:
        with open(filename) as f:
            data = json.loads(f.read())
            if type(data['html']) == str:
                data = {
                    k: v for k, v in data.items()
                    if k in ['url', 'html']
                }
                data['labels'] = label
                yield data


import Levenshtein

def lenmatch(row, comprow, matching_ratios):
    row_urllen = len(row['url'])
    row_htmllen = len(row['html'])
    comprow_urllen = len(comprow['url'][0])
    comprow_htmllen = len(comprow['html'][0])
    url_lenlim = (row_urllen + comprow_urllen) * (1 - matching_ratios['url'])
    html_lenlim = (row_htmllen + comprow_htmllen) * (1 - matching_ratios['html'])
    url_lendiff = abs(row_urllen - comprow_urllen)
    html_lendiff = abs(row_htmllen - comprow_htmllen)

    return (
        url_lendiff <= url_lenlim,
        html_lendiff <= html_lenlim
    )


def compute_lev_dist(row, idx, ds, matching_ratios):
    ret_val = {'lev_ratio': '', 'lev_idx': '', 'lev_feat_type': ''}
    for i in range(ds.num_rows-1, idx, -1):
        comprow = ds.select([i]).to_dict()
        compute_url, compute_html = lenmatch(row, comprow, matching_ratios)
        if compute_url:
            lev_ratio = Levenshtein.ratio(row['url'], comprow['url'][0])
            if lev_ratio >= matching_ratios['url']:
                ret_val['lev_ratio'] += f'{lev_ratio},'
                ret_val['lev_idx'] += f'{i},'
                ret_val['lev_feat_type'] += 'url,'
                # break
        # elif compute_html:
        if compute_html:
            lev_ratio = Levenshtein.ratio(row['html'], comprow['html'][0])
            if lev_ratio >= matching_ratios['html']:
                ret_val['lev_ratio'] += f'{lev_ratio},'
                ret_val['lev_idx'] += f'{i},'
                ret_val['lev_feat_type'] += 'html,'
                # break
    return ret_val

import shutil
def copy_json(params):
    src, dst, urls = params
    i = 0
    with open(src) as f:
        data = json.load(f)
    if data['url'] not in urls:
        shutil.copy(src, dst)
        i = 1
    return i


import numpy as np
from datasets import Dataset, disable_caching
disable_caching()
import multiprocessing as mp

if __name__ == '__main__':
    dir = 'data/splitting/splits/test/phishes'
    # test, train, phishes & benigns can be found in /home/hgowda/projects/phishing/data/splitting/splits
    pred_dir = 'data/predictions/gte/release/test_1024_both_pred/phishes'
        # benigns can be found in /home/hgowda/projects/phishing/data/predictions/gte/release/test_1024_both_pred/benigns
    num_proc = 72
    n_samples = -1 # -1 for all
    matching_ratios = {'url': 0.95, 'html': 0.95} # the threshold for levenstein ratio
    save_flag = True
    save_ds = 'data/misc/levenshtein_dist_r095095' # change r100 based on matching_ratios

    src_path = 'data/splitting/splits/test/phishes'
    dst_path = 'data/splitting/splits/benchmark/phishes'

    print(f"Getting filenames")
    filenames = get_fnames(dir)
    print(f"Got {len(filenames)} files")
    label = 1
    filenames = [(x, label) for x in filenames]

    print(f"Building a dataset from the jsons")
    dataset = Dataset.from_generator(
        read_json, 
        gen_kwargs={'filenames': filenames}, 
        num_proc=num_proc
    )
    print(f"dataset created {dataset}")

    print(f"Loading the predictions from the disk")
    preds = Dataset.load_from_disk(pred_dir)
    print(f"Prediction loaded: {preds}")

    dataset = dataset.add_column('score', np.array(preds['score@temp8'])) # they are in the same order
    dataset = dataset.sort('score', reverse=True) # reverse=False for benigns
    print(f"Dataset added with score and sorted")

    if n_samples > 0: dataset = dataset.select(range(n_samples))

    print(f"Computing the Levenshtein ratios")
    ds = dataset.map(
        compute_lev_dist, with_indices=True, num_proc=num_proc,
        fn_kwargs={'ds': dataset, 'matching_ratios': matching_ratios}
    )
    print(f"Levenstein ratios obtained in dataset: {ds}")

    print(f"Saving the data")
    if save_flag: ds.save_to_disk(save_ds)
    print(f"Levenshtein ratios saved in {save_ds}")

    df = ds.to_pandas()
    print(len(df[df.lev_idx != '']))

    srcs = [os.path.join(src_path, x) for x in os.listdir(src_path)]
    dsts = [os.path.join(dst_path, x) for x in os.listdir(src_path)]

    urls = df[df.lev_idx != '']['url'].to_list()

    params = list(zip(srcs, dsts))
    params = [(*x, urls) for x in params]

    with mp.Pool(64) as pool:
        results = pool.map(copy_json, params)

    print(f"Copied {len(os.listdir(dst_path))} jsons to {dst_path}")