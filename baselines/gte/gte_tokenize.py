import logging
import os
import random
import json
from datasets import disable_caching, Dataset, DatasetDict
disable_caching()
from transformers import AutoTokenizer


def get_fnames(dir, n):
    """
    """
    fnames = [
        os.path.join(dir, x) for x in os.listdir(dir)
        if x.endswith('.json')
    ]
    if n > 0: fnames = random.sample(fnames, n)    
    
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
                    if k in ['sha256', 'url', 'html']
                }
                data['labels'] = label
                yield data


def tokenize(examples, tokenizer, max_length, features):
    if features == 'both':
        text = examples['url'] + '\n' + examples['html']
    else:
        text = examples[features]
    return tokenizer(
        text, padding='max_length', 
        truncation=True, max_length=max_length
    )


def main(params):
    """
    - Prepare dataset
    - Tokenize
    - Save to disk
    """

    logger = logging.getLogger(__name__)
    seed = params['seed']
    n_procs = params['n_procs']
    random.seed(seed)
    class_types = ['phishes', 'benigns']
    save_tokens = params['save_tokens']
    train_test = params['train_test']
    train_test_dir = f'{train_test}_dir'

    
    # get filenames
    logger.info(f"Getting json names...")
    filenames = {}
    for class_type in class_types:
        dir = os.path.join(params[train_test_dir], class_type)
        if os.path.exists(dir):
            n = params['n']

            filenames[class_type] = get_fnames(dir, n)
            label = 1 if class_type == 'phishes' else 0
            filenames[class_type] = [(x, label) for x in filenames[class_type]]
            
            logger.info(f"Fetched {len(filenames[class_type])} {class_type} for {train_test}ing")
        else:
            class_types = [x for x in class_types if x != class_type]

    # read jsons
    logger.info(f"Reading jsons...")
    dataset = DatasetDict()
    for class_type in class_types:
        dataset[class_type] = Dataset.from_generator(
            read_json, 
            gen_kwargs={'filenames': filenames[class_type]}, 
            num_proc=n_procs
        )
        logger.info(f"Read {class_type} jsons into {dataset[class_type]} for {train_test}ing")

    # tokenize
    max_length = params['max_length']
    features = params['features']
    logger.info(f"Tokenizing with features, {features}, for length, {max_length}...")
    tokenizer = AutoTokenizer.from_pretrained(
        params['model_ckpt'], trust_remote_code=True
    )
    
    dataset = dataset.map(
        tokenize,
        remove_columns=['url', 'html'],
        num_proc=n_procs,
        fn_kwargs={
            'tokenizer': tokenizer,
            'max_length': max_length,
            'features': features,
        }
    )
    logger.info(f"Tokenized with features, {features}, for length, {max_length} as: {dataset}")

    if save_tokens:
        # save tokens
        logger.info(f"Saving {train_test} tokens with features: {features} to disk...")
        tokens_dir = params['tokens_dir']
        os.makedirs(tokens_dir, exist_ok=True)
        path = os.path.join(tokens_dir, f'{train_test}_{max_length}_{features}_dataset')
        dataset.save_to_disk(path)
        logger.info(f"{train_test} tokens with features: {features} saved as {path}")

    logger.info(f"Tokenization complete!")


if __name__ == '__main__':
    data_dir = 'data/splitting/splits'
    tokens_dir = 'data/tokens/gte/release'
    n_procs = int(os.cpu_count() * 0.9)
    params = {
        'seed': 42,
        'logging_file': 'gte_tokenize.log',
        'logging_level': 20,
        'train_test': 'benchmark',
        'train_dir': os.path.join(data_dir, 'train'),
        'test_dir': os.path.join(data_dir, 'test'),
        'benchmark_dir': os.path.join(data_dir, 'benchmark'),
        'tokens_dir': tokens_dir,
        'save_tokens': True,
        'n': -1,
        'n_procs': n_procs,
        'model_ckpt': 'Alibaba-NLP/gte-large-en-v1.5',
        # 'model_ckpt': 'jinaai/jina-embeddings-v3',
        'max_length': 1024,
        'features': 'both',
    }

    logging.basicConfig(
        filename=params['logging_file'], 
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        encoding='utf-8', level=params['logging_level']
    )
    logger = logging.getLogger(__name__)
    logger.info(f"\nStarting main...")
    
    main(params)