import logging
import os
import random
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

def predict(example, rank, model, ranks, temperature):
    if rank is None: rank = 0
    device = torch.device(f'cuda:{ranks[rank]}')
    with torch.no_grad():
        model = model.to(device)
        example = {
            k: v.to(device) 
            for k, v in example.items() 
            if k not in ['sha256', 'labels']
        }
        outputs = model(**example)
        scores = F.softmax(outputs.logits/temperature, dim=1)[:,1]
    
    return {
        'logits': outputs.logits,
        f'score@temp{temperature}': scores
    }


def main(params):
    """
    """
    logger = logging.getLogger(__name__)
    seed = params['seed']
    random.seed(seed)
    max_length = params['max_length']
    features = params['features']

    # load test tokens from disk
    tokens_dir = params['tokens_dir']
    test_benchmark = params['test_benchmark']
    test_path = os.path.join(tokens_dir, f'{test_benchmark}_{max_length}_{features}_dataset2')

    if not os.path.exists(test_path):
        logger.error(f"Create and save the token first")
        return
    
    logger.info(f"Loading test tokens from {test_path}...")
    test_dataset = load_from_disk(test_path)
    cols = ['input_ids', 'token_type_ids', 'attention_mask']
    test_dataset.set_format('torch', columns=cols)
    logger.info(f"Test tokens loaded: {test_dataset}")

    # load the model
    logger.info(f"Loading the trained classification model...")
    model_ckpt = params['model_ckpt']
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt,
        trust_remote_code=True
    )
    logger.info('-'*100)
    logger.info(f"Check out the classification model: {model}")
    logger.info('-'*100)

    # predict
    logger.info(f"Predicting...")
    ranks = params['ranks']
    batch_size = params['batch_size']
    temperature = params['temperature']
    cols = ['input_ids', 'token_type_ids', 'attention_mask']
    
    model.eval()
    test_dataset = test_dataset.map(
        predict,
        with_rank = True, 
        fn_kwargs = {
            'model': model, 'ranks': ranks, 
            'temperature': temperature
        },
        batched = True, batch_size = batch_size, 
        remove_columns = cols, num_proc = len(ranks)
    )
    logger.info(f"Predictions created as: {test_dataset}")

    # save predictions
    save_pred = params['save_pred']
    pred_path = os.path.join(
        params['pred_dir'], f'{test_benchmark}_{max_length}_{features}_pred'
    )
    if save_pred:
        logger.info(f"Saving predictions...")
        test_dataset.save_to_disk(pred_path)
        logger.info(f"Predictions saved as {pred_path}")






if __name__ == '__main__':
    tokens_dir = 'data/tokens/gte/release'
    pred_dir = 'data/predictions/gte/release'
    params = {
        'seed': 42,
        'logging_file': 'gte_predict.log',
        'logging_level': 20,
        'tokens_dir': tokens_dir,
        'test_benchmark': 'test',
        'max_length': 1024,
        'features': 'both',
        'model_ckpt': 'logs/gte/release/gte-1024-both/checkpoint-23364',
        'ranks': [0, 1, 2, 3],
        'batch_size': 16,
        'temperature': 8,
        'save_pred': True,
        'pred_dir': pred_dir,
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

    logger.info("Prediction complete!")