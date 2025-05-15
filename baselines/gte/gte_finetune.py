import logging
import os
import random
import numpy as np
import json
from datasets import disable_caching, load_from_disk, concatenate_datasets
disable_caching()
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments, Trainer, TrainerCallback
)
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, roc_curve
)


def compute_metrics(eval_pred):
    """
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    fpr, tpr, thresholds = roc_curve(labels, logits[:, 1])
    gmean = np.sqrt(tpr * (1-fpr))
    i = np.argmax(gmean)
    
    return {
        "accuracy": np.mean(predictions == labels),
        "recall": recall_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "auc": roc_auc_score(labels, logits[:, 1]),
        "gmean_sens_spec": gmean[i],
        "best_threshold": thresholds[i],
        "fpr@best_threshold": fpr[i],
        "tpr@best_threshold": tpr[i]
    }

class LoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
    # will call on_log on each logging step, specified by TrainerArguement. 
    # (i.e TrainerArguement.logginng_step)
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(logs) + "\n")


def main(params):
    """
    - Load training tokens dataset from disk
    - Finetune
    """

    logger = logging.getLogger(__name__)
    seed = params['seed']
    random.seed(seed)

    # load train tokens from disk
    tokens_dir = params['tokens_dir']
    max_length = params['max_length']
    features = params['features']
    train_path = os.path.join(tokens_dir, f'train_{max_length}_{features}_dataset')
    
    if not os.path.exists(train_path):
        logger.error(f"Create and save the token first")
        return
    
    logger.info(f"Loading train tokens from {train_path}...")
    train_dataset = load_from_disk(train_path)
    logger.info(f"Train tokens loaded: {train_dataset}")
    
    # contenate & shuffle
    logger.info(f"Concatenating & shuffling...")
    train_dataset = concatenate_datasets(
        [train_dataset[x] for x in train_dataset]
    ).shuffle(seed)
    logger.info(f"Shuffled training dataset: {train_dataset}")
    
    # finetune
    logger.info(f"Finetuning...")
    model = AutoModelForSequenceClassification.from_pretrained(
        params['model_ckpt'], 
        trust_remote_code=True
    )

    logger.info('-'*100)
    logger.info(f"Check out the model primed for classification: {model}")
    logger.info('-'*100)

    output_dir = os.path.join(params['output_dir'], f'gte-{max_length}-{features}')
    args = TrainingArguments(
        output_dir = output_dir,
        auto_find_batch_size = True,
        num_train_epochs = params['num_train_epochs'],
        eval_strategy = 'no', 
        save_strategy = 'epoch',
        learning_rate = params['learning_rate'],
        adam_beta1 = params['adam_beta1'],
        weight_decay = params['weight_decay'],
        use_cpu = False,
        logging_steps = params['logging_steps'],
        seed = params['seed'],
        save_only_model = params['save_only_model']
    )
    callback_path = os.path.join(output_dir, 'callback_log.jsonl')
    trainer = Trainer(
        model = model,
        args = args,
        train_dataset = train_dataset,
        compute_metrics = compute_metrics,
        callbacks = [LoggingCallback(callback_path)]
    )
    trainer.train(resume_from_checkpoint=params['resume_from_checkpoint'])




if __name__ == '__main__':
    data_dir = 'data/splitting/splits'
    tokens_dir = 'data/tokens/gte/release'
    n_procs = int(os.cpu_count() * 0.9)
    params = {
        'seed': 42,
        'logging_file': 'gte_finetune.log',
        'logging_level': 20,
        'tokens_dir': tokens_dir,
        'model_ckpt': 'Alibaba-NLP/gte-large-en-v1.5',
        'resume_from_checkpoint': None,
        'max_length': 1024,
        'features': 'both',
        'output_dir': 'logs/gte/release',
        'num_train_epochs': 2,
        'learning_rate': 1e-5,
        'adam_beta1': 0.9,
        'weight_decay': 1e-4,
        'logging_steps': 1,
        'save_only_model': False
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
    logger.info(f"Training complete!")