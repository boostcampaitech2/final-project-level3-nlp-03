import torch
from transformers import BertTokenizer, BertForTokenClassification, TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint

from datasets import load_metric
import numpy as np
import os

from torch.utils.data import DataLoader

from preprocessor import Preprocessor
from dataset import CustomDataset
from utils_qa import compute_metrics

from typing import Tuple

import yaml

def main():
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

    preprocessor = Preprocessor(config['max_len'], tokenizer)
    train_dataset = CustomDataset(data_path=config['train_data_path'], transform=preprocessor.get_input_features)
    dev_dataset = CustomDataset(data_path=config['val_data_path'], transform=preprocessor.get_input_features)
    print(len(train_dataset))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForTokenClassification.from_pretrained(config['model_name'], num_labels=4)
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-05)

    training_args = TrainingArguments(
        output_dir=config['output_dir'],          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=1000,                 # model saving step.
        num_train_epochs=4,              # total number of training epochs
        learning_rate=5e-5,               # learning_rate
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        warmup_steps=1000,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=1000,              # log saving step.
        evaluation_strategy='steps', # evaluation strategy to adopt during training
                                    # `no`: No evaluation during training.
                                    # `steps`: Evaluate every `eval_steps`.
                                    # `epoch`: Evaluate every end of epoch.
        eval_steps = 1000,            # evaluation step.
        load_best_model_at_end = True,
    )

    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,         
        eval_dataset=dev_dataset,             
        compute_metrics=compute_metrics
    )

    #resume_from_checkpoint
    last_checkpoint = get_last_checkpoint(config['output_dir'])
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif os.path.isdir(config['model_path']):
        checkpoint = config['model_path']
    else:
        checkpoint = None

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.evaluate()
    trainer.save_model('./models')
    
if __name__ == "__main__":
    main()