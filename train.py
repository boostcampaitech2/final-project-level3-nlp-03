import torch
from transformers import BertTokenizer, BertForTokenClassification, TrainingArguments, Trainer

from datasets import load_metric
import numpy as np

from torch.utils.data import DataLoader

from preprocessor import Preprocessor
from dataset import CustomDataset
from utils_qa import compute_metrics

def main():
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

    preprocessor = Preprocessor(128, tokenizer)
    train_dataset = CustomDataset('../data/reduced_train_v2.csv', preprocessor.get_input_features)
    dev_dataset = CustomDataset('../data/reduced_val_v2.csv', preprocessor.get_input_features)
    print(len(train_dataset))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = BertForTokenClassification.from_pretrained("monologg/kobert", num_labels=4)
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-05)

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=1000,                 # model saving step.
        num_train_epochs=5,              # total number of training epochs
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

    trainer.train()
    trainer.evaluate()
    trainer.save_model('./models')
    
if __name__ == "__main__":
    main()