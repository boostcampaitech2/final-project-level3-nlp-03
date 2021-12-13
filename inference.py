import torch
from transformers import BertTokenizer, BertForTokenClassification, TrainingArguments, Trainer

from datasets import load_metric
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from preprocessor import Preprocessor
from dataset import CustomDataset

def post_process_function()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = 0
    for idx, pred in enumerate(predictions):
        if np.array_equal(predictions[idx],labels[idx]): accuracy+=1

    return {'accuracy': accuracy/len(predictions)}


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # load tokenizer
    Tokenizer_NAME = "monologg/kobert"
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

    model_path = './models'
    model = BertForTokenClassification.from_pretrained(
        model_path, 
        from_tf=bool(".ckpt" in model_path),
        num_labels=4
    )
    model.to(device)

    preprocessor = Preprocessor(128, tokenizer)
    test_dataset = CustomDataset('../data/data/test.csv', preprocessor.get_input_features)

    
    batch_size = 32
    training_args = TrainingArguments(
        output_dir='./results'
    )

    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=None,         
        eval_dataset=test_dataset,             
        compute_metrics=compute_metrics
    )

    a = trainer.predict(test_dataset=test_dataset)
    logits = a.predictions
    labels = a.label_ids
    predictions = np.argmax(logits, axis=-1)

    print(compute_metrics((logits, labels)))