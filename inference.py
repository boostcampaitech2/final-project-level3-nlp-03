import torch
from transformers import BertTokenizer, BertForTokenClassification, TrainingArguments, Trainer, EvalPrediction

from typing import Tuple, Optional, List

from datasets import load_metric
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from preprocessor import Preprocessor
from dataset import CustomDataset
from trainer_qa import SpacingTrainer

from tqdm import tqdm

import os
import json

def post_process_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments,
        output_dir: Optional[str]
    ) -> List[str]:

    text_predictions = []
    text_original = []

    for idx, example in enumerate(tqdm(examples.sentences)):
        original = "".join(example)
        pred = np.argmax(predictions[idx], axis=-1).tolist()
        pred = pred[1:len(original)+1]
        
        pred_text = []
        for idx, text in enumerate(original):
            if pred[idx]==2:
                pred_text += ' ' + text
            else:
                pred_text += text


        text_predictions.append("".join(pred_text[1:]))
        text_original.append(" ".join(example))

    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        text_dict = {i:(text_original[i], text_predictions[i]) for i in range(len(text_predictions))}

        prediction_file = os.path.join(
            output_dir,
            "predictions.json",
        )
                
        with open(prediction_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(text_dict, indent=4, ensure_ascii=False) + "\n"
            )
        
    return text_predictions


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

    trainer = SpacingTrainer(
        model=model,                         
        args=training_args,                  
        train_dataset=None,         
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        post_process_function=post_process_function
    )

    prediction = trainer.predict(test_dataset=test_dataset, test_examples=test_dataset)
    print(prediction["metrics"])