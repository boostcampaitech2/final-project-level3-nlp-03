import torch
from transformers import BertTokenizer, BertForTokenClassification, TrainingArguments, Trainer, EvalPrediction

from typing import Tuple, Optional, List

import numpy as np

from tqdm import tqdm

import os
import json
import yaml
import csv
import re
from pprint import pprint 

from preprocessor import Preprocessor
from dataset import CustomDataset
from trainer_qa import SpacingTrainer
from utils_qa import compute_metrics

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
            if idx>=len(pred): continue
            if pred[idx]==2:
                pred_text += ' ' + text
            else:
                pred_text += text


        text_predictions.append("".join(pred_text[1:]))
        text_original.append(" ".join(example))

    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        if examples.ground_truths:
            text_dict = {i:( 
                            " ".join(examples.sentences[i]), 
                            " ".join(examples.ground_truths[i]),
                            text_predictions[i]
                        ) for i in range(len(text_predictions))}
        else: 
            text_dict = {i:(
                            " ".join(examples.sentences[i]), 
                            text_predictions[i]
                        ) for i in range(len(text_predictions))}


        prediction_file = os.path.join(
            output_dir,
            "predictions.json",
        )
                
        with open(prediction_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(text_dict, indent=4, ensure_ascii=False) + "\n"
            )
        
    return text_predictions

def inf(test_data, simple):
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['tokenizer_name'])

    model = BertForTokenClassification.from_pretrained(
        config['model_path'], 
        from_tf=bool(".ckpt" in config['model_path']),
        num_labels=4
    )
    model.to(device)

    preprocessor = Preprocessor(config['max_len'], tokenizer)
    
    if simple == 1:
        f = open("test_data.csv", "w", newline='')
        wr = csv.writer(f)
        wr.writerow(["id", "correct_sentence", "wrong_sentence"])
        wr.writerow(["1", test_data, test_data])
        f.close()
        test_dataset = CustomDataset('test_data.csv', preprocessor.get_input_features, test=True)
    else:
        test_dataset = CustomDataset(config['test_data_path'], preprocessor.get_input_features, test=True)
    
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
    with open('results/predictions.json') as prediction_stc:
        rlt = json.load(prediction_stc)
    return(rlt['0'][2])

if __name__ == "__main__":
    test_dataset = CustomDataset(config['test_data_path'], preprocessor.get_input_features, test=True)
    inf(test_dataset, 0)