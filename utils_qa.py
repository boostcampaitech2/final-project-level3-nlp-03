import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from typing import Tuple, Optional, List

from transformers import TrainingArguments
import os
import json
from tqdm import tqdm

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = 0
    accuracy_aon = 0
    f1 = 0
    lc = 0
    lc_aon = 0
    lc_f1=0
    for idx, pred in enumerate(predictions):
        pred_list = [x for x in predictions[idx].tolist() if x!=0]
        label_list = [x for x in labels[idx].tolist() if x!=0]
        cut_pred = pred_list[:len(label_list)]
        
        try:
            lc += accuracy_score(label_list, cut_pred)
            lc_aon += (label_list == cut_pred)
            lc_f1 += f1_score(label_list, cut_pred, pos_label=2)
            accuracy += accuracy_score(label_list, pred_list)
            accuracy_aon += (label_list == pred_list)
            f1 += f1_score(pred_list, label_list, pos_label=2)
            
        except:
            pass

    pred_len = len(predictions)

    return {'acc': accuracy/pred_len, 
            'acc_binary': accuracy_aon/pred_len, 
            'f1_score': f1/pred_len}

def post_process_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments,
        output_dir: Optional[str]
    ) -> List[str]:

    text_predictions = []

    for idx, example in enumerate(tqdm(examples.sentences, disable=True)):
        original = "".join(example)
        pred = np.argmax(predictions[idx], axis=-1).tolist()
        pred = pred[1:len(original)+1]
        
        pred_text = []
        pred
        for idx, text in enumerate(original):
            if idx>=len(pred): continue
            if pred[idx]==2:
                pred_text += ' ' + text
            else:
                pred_text += text

        text_predictions.append("".join(pred_text))

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
