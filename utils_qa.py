import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = 0
    accuracy_aon = 0
    for idx, pred in enumerate(predictions):
        pred_list = [x for x in predictions[idx].tolist() if x!=0]
        label_list = [x for x in labels[idx].tolist() if x!=0]
        
        try:
            accuracy += accuracy_score(label_list, pred_list)
            accuracy_aon += (label_list == pred_list)
        except:
            pass


    pred_len = len(predictions)

    return {'acc': accuracy/pred_len, 'acc_all_or_none': accuracy_aon/pred_len}
