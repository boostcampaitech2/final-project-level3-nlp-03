import torch

from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score

from transformers import BertTokenizer, BertForTokenClassification

import yaml

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

model.predict()