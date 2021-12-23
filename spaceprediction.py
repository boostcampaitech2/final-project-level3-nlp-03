from transformers import BertForTokenClassification, BertTokenizer, TrainingArguments

from preprocessor import Preprocessor
from dataset import CustomDataset
from trainer_qa import SpacingTrainer
from utils_qa import post_process_function

tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
model = BertForTokenClassification.from_pretrained(
    './models_baseline', 
    from_tf=bool(".ckpt" in './models'),
    num_labels=4
)

training_args = TrainingArguments(
    output_dir='./results'
)

trainer = SpacingTrainer(
    model=model,                         
    args=training_args,                  
    train_dataset=None,         
    post_process_function=post_process_function
)


import time

class BetweenSpace:
    def __init__(self, max_len=256):
        start = time.time()
        self.tokenizer=tokenizer
        self.model=model
        self.max_len=max_len
        self.preprocessor=Preprocessor(self.max_len, self.tokenizer)
        self.training_args = training_args
        self.trainer = trainer
        print("time :", time.time() - start)
        
    def predict_sentence(self,sent:str) -> str:
        test_dataset = CustomDataset(sentence=sent, transform=self.preprocessor.get_input_features)
        # test_dataset = CustomDataset(data_path='./test_data.csv', transform=self.preprocessor.get_input_features)
        
        prediction = self.trainer.predict(test_dataset=test_dataset, test_examples=test_dataset)

        return prediction['text_prediction'][0]


    def __call__(self, sent:str) -> str:
        sents = sent.split(',')        

        new_sent = [sents[0]]
        idx = 0
        for sent in sents[1:]:
            if len(new_sent[idx]+sent)<100:
                new_sent[idx] = new_sent[idx]+', '+sent
            else:
                new_sent.append(sent)
                idx += 1

        pred_sent = ", ".join([self.predict_sentence(sent.strip()) for sent in new_sent])

        return pred_sent

    def split_sentence(self, full_sent:str):
        split_sent = full_sent.strip().split('.')
        print(split_sent)
        return ". ".join(self(sent.strip()) for sent in split_sent if sent!='').strip()+"."


if __name__ == "__main__":
    spacing = BetweenSpace()
    
    print(spacing("철수와 영희는 감자,당근,양파,고춧가루,닭다리살,설탕과 소금을 사서 집으로 갔다."))
    print(spacing.split_sentence("동해물과백두산이 마르고닳도록 하느님이 보우하사 우리나라 만세. 무궁화삼천리화려강산 대한 사람 대한으로 길이 보전하세."))
    