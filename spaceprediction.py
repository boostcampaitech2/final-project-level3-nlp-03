from transformers import BertForTokenClassification, BertTokenizer, TrainingArguments

from preprocessor import Preprocessor
from dataset import CustomDataset
from trainer_qa import SpacingTrainer
from utils_qa import post_process_function

tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
model = BertForTokenClassification.from_pretrained(
    './models', 
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

class Spacing:
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
        pred_sent = self.predict_sentence(sent)

        return pred_sent


if __name__ == "__main__":
    spacing = Spacing()

    start = time.time()
    print(spacing("아버지가방에들어가신다."))
    print("time :", time.time() - start)

    start = time.time()
    print(spacing("또 국 가 정보 원 장 비 서실장 출신 인 윤 아무 개( 5 7) 씨 가대 학 동 창 인 최 중경 전 지식 경제 부 장관 한 테 ‘ ㅎ 사 가 부탁 한 인 물 을 한국 전 력 자 회 사 고위 직 으 로 임 명해달 라 ’ 고청탁 해 실 제임 명 된 것 을밝혀내 고 원 전 브로커오 씨 로부터“ ㅎ사 로 부 터 2 억 8 0 0 0 만 원 을 받 아윤 씨 한테전달 했 다” 는 진술 을 확보 했 지만,최 전 장관 은 무 혐 의처 분 했 다 ."))
    print("time :", time.time() - start)

    start = time.time()
    print(spacing("전 람 회 , 카니발( 김 동 률 이 적 ), 베 란 다 프 로젝 트 ( 김 동 률 이상순 )를 거 친김 동 률의 여섯번 째 솔로 앨 범 ‘ 동행 ’(사진 ) 이 1일 발 매 된 이 후 돌 풍 을멈 추 지 않 는 다."))
    print("time :", time.time() - start)

    start = time.time()
    print(spacing("며칠 전에 는뉴 이 스트( 황 )민현 이형 이'잘 될수있 을거 야' 라 는말을해주면 서부담감을떨칠 수있도록해줬 다 고했다."))
    print("time :", time.time() - start)