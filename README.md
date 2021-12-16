## 현재 성능
- monologg/kobert accuracy: 0.84902

## Things to do
- [ ] inference.py 개선
- [x] prediction postprocess (ex. [0 2 3 2 ... ] -> 저랑 코딩하실래요?)
- [ ] slot_type_ids 적용하기
- [ ] metrics 적용
- [ ] 모델 성능 개선
- [ ] 코드 개선, 추상화 작업
- [ ] NER task 코드 공부하기
- [x] KoSpacing 비교 코드 작성
- [ ] requirements.txt
- [ ] max_len 문제 해결

## 바뀐 것 설명
### prediction postprocess
- 기존 코드에서는 predict()의 리턴값이 'EvalLoopOutput'이었으나 dictionary 형태로 바꿈. 
- output_dir에 json 파일 형태로 예측값 저장

```
trainer_qa.py에서 
text_prediction = self.post_process_function(
            test_examples, test_dataset, output.predictions, self.args, self.args.output_dir
        )
self.args.output_dir을 지우면 예측값 저장 X
```

### KoSpacing 비교 코드
- inference.py 이후 만들어진 ./results/predictions.json 파일을 이용하여 비교.


## 버그 
id, wrong_sentence, correct_setence 중 correct_sentence에 None을 쓸 시 띄어쓰기 퀄리티가 매우 떨어짐. 
데이터의 형태는 test_data.csv 참고

datset.py - self.test가 True, False여도 같은 버그가 일어남.

'''
if self.test:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
            }

        return {
          'input_ids': input_ids,
          'attention_mask': attention_mask,
          'token_type_ids': token_type_ids,
          # 'slot_labels': slot_labels,
          'labels': targets
        }
'''