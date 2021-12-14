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
- [ ] KoSpacing 비교 코드 작성

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