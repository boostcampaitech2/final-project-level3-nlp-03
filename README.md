# 한국어 띄어쓰기 교정 모델: Between Spaces


## 🐝 Members 🐝
강진선|김다인|김민지|송이현|신원지|이나영
:-:|:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/u/79238023?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/31719240?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/74283190?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/32431157?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/52646313?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/69383548?v=4' height=80 width=80px></img>
[Github](https://github.com/iamtrueline)|[Github](https://github.com/promisemee)|[Github](https://github.com/kimminji2018)|[Github](https://github.com/Ihyun)|[Github](https://github.com/sw6820)|[Github](https://github.com/NayoungLee-de)

## Dataset
- 사용 데이터셋
    - 문법성 판단 말뭉치
    - 신문 말뭉치
    - 문어 말뭉치
    - 개체명 분석 말뭉치 2020

## Demo

### Web
![image](https://user-images.githubusercontent.com/31719240/147256798-ab7925b2-cae3-4004-97c5-505fd957a273.png)

### Code
```
>>> from spaceprediction import BetweenSpace
>>> btwspace = BetweenSpace() 
>>> btwspace("철수와 영희는 감자,당근,양파,고춧가루,닭다리살,설탕과소금을 사서 집으로 갔다.")
"철수와 영희는 감자, 당근, 양파, 고춧가루, 닭다리살, 설탕과 소금을 사서 집으로 갔다."
>>> # 한 문장 이상은 split_sentence() 함수를 써주세요
>>> btwspace.split_sentence("엘리자베스는 조용히 듣고있었지만 그의말에 동의하진 않았다. 그들의 행동은 보기좋은 행동은 아니었다. 빠른 판단과 유연하지않은 사고로 엘리자베스는 그들에 대한 판단을 너무나빨리 결정하였다."))
"엘리자베스는 조용히 듣고 있었지만 그의 말에 동의하진 않았다. 그들의 행동은 보기 좋은 행동은 아니었다. 빠른 판단과 유연하지 않은 사고로 엘리자베스는 그들에 대한 판단을 너무나 빨리 결정하였다."
```

## How to Use

### Installation
```
$ pip install -r requirements.txt
```

### Train
```
$ python train.py
```

### Inference
아래 코드를 돌리면 config.yaml의 output_dir 아래 prediction.json 파일이 생성됩니다. 
inference 결과는 해당 파일을 확인해주세요.
```
$ python inference.py
$ python 
```

### Web
```
$ streamlit run web_test.py
```