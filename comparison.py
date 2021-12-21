## PyKoSpace와 비교하기 위한 코드

import json
from pykospacing import Spacing
from sklearn.metrics import accuracy_score

from tqdm import tqdm

spacing = Spacing()

def get_tags(sentence):
    all_tags = []
    for word in sentence:
        for i, c in enumerate(word):
            if i == 0:
                all_tags.append("B")
            else:
                all_tags.append("I")

    return all_tags


def count_accuracy(y_true, y_pred):
    return y_true==y_pred

def count_accuracy(y_true, y_pred):
    return y_true==y_pred

with open('./results/predictions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

pred_accuracy = 0
spacing_accuracy = 0
pred_binary = 0
spacing_binary = 0

new_data = {}

def get_tag(sentence):
    sentence=sentence.split()
    all_tags = []
    for word in sentence:
        for i, c in enumerate(word):
            if i == 0:
                all_tags.append("B")
            else:
                all_tags.append("I")

    return all_tags


for key in tqdm(data):
    input_data, ground_truth, predicted = data[key]
    ground_truth=ground_truth.strip()
    predicted=predicted.strip()
    spacing_pred = spacing("".join(input_data.split()))
    ground_truth_tag = get_tag(ground_truth)
    predicted_tag = get_tag(predicted)
    spacing_pred_tag = get_tag(spacing_pred)

    pred_binary += count_accuracy(ground_truth, predicted)
    spacing_binary += count_accuracy(ground_truth, spacing_pred)

    try:
        pred_accuracy += accuracy_score(ground_truth_tag, predicted_tag)
        spacing_accuracy += accuracy_score(ground_truth_tag, spacing_pred_tag)
    except:
        pass

    new_data[key] = (input_data, ground_truth, predicted, spacing_pred)

print(f'prediction accuracy: {pred_accuracy/len(data)}, spacing_accuracy: {spacing_accuracy/len(data)} \
        prediction binary: {pred_binary/len(data)}, spacing_binary: {spacing_binary/len(data)}')

with open('./results/compare_prediction.json', "w", encoding="utf-8") as writer:
    writer.write(
        json.dumps(new_data, indent=4, ensure_ascii=False) + "\n"
    )
