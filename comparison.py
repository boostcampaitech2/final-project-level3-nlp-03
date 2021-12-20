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

with open('./results/predictions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

pred_accuracy = 0
spacing_accuracy = 0

new_data = {}

for key in tqdm(data):
    input_data, ground_truth, predicted = data[key]
    spacing_pred = spacing("".join(input_data.split()))
    pred_accuracy += count_accuracy(ground_truth, predicted)
    spacing_accuracy += count_accuracy(ground_truth, spacing_pred)

    new_data[key] = (input_data, ground_truth, predicted, spacing_pred)

print(f'prediction accuracy: {pred_accuracy/len(data)}, spacing_accuracy: {spacing_accuracy/len(data)}')

with open('./results/compare_prediction.json', "w", encoding="utf-8") as writer:
    writer.write(
        json.dumps(new_data, indent=4, ensure_ascii=False) + "\n"
    )
