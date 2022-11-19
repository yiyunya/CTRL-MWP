import json

def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

count = [0, 0, 0, 0, 0, 0]
for i in range(5):
    train_data = load_data('mtokens/DRAW_fold' + str(i) + '_train.jsonl')
    for d in train_data:
        if '<O>' in d['text']:
            count[0] += 1
        elif '<Add>' in d['text']:
            count[1] += 1
        elif '<Mul>' in d['text']:
            count[2] += 1
        elif '<Equ>' in d['text']:
            count[3] += 1
        elif '<Var>' in d['text']:
            count[4] += 1
        elif '<Sol>' in d['text']:
            count[5] += 1
print([x/4 for x in count])