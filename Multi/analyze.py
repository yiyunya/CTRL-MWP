import json
import numpy as np

data = []
filename = 'result_draw/all_results_test.jsonl'
for line in open(filename, 'r'):
    data.append(json.loads(line))
ans1, ans2 = 0, 0
count1, count2 = 0, 0
for d in data:
    scores = [d['tree_score1'], d['tuple_score1']]
    results = [d['tree_result1'], d['tuple_result1']]
    if results[1]:
        ans1 += 1
    if np.array(scores).argmax() == 0:
        count1 += 1
    else:
        count2 += 1
    ans2 += results[np.array(scores).argmax()]
print(ans1/len(data), ans2/len(data))
print(count1, count2)