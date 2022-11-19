import json
import numpy as np

accuarcy = []
correct_data = []
alldata = []
filename = 'result/all_results_test.jsonl'
for line in open(filename, 'r'):
    alldata.append(json.loads(line))
filename = 'result/correct_result_test.jsonl'
for line in open(filename, 'r'):
    correct_data.append(json.loads(line))
accuarcy.append(len(correct_data)/len(alldata)*100)
print('平均准确率：')
print(np.mean(accuarcy))

data = {}
for line in open('data/Math23K_test.jsonl'):
    temp = json.loads(line)
    data[temp['id']] = {'prolen': len(temp['original_text'].strip().split(' ')),
                        'equlen': len(temp['postfix']),
                        'result': False}

filename = 'result/correct_result_test.jsonl'
for line in open(filename, 'r'):
    temp = json.loads(line)
    data[temp['id']]['result'] = True

pro_total = [0, 0, 0, 0]
pro_correct = [0, 0, 0, 0]
equ_total = [0, 0, 0, 0]
equ_correct = [0, 0, 0, 0]
for k,d in data.items():
    if d['equlen'] <= 3:
        equ_total[0] += 1
        if d['result']:
            equ_correct[0] += 1
    elif d['equlen'] <= 5:
        equ_total[1] += 1
        if d['result']:
            equ_correct[1] += 1
    elif d['equlen'] <= 7:
        equ_total[2] += 1
        if d['result']:
            equ_correct[2] += 1
    else:
        equ_total[-1] += 1
        if d['result']:
            equ_correct[-1] += 1
    if d['prolen'] <= 20:
        pro_total[0] += 1
        if d['result']:
            pro_correct[0] += 1
    elif d['prolen'] <= 30:
        pro_total[1] += 1
        if d['result']:
            pro_correct[1] += 1
    elif d['prolen'] <= 40:
        pro_total[2] += 1
        if d['result']:
            pro_correct[2] += 1
    else:
        pro_total[3] += 1
        if d['result']:
            pro_correct[3] += 1

for i in range(len(pro_total)):
    if pro_total[i] != 0:
        pro_correct[i] /= pro_total[i] / 100
        pro_correct[i] = ("%.1f" % pro_correct[i])
for i in range(len(equ_total)):
    if equ_total[i] != 0:
        equ_correct[i] /= equ_total[i] / 100
        equ_correct[i] = ("%.1f" % equ_correct[i])
print(pro_total)
print(equ_total)
print(pro_correct)
print(equ_correct)

# train_data = {}
# dev_data = {}
# test_data = {}
# for line in open('data/math23k/Math23K_train.jsonl'):
#     temp = json.loads(line)
#     train_data[temp['id']] = {'expr': ' '.join(temp['prefix']), 'd':temp}
# for line in open('data/math23k/Math23K_dev.jsonl'):
#     temp = json.loads(line)
#     dev_data[temp['id']] = {'expr': ' '.join(temp['prefix']), 'd':temp}
# for line in open('data/math23k/Math23K_test.jsonl'):
#     temp = json.loads(line)
#     test_data[temp['id']] = {'expr': ' '.join(temp['prefix']), 'd':temp}
# data = train_data
# data.update(dev_data)
# data.update(test_data)
# expr = {}
# for k,d in data.items():
#     if d['expr'] not in expr:
#         expr[d['expr']] = 1
#     else:
#         expr[d['expr']] += 1
# expr = sorted(expr.items(), key=lambda x: -x[-1])
# expr = [x[0] for x in expr[:5]]

# total = [0] * 5
# correct = [0] * 5
# for k,d in test_data.items():
#     if d['expr'] in expr:
#         total[expr.index(d['expr'])] += 1
# filename = 'result_math23k/correct_result_test.jsonl'
# for line in open(filename, 'r'):
#     temp = json.loads(line)
#     if ' '.join(temp['prefix']) in expr:
#         correct[expr.index(' '.join(temp['prefix']))] += 1
# for i in range(len(total)):
#     if total[i] != 0:
#         correct[i] /= total[i] / 100
#         correct[i] = ("%.1f" % correct[i])
# print(total)
# print(correct)

# data = {}
# for line in open('data/math23k/Math23K_train.jsonl'):
#     temp = json.loads(line)
#     data[temp['id']] = {'expr': ' '.join(temp['prefix']), 'd':temp}
# for line in open('data/math23k/Math23K_dev.jsonl'):
#     temp = json.loads(line)
#     data[temp['id']] = {'expr': ' '.join(temp['prefix']), 'd':temp}
# for line in open('data/math23k/Math23K_test.jsonl'):
#     temp = json.loads(line)
#     data[temp['id']] = {'expr': ' '.join(temp['prefix']), 'd':temp}
# expr = {}
# count = 0
# for k,d in data.items():
#     if len(d['expr'].split(' ')) >= 9:
#         if d['expr'] not in expr:
#             expr[d['expr']] = 1
#         else:
#             expr[d['expr']] += 1
#         count += 1
# expr = sorted(expr.items(), key=lambda x: -x[-1])
# print(count)
# print(len(expr))