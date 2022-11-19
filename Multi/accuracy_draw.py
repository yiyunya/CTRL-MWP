import json
import numpy as np

accuarcy = []
for fold in range(5):
    correct_data = []
    alldata = []
    filename = 'result_draw/all_results_dev_' + str(fold) + '.jsonl'
    for line in open(filename, 'r'):
        alldata.append(json.loads(line))
    filename = 'result_draw/correct_result_dev_' + str(fold) + '.jsonl'
    for line in open(filename, 'r'):
        correct_data.append(json.loads(line))
    accuarcy.append(len(correct_data)/len(alldata)*100)
print('平均准确率：')
print(np.mean(accuarcy))

data = {}
for line in open('data/draw/5-fold/DRAW_fold0_train.jsonl'):
    temp = json.loads(line)
    data[temp['id']] = {'prolen': len(temp['original_text'].strip().split(' ')),
                        'equlen': len(temp['prefix']),
                        'result': False}
for line in open('data/draw/5-fold/DRAW_fold0_test.jsonl'):
    temp = json.loads(line)
    data[temp['id']] = {'prolen': len(temp['original_text'].strip().split(' ')),
                        'equlen': len(temp['prefix']),
                        'result': False}

for fold in range(5):
    filename = 'result_draw/correct_result_dev_' + str(fold) + '.jsonl'
    for line in open(filename, 'r'):
        temp = json.loads(line)
        data[temp['id']]['result'] = True

pro_total = [0, 0, 0, 0]
pro_correct = [0, 0, 0, 0]
equ_total = [0, 0, 0, 0]
equ_correct = [0, 0, 0, 0]
for k,d in data.items():
    if d['equlen'] <= 11:
        equ_total[0] += 1
        if d['result']:
            equ_correct[0] += 1
    elif d['equlen'] <= 13:
        equ_total[1] += 1
        if d['result']:
            equ_correct[1] += 1
    elif d['equlen'] <= 15:
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
print('文字准确率：')
print(pro_total)
print(pro_correct)
print('表达式准确率：')
print(equ_total)
print(equ_correct)

data = {}
for line in open('data/draw/5-fold/DRAW_fold0_train.jsonl'):
    temp = json.loads(line)
    data[temp['id']] = {'expr': ' '.join(temp['prefix']), 'd':temp}
for line in open('data/draw/5-fold/DRAW_fold0_test.jsonl'):
    temp = json.loads(line)
    data[temp['id']] = {'expr': ' '.join(temp['prefix']), 'd':temp}
expr = {}
for k,d in data.items():
    if d['expr'] not in expr:
        expr[d['expr']] = 1
    else:
        expr[d['expr']] += 1
expr = sorted(expr.items(), key=lambda x: -x[-1])
pro = [x[1]/len(data)*100 for x in expr[:5]]
for i in range(len(pro)):
    pro[i] = ("%.1f" % pro[i])
expr = [x[0] for x in expr[:5]]

total = [0] * 5
correct = [0] * 5
for k,d in data.items():
    if d['expr'] in expr:
        total[expr.index(d['expr'])] += 1
for fold in range(5):
    filename = 'result_draw/correct_result_dev_' + str(fold) + '.jsonl'
    for line in open(filename, 'r'):
        temp = json.loads(line)
        if ' '.join(temp['prefix']) in expr:
            correct[expr.index(' '.join(temp['prefix']))] += 1
for i in range(len(total)):
    if total[i] != 0:
        correct[i] /= total[i] / 100
        correct[i] = ("%.1f" % correct[i])
print('top5准确率：')
print(expr)
print(total)
print(correct)
print(pro)

# data = {}
# count, total = 0, 0
# for line in open('data/draw/5-fold/DRAW_fold0_train.jsonl'):
#     temp = json.loads(line)
#     data[temp['id']] = temp['prefix'].count('=')
# for line in open('data/draw/5-fold/DRAW_fold0_test.jsonl'):
#     temp = json.loads(line)
#     data[temp['id']] = temp['prefix'].count('=')
# for fold in range(5):
#     filename = 'result_draw/all_results_dev_' + str(fold) + '.jsonl'
#     for line in open(filename, 'r'):
#         temp = json.loads(line)
#         if data[temp['id']] == temp['res'].count('='):
#             count += 1
#         total += 1
# print(count / total)

data = {}
for line in open('data/draw/5-fold/DRAW_fold0_train.jsonl'):
    temp = json.loads(line)
    data[temp['id']] = {'prolen': len(temp['original_text'].strip().split(' ')),
                        'equlen': len(temp['prefix']),
                        'type': temp['prefix'].count('='),
                        'prefix': ''.join(temp['prefix']),
                        'result': False}
for line in open('data/draw/5-fold/DRAW_fold0_test.jsonl'):
    temp = json.loads(line)
    data[temp['id']] = {'prolen': len(temp['original_text'].strip().split(' ')),
                        'equlen': len(temp['prefix']),
                        'type': temp['prefix'].count('='),
                        'prefix': ''.join(temp['prefix']),
                        'result': False}
for fold in range(5):
    filename = 'result_draw/all_results_dev_' + str(fold) + '.jsonl'
    for line in open(filename, 'r'):
        temp = json.loads(line)
        if temp['tree_out1'].count('=') == data[temp['id']]['type']:
            data[temp['id']]['result'] = True

equ_total = [0, 0, 0, 0]
equ_single = [0, 0, 0, 0]
equ_type_correct = [0, 0, 0, 0]
exprs = [set(), set(), set(), set()]
for k,d in data.items():
    if d['equlen'] <= 11:
        equ_total[0] += 1
        if d['type'] == 1:
            equ_single[0] += 1
        if d['result']:
            equ_type_correct[0] += 1
        exprs[0].add(d['prefix'])
    elif d['equlen'] <= 13:
        equ_total[1] += 1
        if d['type'] == 1:
            equ_single[1] += 1
        if d['result']:
            equ_type_correct[1] += 1
        exprs[1].add(d['prefix'])
    elif d['equlen'] <= 15:
        equ_total[2] += 1
        if d['type'] == 1:
            equ_single[2] += 1
        if d['result']:
            equ_type_correct[2] += 1
        exprs[2].add(d['prefix'])
    else:
        equ_total[-1] += 1
        if d['type'] == 1:
            equ_single[-1] += 1
        if d['result']:
            equ_type_correct[-1] += 1
        exprs[-1].add(d['prefix'])

print('方程数量准确率：')
print(equ_type_correct)
for i in range(len(equ_total)):
    equ_type_correct[i] /= equ_total[i] / 100
    equ_type_correct[i] = ("%.1f" % equ_type_correct[i])
print(equ_type_correct)

print(equ_total)
print(equ_single)
for i in range(len(equ_total)):
    equ_single[i] /= equ_total[i] / 100
    equ_single[i] = ("%.1f" % equ_single[i])
print(equ_single)
# exprs = [len(x) for x in exprs]
# print(exprs)
# for i in range(len(equ_total)):
#     exprs[i] /= equ_total[i] / 100
#     exprs[i] = ("%.1f" % exprs[i])
# print(exprs)