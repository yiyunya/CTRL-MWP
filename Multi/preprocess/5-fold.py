import json

filename_train = open('../data/draw/dev_test/DRAW_train.jsonl', 'r')
filename_dev = open('../data/draw/dev_test/DRAW_dev.jsonl', 'r')
filename_test = open('../data/draw/dev_test/DRAW_test.jsonl', 'r')
data = []
for line in filename_train:
    temp = json.loads(line)
    data.append(temp)
for line in filename_dev:
    temp = json.loads(line)
    data.append(temp)
for line in filename_test:
    temp = json.loads(line)
    data.append(temp)

fold_size = int(len(data) * 0.2)
fold_pairs = []
for split_fold in range(4):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(data[fold_start:fold_end])
fold_pairs.append(data[(fold_size * 4):])

for fold in range(5):
    pairs_tested = []
    pairs_trained = []
    for fold_t in range(5):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]
    f = open('../data/draw/5-fold/DRAW_fold'+str(fold)+'_train.jsonl', 'w')
    for d in pairs_trained:
        json.dump(d, f)
        f.write('\n')
    f.close()
    f = open('../data/draw/5-fold/DRAW_fold'+str(fold)+'_test.jsonl', 'w')
    for d in pairs_tested:
        json.dump(d, f)
        f.write('\n')
    f.close()


# filename_train = open('../data/hmwp/dev_test/HMWP_train.jsonl', 'r')
# filename_dev = open('../data/hmwp/dev_test/HMWP_dev.jsonl', 'r')
# filename_test = open('../data/hmwp/dev_test/HMWP_test.jsonl', 'r')
# data = []
# for line in filename_train:
#     temp = json.loads(line)
#     data.append(temp)
# for line in filename_dev:
#     temp = json.loads(line)
#     data.append(temp)
# for line in filename_test:
#     temp = json.loads(line)
#     data.append(temp)

# fold_size = int(len(data) * 0.2)
# fold_pairs = []
# for split_fold in range(4):
#     fold_start = fold_size * split_fold
#     fold_end = fold_size * (split_fold + 1)
#     fold_pairs.append(data[fold_start:fold_end])
# fold_pairs.append(data[(fold_size * 4):])

# for fold in range(5):
#     pairs_tested = []
#     pairs_trained = []
#     for fold_t in range(5):
#         if fold_t == fold:
#             pairs_tested += fold_pairs[fold_t]
#         else:
#             pairs_trained += fold_pairs[fold_t]
#     f = open('../data/hmwp/5-fold/HMWP_fold'+str(fold)+'_train.jsonl', 'w')
#     for d in pairs_trained:
#         json.dump(d, f, ensure_ascii=False)
#         f.write('\n')
#     f.close()
#     f = open('../data/hmwp/5-fold/HMWP_fold'+str(fold)+'_test.jsonl', 'w')
#     for d in pairs_tested:
#         json.dump(d, f, ensure_ascii=False)
#         f.write('\n')
#     f.close()