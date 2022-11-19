from re import I
import copy
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
from transformers import BertTokenizer, AutoTokenizer, AutoConfig, AutoModel, AdamW

from transformers import get_linear_schedule_with_warmup

from configuration.config import *
from models.text import Encoder
from models.tree import TreeDecoder
from models.train_and_evaluate import Solver, train_double, evaluate_double
from preprocess.tuple import generate_tuple, convert_tuple_to_id, convert_id_to_postfix
from preprocess.metric import compute_tree_result, compute_tuple_result

pretrain_model_path = 'yechen/bert-base-chinese'
max_text_len = 1024
max_equ_len = 100
batch_size = 16
epochs = 50
lr = 5e-5
embedding_size = 128
# max_grad_norm = 1.0

def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def set_seed(seed=1): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print('seed:', seed)

def generate_graph(max_num_len, nums):
    diag_ele = np.ones(max_num_len)
    graph1 = np.diag(diag_ele)
    for i in range(len(nums)):
        for j in range(len(nums)):
            if nums[i] <= nums[j]:
                graph1[i][j] = 1
            else:
                graph1[j][i] = 1
    graph2 = graph1.T
    return [graph1.tolist(), graph2.tolist()]

def train():
    set_seed()
    for fold in range(4, 5):
        # 加载数据集
        data_root_path = 'data/hmwp/mtokens/'
        train_data = load_data(data_root_path + 'HMWP_fold' + str(fold) + '_train.jsonl')
        dev_data = load_data(data_root_path + 'HMWP_fold' + str(fold) + '_test.jsonl')

        config = AutoConfig.from_pretrained(pretrain_model_path)
        tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)

        tokens_count = Counter()
        max_nums_len = 0
        for d in train_data + dev_data:
            tokens_count += Counter(sum(d['postfix'], []))
            max_nums_len = max(max_nums_len, len(d['nums']))
        tokens = list(tokens_count)
        op_tokens1 = [x for x in tokens if x[0].lower()!='c' and x[0].lower()!='n' and x[0].lower()!='x']
        op_tokens1.append(';')
        constant_tokens = [x for x in tokens if x[0].lower()=='c' and tokens_count[x]>=20]
        number_tokens = ['N_' + str(x) for x in range(max_nums_len)]
        op_tokens1.sort()
        constant_tokens = sorted(constant_tokens, key=lambda x: float(x[2:].replace('_', '.')))
        constant_tokens1 = constant_tokens + [x for x in tokens if x[0].lower()=='x']
        constant_tokens2 = copy.deepcopy(constant_tokens1)
        number_tokens = sorted(number_tokens, key=lambda x: int(x[2:]))
        mtokens = ['<O>', '<Add>', '<Mul>', '<Var>', '<Equ>', '<Sol>']
        tokens1 = op_tokens1 + constant_tokens1 + number_tokens
        tokens_dict1 = {x:i for i,x in enumerate(tokens1)}
        ids_dict1 = {x[1]:x[0] for x in tokens_dict1.items()}

        source_dict2 = {'C:':0, 'N:':1, 'M:':2}
        source_ids_dict2 = {x[1]:x[0] for x in source_dict2.items()}
        op_tokens2 = [x for x in op_tokens1 if x != ';' and x != '=']
        op_tokens2 = op_tokens2 + ['=', '<s>', '</s>']
        op_dict2 = {x:i for i,x in enumerate(op_tokens2)}
        op_ids_dict2 = {x[1]:x[0] for x in op_dict2.items()}
        constant_dict2 = {x:i for i,x in enumerate(constant_tokens2)}
        constant_ids_dict2 = {x[1]:x[0] for x in constant_dict2.items()}

        tokenizer.add_special_tokens({'additional_special_tokens': number_tokens + mtokens})
        number_tokens_ids = [tokenizer.convert_tokens_to_ids(x) for x in number_tokens]
        number_tokens_ids = set(number_tokens_ids)
        train_batches1 = []
        cached = {}
        for d in train_data:
            src = tokenizer.encode(d['text'], max_length=max_text_len)
            tgt1 = [tokens_dict1.get(x, len(op_tokens1)) for x in d['prefix']]
            tgt2 = generate_tuple(d['postfix'], op_tokens2)
            tgt2 = convert_tuple_to_id(tgt2, op_dict2, constant_dict2, source_dict2)
            num = []
            for i,s in enumerate(src):
                if s in number_tokens_ids:
                    num.append(i)
            assert len(num) == len(d['nums']), "数字个数不匹配！%s vs %s" % (len(num), len(d['nums']))
            value = [eval(x) for x in d['nums']]
            
            train_batches1.append((src, num, value, tgt1, tgt2))
            cached[d['id']] = {'src':src, 'num': num, 'value':value}

        train_batches = train_batches1
        dev_batches = []
        for d in dev_data:
            src1 = tokenizer.encode(d['text'], max_length=max_text_len)
            num = []
            for i,s in enumerate(src1):
                if s in number_tokens_ids:
                    num.append(i)
            value = [eval(x) for x in d['nums']]
            dev_batches.append((src1, num, value, d))

        def data_generator(train_batches, batch_size):
            i = 0
            pairs = []
            while i + batch_size < len(train_batches):
                pair = train_batches[i: i+batch_size]
                pairs.append(pair)
                i += batch_size
            pairs.append(train_batches[i:])
            batches = []
            for pair in pairs:
                text_ids, num_ids, graphs, equ_ids, tuple_ids = [], [], [], [], []
                max_text = max([len(x[0]) for x in pair])
                max_num = max([len(x[1]) for x in pair])
                max_equ = max([len(x[3]) for x in pair])
                max_tuple = max([len(x[4]) for x in pair])
                for _, p in enumerate(pair):
                    text, num, value, equ, tuple = p
                    text_ids.append(text + [tokenizer.pad_token_id] * (max_text-len(text)))
                    num_ids.append(num + [-1] * (max_num-len(num)))
                    graphs.append(generate_graph(max_num, value))
                    equ_ids.append(equ + [-1] * (max_equ-len(equ)))
                    tuple_ids.append(tuple + [[-1, -1, -1 ,-1, -1]] * (max_tuple-len(tuple)))
                text_ids = torch.tensor(text_ids, dtype=torch.long)
                num_ids = torch.tensor(num_ids, dtype=torch.long)
                graphs = torch.tensor(graphs, dtype=torch.float)
                equ_ids = torch.tensor(equ_ids, dtype=torch.long)
                tuple_ids = torch.tensor(tuple_ids, dtype=torch.long)
                text_pads = text_ids != tokenizer.pad_token_id
                text_pads = text_pads.float()
                num_pads = num_ids != -1
                num_pads = num_pads.float()
                equ_pads = equ_ids != -1
                equ_ids[~equ_pads] = 0
                equ_pads = equ_pads.float()
                batches.append((text_ids, text_pads, num_ids, num_pads, graphs, equ_ids, equ_pads, tuple_ids))
            return batches

        pretrain_model = AutoModel.from_pretrained(pretrain_model_path)
        pretrain_model.resize_token_embeddings(len(tokenizer))
        encoder = Encoder(pretrain_model)
        treedecoder = TreeDecoder(config, len(op_tokens1), len(constant_tokens1), embedding_size)
        solver = Solver(encoder, treedecoder)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        solver.to(device)

        
        train_generator = data_generator(train_batches, batch_size)
        optimizer = AdamW(solver.parameters(), lr=lr, weight_decay=0.01)
        global_steps = len(train_generator) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=global_steps*0.1, num_training_steps=global_steps)

        # train
        solver.zero_grad()
        log = open('result_hmwp/log_' + str(fold) + '.txt', 'w')
        for e in range(epochs):
            print("epoch:", e)
            solver.train()
            loss_total = 0.0
            random.shuffle(train_batches)
            train_generator = data_generator(train_batches, batch_size)
            bar = tqdm(enumerate(train_generator), total=len(train_generator))
            for step, batch in bar:
                batch = [_.to(device) for _ in batch]
                text_ids, text_pads, num_ids, num_pads, graphs, equ_ids, equ_pads, tuple_ids = batch
                loss = train_double(solver, text_ids, text_pads, num_ids, num_pads, graphs, equ_ids, equ_pads, tuple_ids, op_tokens1, constant_tokens1)
                loss_total += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            loss_total /= len(train_generator)
            log.write("epoch:" + str(e)+ "\tloss:" + str(loss_total) + "\n")
            logger.info(f"epoch: {e} - loss: {loss_total}")
            
            if (e > 0 and e % 10 == 0) or epochs - e < 5:
                solver.eval()

                value_ac = 0
                equation_ac = 0
                eval_total = 0
                all_results = []
                correct_results = []
                wrong_results = []
                bar = tqdm(enumerate(dev_batches), total=len(dev_batches))
                for _,(text1, num, value, d) in bar:
                    text1_ids = torch.tensor([text1], dtype=torch.long)
                    num_ids = torch.tensor([num], dtype=torch.long)
                    graphs = generate_graph(len(num), value)
                    graphs = torch.tensor([graphs], dtype=torch.float)
                    text_pads = text1_ids != tokenizer.pad_token_id
                    text_pads = text_pads.float()
                    num_pads = num_ids != -1
                    num_pads = num_pads.float()
                    batch = [text1_ids, text_pads, num_ids, num_pads, graphs]
                    batch = [_.to(device) for _ in batch]
                    text1_ids, text_pads, num_ids, num_pads, graphs = batch
                    tree_res1 = evaluate_double(solver, text1_ids, text_pads, num_ids, num_pads, graphs,
                                                 op_tokens1, constant_tokens1, op_dict2, max_equ_len, beam_size=3)
                    tree_out1, tree_score1 = tree_res1.out, tree_res1.score
                    tree_out1 = [ids_dict1[x] for x in tree_out1]
                    tree_val_ac1, tree_equ_ac1 = False, False
                    tree_val_ac1, tree_equ_ac1 = compute_tree_result(tree_out1, d['prefix'], d['answer'], d['nums'])
                    scores = [tree_score1]
                    score_index = np.array(scores).argmax()
                    if score_index == 0:
                        val_ac = tree_val_ac1
                        equ_ac = tree_equ_ac1
                    value_ac += val_ac
                    equation_ac += equ_ac
                    eval_total += 1
                    temp = {}
                    temp['id'] = d['id']
                    temp['text'] = d['text']
                    temp['nums'] = d['nums']
                    temp['ans'] = d['answer']
                    temp['prefix'] = d['prefix']
                    temp['postfix'] = d['postfix']
                    temp['tree_out1'] = tree_out1
                    temp['tree_score1'] = tree_score1
                    temp['tree_result1'] = tree_val_ac1
                    if val_ac:
                        correct_results.append(temp)
                    else:
                        wrong_results.append(temp)
                    all_results.append(temp)
                
                f = open('result_hmwp/correct_result_dev_' + str(fold) + '.jsonl', 'w')
                for d in correct_results:
                    json.dump(d, f, ensure_ascii=False)
                    f.write("\n")
                f.close()
                f = open('result_hmwp/wrong_results_dev_' + str(fold) + '.jsonl', 'w')
                for d in wrong_results:
                    json.dump(d, f, ensure_ascii=False)
                    f.write("\n")
                f.close()
                f = open('result_hmwp/all_results_dev_' + str(fold) + '.jsonl', 'w')
                for d in all_results:
                    json.dump(d, f, ensure_ascii=False)
                    f.write("\n")
                f.close()

                log.write("epoch:" + str(e)+ "\tequ_acc:" + str(float(equation_ac) / eval_total) + "\tval_acc:" + str(float(value_ac) / eval_total) + "\n")
                logger.info(f"epoch: {e} - equ_acc: {float(equation_ac) / eval_total} - val_acc: {float(value_ac) / eval_total}")

                solver.save_pretrained('result_hmwp/fold_' + str(fold) +'/models')
                tokenizer.save_pretrained('result_hmwp/fold_' + str(fold) +'/models')
        log.close()

train()
# test(49)