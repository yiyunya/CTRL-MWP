import copy
import json
import random

from .preprocess import from_infix_to_prefix, from_infix_to_postfix

def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def generate_add(prefix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    prefix = copy.deepcopy(prefix)
    prefix.reverse()
    for p in prefix:
        if p not in operators:
            st.append(p)
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            # st.append(" ".join([b, "+", a]))
            st.append(" ".join(["(", b, ")", "+", "(", a, ")"]))
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", a, ")", "*", "(", b, ")"]))
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", a, ")", "/", "(", b, ")"]))
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", a, ")", "-", "(", b, ")"]))
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", a, ")", "^", "(", b, ")"]))
        else:
            return None
    if len(st) == 1:
        res = st.pop()
        res1 = from_infix_to_prefix(res.split(" "))
        res2 = from_infix_to_postfix(res.split(" "))
        if res1 != prefix[::-1]:
            return res1, res2
    return None, None

def generate_mul(prefix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    prefix = copy.deepcopy(prefix)
    prefix.reverse()
    for p in prefix:
        if p not in operators:
            st.append(p)
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            # st.append(" ".join([a, "+", b]))
            st.append(" ".join(["(", a, ")", "+", "(", b, ")"]))
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", b, ")", "*", "(", a, ")"]))
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", a, ")", "/", "(", b, ")"]))
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", a, ")", "-", "(", b, ")"]))
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", a, ")", "^", "(", b, ")"]))
        else:
            return None
    if len(st) == 1:
        res = st.pop()
        res1 = from_infix_to_prefix(res.split(" "))
        res2 = from_infix_to_postfix(res.split(" "))
        if res1 != prefix[::-1]:
            return res1, res2
    return None, None


def generate_mtokens(train_data):
    train_batches = []
    for d in train_data:
        prefix = copy.deepcopy(d['prefix'])
        prefix_add, postfix_add = generate_add(prefix)
        prefix_mul, postfix_mul = generate_mul(prefix)
        if prefix_add and '+' in d['prefix']:
            train_batches.append((d['id'], '<Add>', prefix_add, postfix_add))
        if prefix_mul and '*' in d['prefix']:
            train_batches.append((d['id'], '<Mul>', prefix_mul, postfix_mul))
    return train_batches