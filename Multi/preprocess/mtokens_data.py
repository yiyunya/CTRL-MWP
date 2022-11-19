import copy
import json
import random

from .preprocess import from_infix_to_prefix

def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def generate_prefix_add(prefix):
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
            st.append(" ".join([b, "+", a]))
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
        res = from_infix_to_prefix(res.split(" "))
        if res != prefix[::-1]:
            return res
    return None

def generate_prefix_mul(prefix):
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
            st.append(" ".join([a, "+", b]))
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
        res = from_infix_to_prefix(res.split(" "))
        if res != prefix[::-1]:
            return res
    return None

def generate_prefix_mtokens(train_data):
    train_batches = []
    for d in train_data:
        prefix = copy.deepcopy(d['prefix'])
        prefix_add = generate_prefix_add(prefix)
        prefix_mul = generate_prefix_mul(prefix)
        if prefix_add:
            train_batches.append((d['id'], '<Add>', prefix_add))
        if prefix_mul:
            train_batches.append((d['id'], '<Mul>', prefix_mul))
    return train_batches