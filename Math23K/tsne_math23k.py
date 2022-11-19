import json
import torch
import numpy as np
from collections import Counter
from sklearn import manifold, datasets
from sklearn.metrics import calinski_harabasz_score
import matplotlib
import matplotlib.pyplot as plt

def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

alldata = {}
data = load_data('result/embeddings.jsonl')
for d in data:
    if d['id'] not in alldata:
        alldata[d['id']] = d
    else:
        alldata[d['id']]['embedding'] += d['embedding']

expr = {}
for k,d in alldata.items():
    if d['prefix'] not in expr:
        expr[d['prefix']] = 1
    else:
        expr[d['prefix']] += 1
expr = sorted(expr.items(), key=lambda x: -x[1])
expr = [x[0] for x in expr]
expr_dict = {x:i for i,x in enumerate(expr)} 
X, y = list(), list()
for k,d in alldata.items():
    X.append(np.array(d['embedding'][0]))
    y.append(d['prefix'])
X, y = np.array(X), np.array(y)
yc = [plt.cm.Set1(expr_dict[x]) for x in y]

tsne = manifold.TSNE(n_components=2, init='pca', random_state=1)
X_tsne = tsne.fit_transform(X)
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)

plt.scatter(X_norm[:,0], X_norm[:,1], c=yc)
plt.show()

print(calinski_harabasz_score(X_norm, y))