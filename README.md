# Seeking Diverse Reasoning Logic: Controlled Equation Expression Generation for Solving Math Word Problems.
Shen Y.*, Liu Q.*, Mao Z., Wan Z., Cheng F., Sadao K. (2022)

This paper has been accepted for publication in *AACL-IJCNLP 2022*.
The arxiv preprint could be found [here](https://arxiv.org/abs/2209.10310).

To solve Math Word Problems, human students leverage diverse reasoning logic that reaches different possible equation solutions. However, the mainstream sequence-to-sequence approach of automatic solvers aims to decode a fixed solution equation supervised by human annotation. In this paper, we propose a controlled equation generation solver by leveraging a set of control codes to guide the model to consider certain reasoning logic and decode the corresponding equations expressions transformed from the human reference. The empirical results suggest that our method universally improves the performance on single-unknown (Math23K) and multiple-unknown (DRAW1K, HMWP) benchmarks, with substantial improvements up to 13.2% accuracy on the challenging multiple-unknown datasets.


## Data

All data used for this paper could be found in /Math23K/data folder and /Multi/data folder.

The diverse expression data is created in-run for Math23K, and is generated seperately for Multiple Unknown datasets via mtokens.py for DRAW1K and mtokens_chinese.py for HMWP.


## Code

### Math23K Reproduction

For reproducing Math23K results, go to the /Math23K folder and please run:

```
python main.py
```


### DRAW1K Reproduction

For reproducing DRAW1K results, go to the /Multi folder and please run:

```
python main_draw.py
```


### HMWP Reproduction

For reproducing HMWP results, go to the /Multi folder and please run:

```
python main_hmwp.py
```

## Citation



If you find this repo useful, please cite the following paper:

```
@misc{https://doi.org/10.48550/arxiv.2209.10310,
  doi = {10.48550/ARXIV.2209.10310},
  url = {https://arxiv.org/abs/2209.10310},
  author = {Shen, Yibin and Liu, Qianying and Mao, Zhuoyuan and Wan, Zhen and Cheng, Fei and Kurohashi, Sadao},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Seeking Diverse Reasoning Logic: Controlled Equation Expression Generation for Solving Math Word Problems},
  publisher = {arXiv},
  year = {2022},
}
```