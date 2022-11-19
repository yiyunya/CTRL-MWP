import re
import copy
import json
import sympy
from transformers import BertTokenizer


def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data

def from_infix_to_prefix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    expression = copy.deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res

def from_infix_to_postfix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    for e in expression:
        if e in ["(", "["]:
            st.append(e)
        elif e == ")":
            c = st.pop()
            while c != "(":
                res.append(c)
                c = st.pop()
        elif e == "]":
            c = st.pop()
            while c != "[":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in ["(", "["] and priority[e] <= priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    return res

def from_prefix_to_infix(prefix):
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
            operands = [a, b]
            operands.sort()
            a, b = operands
            st.append(" ".join([a, "+", b]))
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            operands = [a, b]
            operands.sort()
            a, b = operands
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
        return st.pop()
    return None

def from_postfix_to_infix(postfix):
    st = []
    operators = ["+", "-", "^", "*", "/"]
    for p in postfix:
        if p not in operators:
            st.append(p)
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            operands = [a, b]
            operands.sort()
            a, b = operands
            st.append(" ".join([a, "+", b]))
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            operands = [a, b]
            operands.sort()
            a, b = operands
            st.append(" ".join(["(", a, ")", "*", "(", b, ")"]))
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", b, ")", "/", "(", a, ")"]))
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", b, ")", "-", "(", a, ")"]))
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", b, ")", "^", "(", a, ")"]))
        else:
            return None
    if len(st) == 1:
        return st.pop()

def out_expression_list(test, nums):
    res = []
    for i in test:
        if i[0] == 'N':
            res.append(nums[int(i[2:])%len(nums)])
        elif i[0] == 'C':
            res.append(i[2:].replace('_', '.'))
        else:
            res.append(i)
    return res

def compute_prefix_expression(pre_fix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    pre_fix = copy.deepcopy(pre_fix)
    pre_fix.reverse()
    for p in pre_fix:
        if p not in operators:
            st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if b == 0:
                return None
            st.append(a / b)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a - b)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
#            if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
#                return None
            st.append(a ** b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None

# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# def from_postfix_to_tree(postfix):
#     st = list()
#     operators = ["+", "-", "^", "*", "/"]
#     for p in postfix:
#         if p not in operators:
#             st.append(TreeNode(p))
#         elif p == "+" and len(st) > 1:
#             a = st.pop()
#             b = st.pop()
#             st.append(TreeNode(p, b, a))
#         elif p == "*" and len(st) > 1:
#             a = st.pop()
#             b = st.pop()
#             st.append(TreeNode(p, b, a))
#         elif p == "/" and len(st) > 1:
#             a = st.pop()
#             b = st.pop()
#             st.append(TreeNode(p, b, a))
#         elif p == "-" and len(st) > 1:
#             a = st.pop()
#             b = st.pop()
#             st.append(TreeNode(p, b, a))
#         elif p == "^" and len(st) > 1:
#             a = st.pop()
#             b = st.pop()
#             st.append(TreeNode(p, b, a))
#     return st.pop()


# def inorderTraversal(root):
#     ans = []
#     def dfs(path, node):
#         if not node:
#             ans.append(path)
#             return
#         dfs(path, node.left)
#         path.append(node.val)
#         dfs(path, node.right)
    
#     dfs([], root)
#     return ans[0]


# def preorderTraversal(root):
#     ans = []
#     def dfs(root):
#         if not root:
#             return
#         ans.append(root.val)
#         dfs(root.left)
#         dfs(root.right)
    
#     dfs(root)
#     return ans


# def postorderTraversal(root):
#     ans = []
#     def dfs(root):
#         if not root:
#             return
#         dfs(root.left)
#         dfs(root.right)
#         ans.append(root.val)
    
#     dfs(root)
#     return ans

def transfer_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    for d in data:
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                # input_seq.append('_'+s[pos.start():pos.end()]+'[N]')
                # input_seq.append("NUM")
                input_seq.append("N_" + str(len(nums)-1))
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        nums_fraction = []
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) >= 1:
                        res.append("N_"+str(nums.index(n)))
                    else:
                        n = "C_" + n
                        n = n.replace('.', '_')
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) >= 1:
                    res.append("N_"+str(nums.index(st_num)))
                else:
                    st_num = "C_" + st_num
                    st_num = st_num.replace('.', '_')
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        out_seq = ' '.join(out_seq)
        out_seq = out_seq.replace('[', '(')
        out_seq = out_seq.replace(']', ')')
        out_seq = out_seq.split(' ')
        num_values = []
        for p in nums:
            pos1 = re.search("\d+\(", p)
            pos2 = re.search("\)\d+", p)
            if pos1:
                num_values.append(str(eval(p[pos1.start(): pos1.end() - 1] + "+" + p[pos1.end() - 1:])))
            elif pos2:
                num_values.append(str(eval(p[:pos2.start() + 1] + "+" + p[pos2.start() + 1: pos2.end()])))
            elif p[-1] == "%":
                num_values.append(str(float(p[:-1]) / 100))
            else:
                num_values.append(str(eval(p)))
        temp = {}
        temp['id'] = d['id']
        # temp['answer'] = d['ans']
        temp['text'] = ''.join(input_seq)
        temp['original_text'] = d['original_text']
        temp['infix'] = out_seq
        prefix = from_infix_to_prefix(out_seq)
        postfix = from_infix_to_postfix(out_seq)
        ans = compute_prefix_expression(out_expression_list(prefix, num_values))
        if ans is None:
            print(d)
            continue
        # tree = from_postfix_to_tree(postfix)
        # if ' '.join(preorderTraversal(tree)) != ' '.join(prefix) or ' '.join(postorderTraversal(tree)) != ' '.join(postfix):
        #     print(d)
        #     continue
        temp['prefix'] = from_infix_to_prefix(from_prefix_to_infix(prefix).split(' '))
        temp['postfix'] = from_infix_to_postfix(from_postfix_to_infix(postfix).split(' '))

        temp['nums'] = num_values
        temp['answer'] = ans
        pairs.append(temp)
    return pairs
        
train_data = load_raw_data('../data/Math_23K_train.json')
dev_data = load_raw_data('../data/Math_23K_valid.json')
test_data = load_raw_data('../data/Math_23K_test.json')
train_data = transfer_num(train_data)
dev_data = transfer_num(dev_data)
test_data = transfer_num(test_data)
f = open('../data/Math23K_train_norm.jsonl', 'w')
for d in train_data:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()
f = open('../data/Math23K_dev_norm.jsonl', 'w')
for d in dev_data:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()
f = open('../data/Math23K_test_norm.jsonl', 'w')
for d in test_data:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()