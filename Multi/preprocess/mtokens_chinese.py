import copy
import json
import sympy

def from_infix_to_postfix(expression):
    st = list()
    res = list()
    priority = {"=":-1, "+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
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

def generate_add(postfix):
    st = list()
    operators = ["+", "-", "^", "*", "/", "="]
    for p in postfix:
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
            st.append(" ".join(["(", b, ")", "/", "(", a, ")"]))
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", b, ")", "-", "(", a, ")"]))
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", b, ")", "^", "(", a, ")"]))
        elif p == "=" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join([b, "=", a]))
        else:
            return None
    if len(st) == 1:
        res = st.pop()
        res = from_infix_to_postfix(res.split(" "))
        return res
    return None

def generate_mul(postfix):
    st = list()
    operators = ["+", "-", "^", "*", "/", "="]
    for p in postfix:
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
            st.append(" ".join(["(", b, ")", "/", "(", a, ")"]))
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", b, ")", "-", "(", a, ")"]))
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", b, ")", "^", "(", a, ")"]))
        elif p == "=" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join([b, "=", a]))
        else:
            return None
    if len(st) == 1:
        res = st.pop()
        res = from_infix_to_postfix(res.split(" "))
        return res
    return None

def generate_var(postfix):
    res = []
    for p in postfix:
        if p == 'X_0':
            res.append('X_1')
        elif p == 'X_1':
            res.append('X_0')
        else:
            res.append(p)
    return res

def from_postfix_to_infix(postfix):
    st = []
    operators = ["+", "-", "^", "*", "/", "=", ';']
    for p in postfix:
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
            st.append(" ".join(["(", b, ")", "*", "(", a, ")"]))
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
        elif p == "=" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join([b, "=", a]))
        elif p == ";" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join([b, ";", a]))
        else:
            return None
    if len(st) == 1:
        return st.pop()

def from_C_to_number(st):
    ans = []
    for s in st:
        if s[:2] == "C_":
            if s == "C_-1":
                ans.append("-1")
            elif s in ["C_0", "C_1"]:
                ans.append(s[-1])
            else:
                ans.append(s)
        else:
            ans.append(s)
    return " ".join(ans)

def from_number_to_C(st):
    st = norm_equation(st)
    if st[0] == "-":
        st = ["C_-1", "*"] + st[1:]
    ans = []
    for s in st:
        if s[0] == "-" and len(s) > 1:
            ans += ["C_-1", "*"]
            s = s[1:]
        if s.isdigit() or "." in s:
            temp = "C_" + s
            temp = temp.replace(".", "_")
            if temp.count("_") == 1:
                temp += "_0"
            ans += [temp]
        else:
            ans += [s]
    return " ".join(ans)

def norm_equation(st):
    import re
    ans = []
    pos_st = re.search("-\w_\d+_\d+|\w_\d+_\d+|-\d+\.\d+|\d+\.\d+|-\w_\d+|\w_\d+|-\d+|\d+", st)

    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            ans += norm_equation(st[:p_start])
        st_num = st[p_start:p_end]
        ans.append(st_num)
        if p_end < len(st):
            ans += norm_equation(st[p_end:])
        return ans
    
    for ss in st:
        if ss != " ":
            ans.append(ss)
    return ans

def generate_solution(postfix):
    res = []
    if len(postfix) == 1:
        x = sympy.symbols("X_0")
        expr1 = from_C_to_number(from_postfix_to_infix(postfix[0]).split(" "))
        equal1 = expr1.find("=")
        x_sy = sympy.Eq(sympy.sympify(expr1[:equal1]), sympy.sympify(expr1[equal1+1:]))
        try:
            x_ans = sympy.solve(x_sy, x)[0]
            x_ans = str(x_ans).replace("**", "^")
            if "sqrt" not in x_ans:
                x_ans = "X_0 = " + from_number_to_C(x_ans)
                res.append(from_infix_to_postfix(x_ans.split(" ")))
            else:
                res = None
        except:
            res = None
    
    elif len(postfix) == 2:
        expr1 = from_C_to_number(from_postfix_to_infix(postfix[0]).split(" "))
        expr2 = from_C_to_number(from_postfix_to_infix(postfix[1]).split(" "))
        equal1 = expr1.find("=")
        equal2 = expr2.find("=")
        x = sympy.symbols("X_0")
        y = sympy.symbols("X_1")
        x_sy = expr1[:equal1] + "- (" + expr1[equal1+1:] + " )"
        y_sy = expr2[:equal2] + "- (" + expr2[equal2+1:] + " )"
        try:
            ans = sympy.solve([x_sy, y_sy], [x, y])
            x_ans, y_ans = ans[x], ans[y]
            x_ans = str(x_ans).replace("**", "^")
            y_ans = str(y_ans).replace("**", "^")
            if "sqrt" not in x_ans and "sqrt" not in y_ans:
                x_ans = "X_0 = " + from_number_to_C(x_ans)
                y_ans = "X_1 = " + from_number_to_C(y_ans)
                res.append(from_infix_to_postfix(x_ans.split(" ")))
                res.append(from_infix_to_postfix(y_ans.split(" ")))
            else:
                res = None
        except:
            res = None    
    return res

def out_expression_list(test, nums):
    res = []
    for i in test:
        if i[0] == 'N':
            res.append(nums[int(i[2:])%len(nums)])
        elif i[0] == 'C':
            if i == 'C_-1':
                res.append('-1')
            else:
                res.append(i[2:].replace('_', '.'))
        else:
            res.append(i)
    return res


def compute_ans(ans_infix):
    equs = ans_infix.split(' ; ')
    if len(equs) == 1:
        equal_pos1 = equs[0].find("=")
        infix1_sy = sympy.Eq(sympy.sympify(equs[0][:equal_pos1]), sympy.sympify(equs[0][equal_pos1+1:]))
        X_0 = sympy.symbols("X_0")
        infix_res = sympy.solve([infix1_sy], [X_0])
    else:
        equal_pos1 = equs[0].find("=")
        equal_pos2 = equs[1].find("=")
        infix1_sy = sympy.Eq(sympy.sympify(equs[0][:equal_pos1]), sympy.sympify(equs[0][equal_pos1+1:]))
        infix2_sy = sympy.Eq(sympy.sympify(equs[1][:equal_pos2]), sympy.sympify(equs[1][equal_pos2+1:]))
        X_0 = sympy.symbols("X_0")
        X_1 = sympy.symbols("X_1")
        infix_res = sympy.solve([infix1_sy, infix2_sy], [X_0, X_1])
    return infix_res

def check_solution(postfix, num_values, real_ans):
    postfix = sum(postfix, [])
    postfix += [';'] * (postfix.count('=')-1)
    ans_infix = from_postfix_to_infix(out_expression_list(postfix, num_values))
    ans = compute_ans(ans_infix)
    if type(ans) == type(dict()):
        ans = list(ans.values())
    else:
        new_ans = []
        for a in ans:
            for b in a:
                new_ans.append(b)
        ans = new_ans
    ans = [float(x) for x in ans]
    ans.sort()
    res = True
    if len(ans) != len(real_ans):
        res = False
    if res:
        for i,value in enumerate(ans):
            if abs(ans[i] - real_ans[i]) > 1e-3:
                res = False
    return res
        
def generate_train_mtokens(train_data):
    train_batches = []
    for d in train_data:
        postfix = copy.deepcopy(d['postfix'])
        postfix_add = [generate_add(x) for x in postfix]
        if postfix_add == postfix:
            postfix_add = None
        postfix_mul = [generate_mul(x) for x in postfix]
        if postfix_mul == postfix:
            postfix_mul = None
        if len(postfix) > 1:
            postfix_var =[generate_var(x) for x in postfix]
            postfix_equ = postfix[::-1]
        else:
            postfix_var = None
            postfix_equ = None
        if len(d['answer']) == len(postfix):
            postfix_solution = generate_solution(postfix)
            if postfix_solution:
                res = check_solution(postfix_solution, d['nums'], d['answer'])
                if not res:
                    print(d)
                    postfix_solution = None
        else:
            postfix_solution = None
        
        temp = copy.deepcopy(d)
        temp['text'] = '<O>' + d['text']
        temp['postfix'] = postfix
        train_batches.append(temp)
        if postfix_add and '+' in d['prefix']:
            temp = copy.deepcopy(d)
            temp['text'] = '<Add>' + d['text']
            temp['postfix'] = postfix_add
            train_batches.append(temp)
        if postfix_mul and '*' in d['prefix']:
            temp = copy.deepcopy(d)
            temp['text'] = '<Mul>' + d['text']
            temp['postfix'] = postfix_mul
            train_batches.append(temp)
        if postfix_var:
            temp = copy.deepcopy(d)
            temp['text'] = '<Var>' + d['text']
            temp['postfix'] = postfix_var
            train_batches.append(temp)
        if postfix_equ:
            temp = copy.deepcopy(d)
            temp['text'] = '<Equ>' + d['text']
            temp['postfix'] = postfix_equ
            train_batches.append(temp)
        if postfix_solution:
            temp = copy.deepcopy(d)
            temp['text'] = '<Sol>' + d['text']
            temp['postfix'] = postfix_solution
            train_batches.append(temp)
    return train_batches
        
def generate_test_mtokens(test_data):
    test_batches = []
    for d in test_data:
        temp = copy.deepcopy(d)
        temp['text'] = '<O>' + d['text']
        test_batches.append(temp)
    return test_batches

def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

for fold in range(5):
    fold = str(fold)
    trainname = '../data/hmwp/5-fold/HMWP_fold' + fold + '_train.jsonl'
    testname = '../data/hmwp/5-fold/HMWP_fold' + fold + '_test.jsonl'
    train_data = load_data(trainname)
    test_data = load_data(testname)
    train_data_mtokens = generate_train_mtokens(train_data)
    test_data_mktokens = generate_test_mtokens(test_data)
    f = open('../data/hmwp/mtokens/HMWP_fold' + fold + '_train.jsonl', 'w')
    for d in train_data_mtokens:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()
    f = open('../data/hmwp/mtokens/HMWP_fold' + fold + '_test.jsonl', 'w')
    for d in test_data_mktokens:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()
    print(fold)