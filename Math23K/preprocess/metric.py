import re
import copy

def out_expression_list(test, nums):
    try:
        res = []
        for i in test:
            if len(i) > 1 and i[0].lower() == 'n':
                res.append(nums[int(i[2:])%len(nums)])
            elif len(i) > 1 and i[0].lower() == 'c':
                res.append(i[2:].replace('_', '.'))
            else:
                res.append(i)
        return res
    except:
        return None

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
            st.append(a ** b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None

def compute_infix_expression(in_fix):
    in_fix = copy.deepcopy(in_fix)
    in_fix = ''.join(in_fix)
    in_fix = in_fix.replace('[', '(')
    in_fix = in_fix.replace(']', ')')
    in_fix = in_fix.replace('^', '**')
    try:
        return eval(in_fix)
    except:
        return None

def compute_postfix_expression(post_fix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    for p in post_fix:
        if p not in operators:
            st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(b + a)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(b * a)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if a == 0:
                return None
            st.append(b / a)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(b - a)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(b ** a)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None

def compute_prefix_tree_result(test, tar, ans, nums):
    test = out_expression_list(test, nums)
    tar = out_expression_list(tar, nums)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - ans) < 1e-3:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar

def compute_infix_tree_result(test, tar, ans, nums):
    test = out_expression_list(test, nums)
    tar = out_expression_list(tar, nums)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_infix_expression(test) - ans) < 1e-3:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar

def compute_postfix_tree_result(test, tar, ans, nums):
    test = out_expression_list(test, nums)
    tar = out_expression_list(tar, nums)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_postfix_expression(test) - ans) < 1e-3:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar