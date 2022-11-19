import re
import copy
import sympy
# from func_timeout import func_set_timeout

import time
import signal
 
 
class TimeoutError(Exception):
    def __init__(self, msg):
        super(TimeoutError, self).__init__()
        self.msg = msg
 
 
def time_out(interval, callback):
    def decorator(func):
        def handler(signum, frame):
            raise TimeoutError("run func timeout")
 
        def wrapper(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(interval)
                result = func(*args, **kwargs)
                signal.alarm(0)
                return result
            except Exception as e:
                callback(e)
        return wrapper
    return decorator
 
 
def timeout_callback(e):
    print(e.msg)


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

@time_out(10, timeout_callback)
def compute_ans(ans_infix):
    equs = ans_infix.split(' ; ')
    if len(equs) == 1:
        equal_pos1 = equs[0].find("=")
        infix1_sy = sympy.Eq(sympy.sympify(equs[0][:equal_pos1]), sympy.sympify(equs[0][equal_pos1+1:]))
        X_0 = sympy.symbols("X_0")
        infix_res = sympy.solve([infix1_sy], [X_0])
    elif len(equs) == 2:
        equal_pos1 = equs[0].find("=")
        equal_pos2 = equs[1].find("=")
        infix1_sy = sympy.Eq(sympy.sympify(equs[0][:equal_pos1]), sympy.sympify(equs[0][equal_pos1+1:]))
        infix2_sy = sympy.Eq(sympy.sympify(equs[1][:equal_pos2]), sympy.sympify(equs[1][equal_pos2+1:]))
        X_0 = sympy.symbols("X_0")
        X_1 = sympy.symbols("X_1")
        infix_res = sympy.solve([infix1_sy, infix2_sy], [X_0, X_1])
    else:
        return None
    return infix_res

def from_prefix_to_infix(prefix):
    st = []
    operators = ["+", "-", "^", "*", "/", "=", ";"]
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
        elif p == "=" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join([a, "=", b]))
        elif p == ";" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join([a, ";", b]))
        else:
            return None
    if len(st) == 1:
        return st.pop()

def from_postfix_to_infix(postfix):
    st = []
    operators = ["+", "-", "^", "*", "/", "=", ";"]
    for p in postfix:
        if p not in operators:
            st.append(p)
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join([b, "+", a]))
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

@time_out(10, timeout_callback)
def compute_tree_result(test, tar, ans, nums):
    if test == tar:
        return True, True
    test = out_expression_list(test, nums)
    if test is None:
        return False, False
    test = from_prefix_to_infix(test)
    if test is None:
        return False, False
    try:
        test = compute_ans(test)
        if type(test) == type(dict()):
            test_ans = list(test.values())
            test_ans = [float(x) for x in test_ans]       
        else:
            test_ans = []
            for a in test:
                for b in a:
                    test_ans.append(b)
        test_ans.sort()
        if len(test_ans) != len(ans):
            return False, False
        result = True
        for i,value in enumerate(ans):
            if abs(test_ans[i] - ans[i]) > 1e-3:
                result = False
                break
        if result:
            return True, False
        else:
            return False, False
    except:
        return False, False

@time_out(10, timeout_callback) 
def compute_tuple_result(test, tar, ans, nums):
    if test == tar:
        return True, True
    test = out_expression_list(test, nums)
    if test is None:
        return False, False
    test = from_postfix_to_infix(test)
    if test is None:
        return False, False
    try:
        test = compute_ans(test)
        if type(test) == type(dict()):
            test_ans = list(test.values())
            test_ans = [float(x) for x in test_ans]       
        else:
            test_ans = []
            for a in test:
                for b in a:
                    test_ans.append(b)
        test_ans.sort()
        if len(test_ans) != len(ans):
            return False, False
        result = True
        for i,value in enumerate(ans):
            if abs(test_ans[i] - ans[i]) > 1e-3:
                result = False
                break
        if result:
            return True, False
        else:
            return False, False
    except:
        return False, False