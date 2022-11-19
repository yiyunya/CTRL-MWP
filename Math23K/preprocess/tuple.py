import json

def generate_tuple(formulae, op_tokens):
    memories = []
    for expr in formulae:
        normalized = []
        for token in expr:
            if token[0].lower() == 'n':
                token = int(token[2:])
                normalized.append(('N:', token))
            elif token[0].lower() in ['c', 'x']:
                normalized.append(('C:', token))
            else:
                normalized.append(token)
        stack = []
        for tok in normalized:
            if tok in op_tokens:
                args = stack[-2:]
                stack = stack[:-2]
                stack.append(('M:', len(memories)))
                memories.append((tok, args))
            else:
                stack.append(tok)
    return memories

def convert_tuple_to_id(tuple, op_dict, constant_dict, source_dict):
    res = [[op_dict['<s>'], -1, -1, -1, -1]]
    for expression in tuple:
        operator, operand = expression
        operator = -1 if operator is None else op_dict[operator]
        new_operands = []
        for src, a in operand:
            # if src is None:
            #     new_operands += [-1, -1]
            # else:
            new_operands.append(source_dict[src])
            if src == 'C:':
                new_operands.append(constant_dict.get(a, 0))
            else:
                new_operands.append(a)
        res.append([operator] + new_operands)
    res.append([op_dict['</s>'], -1, -1, -1, -1])
    return res


def convert_id_to_expression(item, op_ids_dict, constant_ids_dict, source_ids_dict):
    expression = []
    for token in item:
        operator = op_ids_dict[token[0]]
        if operator == '<s>':
            expression.clear()
            continue
        if operator == '</s>':
            break

        operands = []
        for i in range(1, len(token), 2):
            src = token[i]
            if src != -1:
                src = source_ids_dict[src]
                operand = token[i + 1]
                if src == 'C:':
                    operand = constant_ids_dict[operand]
                operands.append((src, operand))

        expression.append((operator, operands))

    return expression

def convert_id_to_postfix(item, op_ids_dict, constant_ids_dict, source_ids_dict):
    item = convert_id_to_expression(item, op_ids_dict, constant_ids_dict, source_ids_dict)
    computation_history = []
    expression_used = []

    for operator, operands in item:
        computation = []
        for src, operand in operands:
            if src == 'N:':
                computation.append('N_' + str(operand))
            elif src == 'M:':
                if operand < len(computation_history):
                    computation += computation_history[operand]
                    expression_used[operand] = True
                else:
                    computation.append(constant_ids_dict[0])
            else:
                computation.append(operand)
        computation.append(operator)

        computation_history.append(computation)
        expression_used.append(False)

    computation_history = [equation for used, equation in zip(expression_used, computation_history) if not used]

    return sum(computation_history, [])