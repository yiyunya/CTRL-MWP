import copy
import torch
import torch.nn as nn
from .tree import TreeNode, TreeEmbedding, TreeBeam, copy_list

def train_distance(distance_model, text1_ids, text1_pads, text2_ids, text2_pads, labels, mean, std):
    preds = distance_model(text1_ids, text1_pads, text2_ids, text2_pads)
    preds = preds * std + mean
    losses = torch.sum((preds-labels)**2)
    loss = losses / text1_ids.size(0)
    return loss

def evaluate_distance(distance_model, text1_ids, text1_pads, text2_ids, text2_pads, labels, mean, std):
    preds = distance_model(text1_ids, text1_pads, text2_ids, text2_pads)
    preds = preds * std + mean
    losses = torch.sum((preds-labels)**2)
    return losses

class Solver(nn.Module):
    def __init__(self, encoder, decoder1):
        super().__init__()
        self.encoder = encoder
        # self.grapher = grapher
        self.decoder1 = decoder1
        # self.decoder2 = decoder2
    
    def save_pretrained(self, save_directory):
        self.encoder.save_pretrained(save_directory)
        # torch.save(self.grapher.state_dict(), save_directory + "/grapher.pt")
        torch.save(self.decoder1.state_dict(), save_directory + "/decoder1.pt")
        # torch.save(self.decoder2.state_dict(), save_directory + "/decoder2.pt")

def masked_cross_entropy(logits, target, mask):
    target[~mask] = 0
    logits_flat = logits.reshape(-1, logits.size(-1))
    target_flat = target.reshape(-1, 1)
    losses_flat = torch.gather(logits_flat, index=target_flat, dim=1)
    losses = losses_flat.reshape(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / logits.size(0)
    return loss

def train_tree(solver, encoded, text_ids, text_pads, num_ids, num_pads, equ_ids, equ_pads, op_tokens, constant_tokens):
    # encoded = solver.encoder(text_ids, text_pads, num_ids, num_pads)
    encoder_outputs = encoded['text'].transpose(0, 1)
    problem_output = encoder_outputs[0]
    all_nums_encoder_outputs = encoded['num']

    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
    batch_size, max_target_length = equ_ids.shape
    all_node_outputs = []
    num_start = len(op_tokens)
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    padding_hidden = torch.zeros(1, solver.decoder1.predict.hidden_size, dtype=encoder_outputs.dtype, device=encoder_outputs.device)
    constant_pads = torch.ones(batch_size, len(constant_tokens), dtype=encoder_outputs.dtype, device=encoder_outputs.device)
    operand_pads = torch.cat((constant_pads, num_pads), dim=1)
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = solver.decoder1.predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, 1-text_pads, 1-operand_pads)

        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        generate_input = equ_ids[:, t].clone()
        generate_input[generate_input >= len(op_tokens)] = 0
        left_child, right_child, node_label = solver.decoder1.generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, equ_ids[:, t].contiguous().tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = solver.decoder1.merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    target = equ_ids.clone()
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
    all_node_outputs = all_node_outputs.to(target.device)
    all_node_outputs = -torch.log_softmax(all_node_outputs, dim=-1)
    loss = masked_cross_entropy(all_node_outputs, target, equ_pads.bool())
    return loss  # , loss_0.item(), loss_1.item()

def evaluate_tree(solver, encoded, text_ids, text_pads, num_ids, num_pads, op_tokens, constant_tokens, max_length, beam_size=3):
    # encoded = solver.encoder(text_ids, text_pads, num_ids, num_pads)
    encoder_outputs = encoded['text'].transpose(0, 1)
    problem_output = encoder_outputs[0]
    all_nums_encoder_outputs = encoded['num']
    batch_size = text_ids.size(0)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
    num_start = len(op_tokens)
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]
    padding_hidden = torch.zeros(1, solver.decoder1.predict.hidden_size, dtype=encoder_outputs.dtype, device=encoder_outputs.device)
    constant_pads = torch.ones(batch_size, len(constant_tokens), dtype=num_pads.dtype, device=encoder_outputs.device)
    operand_pads = torch.cat((constant_pads, num_pads), dim=1)

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = solver.decoder1.predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, 1-text_pads, 1-operand_pads)

            out_score = torch.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            topv, topi = out_score.topk(beam_size)

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    generate_input = generate_input.to(encoder_outputs.device)
                    left_child, right_child, node_label = solver.decoder1.generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = solver.decoder1.merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0]

def train_tuple(solver, encoded, text_ids, text_pads, num_ids, num_pads, tuple_ids):
    # encoded = solver.encoder(text_ids, text_pads, num_ids, num_pads)
    text, text_pad, text_num, text_numpad = encoded['text'], 1-encoded['text_pads'], encoded['num'], 1-encoded['num_pads']
    outputs, targets = solver.decoder2(text, text_pad, text_num, text_numpad, tuple_ids)
    result = {'loss': 0.0}
    for key in targets:
        logits = -outputs[key][:, :-1].clone()
        target = targets[key][:, 1:].clone()
        mask = target != -1
        result['loss'] += masked_cross_entropy(logits, target, mask)
    return result['loss']

def evaluate_tuple(solver, encoded, text_ids, text_pads, num_ids, num_pads, op_dict2, max_length, beam_size=3):
    # encoded = solver.encoder(text_ids, text_pads, num_ids, num_pads)
    text, text_pad, text_num, text_numpad = encoded['text'], 1-encoded['text_pads'], encoded['num'], 1-encoded['num_pads']
    batch_size = text.size(0)
    batch_range = range(batch_size)
    device = text.device

    con_offset = 0
    num_offset = solver.decoder2.constant_size
    mem_offset = num_offset + text_num.shape[1]
    con_range = lambda n: n < num_offset
    num_range = lambda n: num_offset <= n < mem_offset
    function_arities = {}
    init = [len(op_dict2)-2] + [-1] * (4)
    result = torch.tensor([[[init]] for _ in batch_range], dtype=torch.long)
    beamscores = torch.zeros(batch_size, 1)
    all_exit = False
    seq_len = 1
    while seq_len < max_length and not all_exit:
        scores = solver.decoder2(text, text_pad, text_num, text_numpad, result.to(device))
        scores = {key: score[:, :, -1].cpu().detach() for key, score in scores.items()}
        beam_function_score = scores['operator'] + beamscores.unsqueeze(-1)
        next_beamscores = torch.zeros(batch_size, beam_size)
        next_result = torch.full((batch_size, beam_size, seq_len + 1, 1 + 4), fill_value=-1, dtype=torch.long)
        beam_range = range(beam_function_score.shape[1])
        operator_range = range(beam_function_score.shape[2])
        for i in batch_range:
            score_i = []
            for m in beam_range:
                last_item = result[i, m, -1, 0].item()
                after_last = last_item in {-1, len(op_dict2)-1}

                if after_last:
                    score_i.append((beamscores[i, m].item(), m, -1, []))
                    continue

                operator_scores = {}
                for f in operator_range:
                    operator_score = beam_function_score[i, m, f].item()

                    if f >= len(op_dict2)-2:
                        if f == len(op_dict2)-1 and last_item == len(op_dict2)-2:
                            continue
                        score_i.append((operator_score, m, f, []))
                    else:
                        operator_scores[f] = operator_score

                operand_beams = [(0.0, [])]
                for a in range(2):
                    score_ia, index_ia = scores['operand_%s' % a][i, m].topk(beam_size)
                    score_ia = score_ia.tolist()
                    index_ia = index_ia.tolist()
                    operand_beams = [(s_prev + s_a, arg_prev + [arg_a])
                                        for s_prev, arg_prev in operand_beams
                                        for s_a, arg_a in zip(score_ia, index_ia)]
                    operand_beams = sorted(operand_beams, key=lambda t: t[0], reverse=True)[:beam_size]
                    for f, s_f in operator_scores.items():
                        if function_arities.get(f, 2) == a + 1:
                            score_i += [(s_f + s_args, m, f, args) for s_args, args in operand_beams]

            beam_registered = set()
            for score, prevbeam, operator, operands in sorted(score_i, key=lambda t: t[0], reverse=True):
                if len(beam_registered) == beam_size:
                    break
                beam_signature = (prevbeam, operator, *operands)
                if beam_signature in beam_registered:
                    continue
                newbeam = len(beam_registered)
                next_beamscores[i, newbeam] = score
                next_result[i, newbeam, :-1] = result[i, prevbeam]
                new_tokens = [operator]
                for j, a in enumerate(operands):
                    if con_range(a):
                        new_tokens += [0, a - con_offset]
                    elif num_range(a):
                        new_tokens += [1, a - num_offset]
                    else:
                        new_tokens += [2, a - mem_offset]
                new_tokens = torch.as_tensor(new_tokens, dtype=torch.long, device=device)
                next_result[i, newbeam, -1, :new_tokens.shape[0]] = new_tokens
                beam_registered.add(beam_signature)

        beamscores = next_beamscores
        last_tokens = next_result[:, :, -1, 0]
        all_exit = ((last_tokens == -1) | (last_tokens == len(op_dict2)-1)).all().item()

        result = next_result
        seq_len += 1
    
    return result[:, 0], beamscores[:, 0]
    # return result, beamscores

def train_double(solver, text_ids, text_pads, num_ids, num_pads, graphs, equ_ids, equ_pads, tuple_ids, op_tokens, constant_tokens):
    encoded = solver.encoder(text_ids, text_pads, num_ids, num_pads)
    # encoded['num'] = solver.grapher(encoded['num'], graphs.clone())
    loss1 = train_tree(solver, encoded, text_ids, text_pads, num_ids, num_pads, equ_ids, equ_pads, op_tokens, constant_tokens)
    # loss2 = train_tuple(solver, encoded, text_ids, text_pads, num_ids, num_pads, tuple_ids)
    loss = loss1
    return loss

def evaluate_double(solver, text_ids, text_pads, num_ids, num_pads, graphs, op_tokens, constant_tokens, op_dict2, max_length, beam_size=3):
    encoded = solver.encoder(text_ids, text_pads, num_ids, num_pads)
    # encoded['num'] = solver.grapher(encoded['num'], graphs.clone())
    tree_res = evaluate_tree(solver, encoded, text_ids, text_pads, num_ids, num_pads, op_tokens, constant_tokens, max_length, beam_size)
    # tuple_res = evaluate_tuple(solver, encoded, text_ids, text_pads, num_ids, num_pads, op_dict2, max_length, beam_size)
    return tree_res