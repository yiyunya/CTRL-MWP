import copy
import torch
import torch.nn as nn
from .tree import TreeNode, TreeEmbedding, TreeBeam, copy_list

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


def train_double(solver, text_ids, text_pads, num_ids, num_pads, graphs, equ_ids, equ_pads, tuple_ids, op_tokens, constant_tokens):
    encoded = solver.encoder(text_ids, text_pads, num_ids, num_pads)
    loss1 = train_tree(solver, encoded, text_ids, text_pads, num_ids, num_pads, equ_ids, equ_pads, op_tokens, constant_tokens)
    loss = loss1
    return loss

def evaluate_double(solver, text_ids, text_pads, num_ids, num_pads, graphs, op_tokens, constant_tokens, op_dict2, max_length, beam_size=3):
    encoded = solver.encoder(text_ids, text_pads, num_ids, num_pads)
    tree_res = evaluate_tree(solver, encoded, text_ids, text_pads, num_ids, num_pads, op_tokens, constant_tokens, max_length, beam_size)
    return tree_res