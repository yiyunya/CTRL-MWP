from math import log10
from sympy import Eq
from itertools import groupby

# Index for padding
PAD_ID = -1

# Key indices for preprocessing and field input
PREP_KEY_EQN = 0
PREP_KEY_ANS = 1
PREP_KEY_MEM = 2

# Token for text field
NUM_TOKEN = '[N]'

# String key names for inputs
IN_TXT = 'text'
IN_TPAD = 'text_pad'
IN_TNUM = 'text_num'
IN_TNPAD = 'text_numpad'
IN_EQN = 'equation'
IN_PRO = 'problem'

# Dictionary of operators
OPERATORS = {
    '+': {'arity': 2, 'commutable': True, 'top_level': False, 'convert': (lambda *x: x[0] + x[1])},
    '-': {'arity': 2, 'commutable': False, 'top_level': False, 'convert': (lambda *x: x[0] - x[1])},
    '*': {'arity': 2, 'commutable': True, 'top_level': False, 'convert': (lambda *x: x[0] * x[1])},
    '/': {'arity': 2, 'commutable': False, 'top_level': False, 'convert': (lambda *x: x[0] / x[1])},
    '^': {'arity': 2, 'commutable': False, 'top_level': False, 'convert': (lambda *x: x[0] ** x[1])},
    '=': {'arity': 2, 'commutable': True, 'top_level': True,
          'convert': (lambda *x: Eq(x[0], x[1], evaluate=False))}
}

# Arity and top-level classes
TOP_LEVEL_CLASSES = ['Eq']
ARITY_MAP = {key: [item[-1] for item in lst]
             for key, lst in groupby(sorted([((op['arity'], op['top_level']), key) for key, op in OPERATORS.items()],
                                            key=lambda t: t[0]), key=lambda t: t[0])}

# Infinity values
NEG_INF = float('-inf')
POS_INF = float('inf')

# FOR EXPRESSION INPUT
# Token for operator field
FUN_NEW_EQN = '__NEW_EQN'
FUN_END_EQN = '__DONE'
FUN_NEW_VAR = '__NEW_VAR'
FUN_TOKENS = [FUN_NEW_EQN, FUN_END_EQN, FUN_NEW_VAR]
FUN_NEW_EQN_ID = FUN_TOKENS.index(FUN_NEW_EQN)
FUN_END_EQN_ID = FUN_TOKENS.index(FUN_END_EQN)
FUN_NEW_VAR_ID = FUN_TOKENS.index(FUN_NEW_VAR)

FUN_TOKENS_WITH_EQ = FUN_TOKENS + ['=']
FUN_EQ_SGN_ID = FUN_TOKENS_WITH_EQ.index('=')

# Token for operand field
ARG_CON = 'CONST:'
ARG_NUM = 'NUMBER:'
ARG_MEM = 'MEMORY:'
ARG_TOKENS = [ARG_CON, ARG_NUM, ARG_MEM]
ARG_CON_ID = ARG_TOKENS.index(ARG_CON)
ARG_NUM_ID = ARG_TOKENS.index(ARG_NUM)
ARG_MEM_ID = ARG_TOKENS.index(ARG_MEM)
ARG_UNK = 'UNK'
ARG_UNK_ID = 0

# Maximum capacity of variable, numbers and expression memories
VAR_MAX = 2
NUM_MAX = 32
MEM_MAX = 32

# FOR OP INPUT
SEQ_NEW_EQN = FUN_NEW_EQN
SEQ_END_EQN = FUN_END_EQN
SEQ_UNK_TOK = ARG_UNK
SEQ_TOKENS = [SEQ_NEW_EQN, SEQ_END_EQN, SEQ_UNK_TOK, '=']
SEQ_PTR_NUM = '__NUM'
SEQ_PTR_VAR = '__VAR'
SEQ_PTR_TOKENS = SEQ_TOKENS + [SEQ_PTR_NUM, SEQ_PTR_VAR]
SEQ_NEW_EQN_ID = SEQ_PTR_TOKENS.index(SEQ_NEW_EQN)
SEQ_END_EQN_ID = SEQ_PTR_TOKENS.index(SEQ_END_EQN)
SEQ_UNK_TOK_ID = SEQ_PTR_TOKENS.index(SEQ_UNK_TOK)
SEQ_EQ_SGN_ID = SEQ_PTR_TOKENS.index('=')
SEQ_PTR_NUM_ID = SEQ_PTR_TOKENS.index(SEQ_PTR_NUM)
SEQ_PTR_VAR_ID = SEQ_PTR_TOKENS.index(SEQ_PTR_VAR)
SEQ_GEN_NUM_ID = SEQ_PTR_NUM_ID
SEQ_GEN_VAR_ID = SEQ_GEN_NUM_ID + NUM_MAX

# Format of variable/number/expression tokens
FORMAT_VAR = 'X_%%0%dd' % (int(log10(VAR_MAX)) + 1)
FORMAT_NUM = 'N_%%0%dd' % (int(log10(NUM_MAX)) + 1)
FORMAT_MEM = 'M_%%0%dd' % (int(log10(MEM_MAX)) + 1)
VAR_PREFIX = 'X_'
NUM_PREFIX = 'N_'
CON_PREFIX = 'C_'
MEM_PREFIX = 'M_'

# Key for field names
FIELD_OP_GEN = 'op_gen'
FIELD_EXPR_GEN = 'expr_gen'
FIELD_EXPR_PTR = 'expr_ptr'

# Model names
MODEL_VANILLA_TRANS = 'vanilla'  # Vanilla Op Transformer
MODEL_EXPR_TRANS = 'expr'  # Vanilla Transformer + Expression (Expression Transformer)
MODEL_EXPR_PTR_TRANS = 'ept'  # Expression-Pointer Transformer


def postfix_parser(equation, memory: list) -> int:
    stack = []

    for tok in equation:
        if tok in OPERATORS:
            # If token is an operator, form expression and push it into the memory and stack.
            op = OPERATORS[tok]
            arity = op['arity']

            # Retrieve arguments
            args = stack[-arity:]
            stack = stack[:-arity]

            # Store the result with index where the expression stored
            stack.append((ARG_MEM, len(memory)))
            # Store the expression into the memory.
            memory.append((tok, args))
        else:
            # Push an operand before meet an operator
            stack.append(tok)

    return len(stack)

variable_prefixes = ['X_']
number_perfixes = ['N_']
constant_prefix = 'C_'
force_generation = False
max_arity=2
variables = []
memories = []
import re
formulae = [['N_0', 'N_1', '+', 'C_3', '*'], ['N_0', 'N_1', '+', 'C_3', '*']]
variables = []
memories = []

for expr in formulae:
    # Replace number, const, variable tokens with N_<id>, C_<value>, X_<id>
    normalized = []
    for token in expr:
        if any(token.startswith(prefix) for prefix in variable_prefixes):
            # Case 1: Variable
            if token not in variables:
                variables.append(token)

            # Set as negative numbers, since we don't know how many variables are in the list.
            normalized.append((ARG_MEM, - variables.index(token) - 1))
        elif any(token.startswith(prefix) for prefix in number_perfixes):
            # Case 2: Number
            token = int(token.split('_')[-1])
            if force_generation:
                # Treat number indicator as constant.
                normalized.append((ARG_NUM, FORMAT_NUM % token))
            else:
                normalized.append((ARG_NUM, token))
        elif token.startswith(constant_prefix):
            normalized.append((ARG_CON, token.replace(constant_prefix, CON_PREFIX)))
        else:
            normalized.append(token)

    # Build expressions (ignore answer tuples)
    stack_len = postfix_parser(normalized, memories)
    assert stack_len == 1, "Equation is not correct! '%s'" % expr

# Reconstruct indices for result of prior expression.
var_length = len(variables)
# Add __NEW_VAR at the front of the sequence. The number of __NEW_VAR()s equals to the number of variables used.
preprocessed = [(FUN_NEW_VAR, []) for _ in range(var_length)]
for operator, operands in memories:
    # For each expression
    new_arguments = []
    for typ, tok in operands:
        if typ == ARG_MEM:
            # Shift index of prior expression by the number of variables.
            tok = tok + var_length if tok >= 0 else -(tok + 1)

            if force_generation:
                # Build as a string
                tok = FORMAT_MEM % tok

        new_arguments.append((typ, tok))

    # Register an expression
    preprocessed.append((operator, new_arguments))



def process(self, batch: List[List[Tuple[str, list]]], device=None, **kwargs):
    return self.numericalize(self.pad(batch), device=device)

def pad(self, minibatch: List[List[Tuple[str, list]]]) -> List[List[Tuple[str, list]]]:
    # Compute maximum length with __NEW_EQN and __END_EQN.
    max_len = max(len(item) for item in minibatch) + 2  # 2 = BOE/EOE
    padded_batch = []

    # Padding for no-operand functions (i.e. special commands)
    max_arity_pad = [(None, None)] * self.max_arity

    for item in minibatch:
        padded_item = [(FUN_NEW_EQN, max_arity_pad)]

        for operator, operands in item:
            # We also had to pad operands.
            remain_arity = max(0, self.max_arity - len(operands))
            operands = operands + max_arity_pad[:remain_arity]

            padded_item.append((operator, operands))

        padded_item.append((FUN_END_EQN, max_arity_pad))
        padded_item += [(None, max_arity_pad)] * max(0, max_len - len(padded_item))

        # Add batched item
        padded_batch.append(padded_item)

    return padded_batch

def convert_token_to_id(self, expression):
    # Destructure the tuple.
    operator, operand = expression

    # Convert operator into index.
    operator = PAD_ID if operator is None else op_dict[operator]
    # Convert operands
    new_operands = []
    for src, a in operand:
        # For each operand, we attach [Src, Value] after the end of new_args.
        if src is None:
            new_operands += [PAD_ID, PAD_ID]
        else:
            # Get the source
            new_operands.append(ARG_TOKENS.index(src))
            # Get the index of value
            if src == ARG_CON or self.force_generation:
                # If we need to look up the vocabulary, then find the index in it.
                new_operands.append(constant_dict[a])
            else:
                # Otherwise, use the index information that is already specified in the operand.
                new_operands.append(a)

    # Return the flattened list of operator and operands.
    return [operator] + new_operands

def numericalize(self, minibatch: List[List[Tuple[str, list]]], device=None) -> torch.Tensor:
    """
    Make a Tensor from the given minibatch.

    :param List[List[Tuple[str, list]]] minibatch: Padded minibatch to form a tensor
    :param torch.device device: Device to store the result
    :rtype: torch.Tensor
    :return: A Long Tensor of given minibatch. Shape [B, T, 1+2A],
        where B = batch size, T = length of op-token sequence, and A = maximum arity.
    """
    minibatch = [[self.convert_token_to_id(token) for token in item] for item in minibatch]
    return torch.as_tensor(minibatch, dtype=torch.long, device=device)



def shift_target(target: torch.Tensor, fill_value=-1):
    # Target does not require gradients.
    with torch.no_grad():
        pad_at_end = torch.full((target.shape[0], 1), fill_value=fill_value, dtype=target.dtype, device=target.device)
        return torch.cat([target[:, 1:], pad_at_end], dim=-1).contiguous()

def loss_and_accuracy(predicted: torch.Tensor, target: torch.Tensor, prefix: str,
                      loss_factor: float = 1.0):
    PAD_ID = -1
    tdim = target.dim()
    pdim = predicted.dim()
    tdtype = target.dtype

    result = {}

    if tdtype == torch.long and tdim + 1 == pdim:
        target_focus = target != PAD_ID
        greedy_choice_correct = predicted.argmax(dim=-1) == target
        predicted = predicted.flatten(0, -2)
        target = target.flatten()
        # loss_fct = SmoothedCrossEntropyLoss(ignore_index=PAD_ID)
        loss_fct = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    else:
        raise NotImplementedError('There are no such rules for computing loss between %s-dim predicted %s tensor '
                                  'and %s-dim target %s tensor' % (pdim, predicted.dtype, tdim, tdtype))

    loss = loss_fct(predicted, target)

    if loss_factor != 1.0:
        loss = loss * loss_factor

    if not torch.isfinite(loss).all().item():
        print('NAN')

    result['loss'] = loss

    return {prefix + '/' + key: value for key, value in result.items()}

def mask_forward(sz, diagonal=1):
    return torch.ones(sz, sz, dtype=torch.bool, requires_grad=False).triu(diagonal=diagonal).contiguous()

def masked_cross_entropy(logits, target, mask):
    target[~mask] = 0
    logits_flat = logits.reshape(-1, logits.size(-1))
    target_flat = target.reshape(-1, 1)
    losses_flat = torch.gather(logits_flat, index=target_flat, dim=1)
    losses = losses_flat.reshape(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / logits.size(0)
    return loss

class BaseDecoder(nn.Module):
    def __init__(self, config, operator_size, constant_size):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.embedding_dim = config.hidden_size
        self.num_hidden_layers = 6
        self.layernorm_eps = config.layer_norm_eps
        self.num_heads = config.num_attention_heads
        self.num_pointer_heads = 1
        self.operator_size = operator_size
        self.constant_size = constant_size

        """ Embedding layers """
        self.operator_word_embedding = nn.Embedding(self.operator_size, self.hidden_dim)
        self.operator_pos_embedding = PositionalEncoding(self.hidden_dim)
        self.dropout_operator = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_operand = nn.Dropout(config.hidden_dropout_prob)
        self.operand_source_embedding = nn.Embedding(3, self.hidden_dim)

        """ Scalar parameters """
        degrade_factor = self.embedding_dim ** 0.5
        self.operator_pos_factor = nn.Parameter(torch.tensor(degrade_factor), requires_grad=True)
        self.operand_source_factor = nn.Parameter(torch.tensor(degrade_factor), requires_grad=True)

        """ Layer Normalizations """
        self.operator_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)
        self.operand_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)

        """ Linear Transformation """
        self.embed_to_hidden = nn.Linear(self.hidden_dim * 3, self.hidden_dim)

        """ Transformer layer """
        self.shared_decoder_layer = TransformerLayer(config, self.num_heads)
        # self.attn = MultiheadAttention(config)

        """ Output layer """
        self.operator_out = nn.Linear(self.hidden_dim, self.operator_size)
        # # Softmax layers, which can handle infinity values properly (used in Equation 6, 10)
        self.softmax = LogSoftmax(dim=-1)

    def _build_operand_embed(self, ids, mem_pos, nums):
        raise NotImplementedError()

    def _build_decoder_input(self, ids, nums):
        operator = get_embedding_without_pad(self.operator_word_embedding, ids.select(dim=-1, index=0))
        operator_pos = self.operator_pos_embedding(ids.shape[1])
        operator = self.operator_norm(operator * self.operator_pos_factor + operator_pos.unsqueeze(0)).unsqueeze(2)
        # operator = operator.unsqueeze(2)
        # operator = self.dropout_operator(operator)

        operand = get_embedding_without_pad(self.operand_source_embedding, ids[:, :, 1::2]) * self.operand_source_factor
        operand += self._build_operand_embed(ids, operator_pos, nums)
        operand = self.operand_norm(operand)
        # operand = self._build_operand_embed(ids, operator_pos, nums)
        # operand = self.dropout_operator(operand)

        operator_operands = torch.cat([operator, operand], dim=2).contiguous().flatten(start_dim=2)
        return self.embed_to_hidden(operator_operands)

    def _build_decoder_context(self, embedding, embedding_pad=None, text=None, text_pad=None):
        mask = mask_forward(embedding.shape[1]).to(embedding.device)
        output = embedding
        for _ in range(self.num_hidden_layers):
            output = self.shared_decoder_layer(target=output, memory=text, target_attention_mask=mask,
                                               target_ignorance_mask=embedding_pad, memory_ignorance_mask=text_pad)
        return output

    def _forward_single(self, text=None, text_pad=None, text_num=None, equation=None):
        PAD_ID = -1
        FUN_EQ_SGN_ID = 3
        FUN_NEW_EQN_ID = 0
        FUN_END_EQN_ID = 1
        NEG_INF = -float('inf')
        operator_ids = equation.select(dim=2, index=0)
        output = self._build_decoder_input(ids=equation, nums=text_num)
        output_pad = operator_ids == PAD_ID

        output_not_usable = output_pad.clone()
        output_not_usable[:, :-1].masked_fill_(operator_ids[:, 1:] == FUN_EQ_SGN_ID, True)
        output = self._build_decoder_context(embedding=output, embedding_pad=output_pad, text=text, text_pad=text_pad)
        operator_out = self.operator_out(output)

        if not self.training:
            operator_out[:, :, FUN_NEW_EQN_ID] = NEG_INF
            operator_out[:, :, FUN_END_EQN_ID].masked_fill_(operator_ids != FUN_EQ_SGN_ID, NEG_INF)

        result = {'operator': self.softmax(operator_out),
                  '_out': output, '_not_usable': output_not_usable}

        return result

class TupleDecoder(BaseDecoder):
    def __init__(self, config, operator_size, constant_size):
        super().__init__(config, operator_size, constant_size)

        """ Operand embedding """
        self.constant_word_embedding = nn.Embedding(self.constant_size, self.hidden_dim)

        """ Output layer """
        self.operand_out = nn.ModuleList([
            nn.ModuleDict({
                '0_attn': MultiheadAttentionWeights(config, self.num_pointer_heads),
                '1_mean': Squeeze(dim=-1) if self.num_pointer_heads == 1 else AveragePooling(dim=-1)
            }) for _ in range(2)
        ])

    def _build_operand_embed(self, ids, mem_pos, nums):
        PAD_ID = -1
        ARG_CON_ID, ARG_NUM_ID, ARG_MEM_ID = 0, 1, 2
        operand_source = ids[:, :, 1::2]
        operand_value = ids[:, :, 2::2]

        number_operand = operand_value.masked_fill(operand_source != ARG_NUM_ID, PAD_ID)
        operand = torch.stack([get_embedding_without_pad(nums[b], number_operand[b])
                               for b in range(ids.shape[0])], dim=0).contiguous()

        operand += get_embedding_without_pad(self.constant_word_embedding,
                                             operand_value.masked_fill(operand_source != ARG_CON_ID, PAD_ID))

        prior_result_operand = operand_value.masked_fill(operand_source != ARG_MEM_ID, PAD_ID)
        operand += get_embedding_without_pad(mem_pos, prior_result_operand)
        return operand

    def _build_attention_keys(self, num, mem, num_pad=None, mem_pad=None):
        batch_sz = num.shape[0]
        const_sz = self.constant_size
        const_num_sz = const_sz + num.shape[1]
        const_key = self.constant_word_embedding.weight.unsqueeze(0).expand(batch_sz, const_sz, self.hidden_dim)
        key = torch.cat([const_key, num, mem], dim=1).contiguous()
        key_ignorance_mask = torch.zeros(key.shape[:2], dtype=torch.bool, device=key.device)
        if num_pad is not None:
            key_ignorance_mask[:, const_sz:const_num_sz] = num_pad
        if mem_pad is not None:
            key_ignorance_mask[:, const_num_sz:] = mem_pad

        attention_mask = torch.zeros(mem.shape[1], key.shape[1], dtype=torch.bool, device=key.device)
        attention_mask[:, const_num_sz:] = mask_forward(mem.shape[1], diagonal=0).to(key_ignorance_mask.device)

        return key, key_ignorance_mask, attention_mask

    def _forward_single(self, text=None, text_pad=None, text_num=None, text_numpad=None, equation=None):
        result = super()._forward_single(text, text_pad, text_num, equation)
        output = result.pop('_out')
        output_not_usable = result.pop('_not_usable')
        key, key_ign_msk, attn_msk = self._build_attention_keys(num=text_num, mem=output,
                                                                num_pad=text_numpad, mem_pad=output_not_usable)
        for j, layer in enumerate(self.operand_out):
            score = apply_module_dict(layer, encoded=output, key=key, key_ignorance_mask=key_ign_msk,
                                      attention_mask=attn_msk)
            result['operand_%s' % j] = self.softmax(score)

        return result

    def _build_target_dict(self, text_num=None, equation=None):
        ARG_CON_ID, ARG_NUM_ID, ARG_MEM_ID = 0, 1, 2
        num_offset = self.constant_size
        mem_offset = num_offset + text_num.shape[1]
        targets = {'operator': equation.select(dim=-1, index=0)}
        for i in range(2):
            operand_source = equation[:, :, (i * 2 + 1)]
            operand_value = equation[:, :, (i * 2 + 2)].clamp_min(0)
            operand_value += operand_source.masked_fill(operand_source == ARG_NUM_ID, num_offset) \
                .masked_fill_(operand_source == ARG_MEM_ID, mem_offset)
            targets['operand_%s' % i] = operand_value

        return targets
    
    def forward(self, text, text_pad, text_num, text_numpad, equation=None):
        # if self.training:
        #     outputs = self._forward_single(text, text_pad, text_num, text_numpad, equation)
        #     with torch.no_grad():
        #         targets = self._build_target_dict(text_num, equation)
        #         return outputs, targets
        # else:
        #     with torch.no_grad():
        #         tensor = {}
        #         tensor['text'] = text
        #         tensor['text_pad'] = text_pad
        #         tensor['text_num'] = text_num
        #         tensor['text_numpad'] = text_numpad
        #         tensor['equation'] = equation
        #         outputs = apply_across_dim(self._forward_single, dim=1, shared_keys={'text', 'text_pad', 'text_num', 'text_numpad'}, **tensor)
        # return outputs

        result = {}
        if self.training:
            outputs = self._forward_single(text, text_pad, text_num, text_numpad, equation)
            with torch.no_grad():
                targets = self._build_target_dict(text_num, equation)

            for key in targets:
                result.update(loss_and_accuracy(outputs[key], shift_target(targets[key]), prefix='Train_%s' % key))
        else:
            with torch.no_grad():
                tensor = {}
                tensor['text'] = text
                tensor['text_pad'] = text_pad
                tensor['text_num'] = text_num
                tensor['text_numpad'] = text_numpad
                tensor['equation'] = equation
                result.update(apply_across_dim(self._forward_single, dim=1,
                                               shared_keys={'text', 'text_pad', 'text_num', 'text_numpad'}, **tensor))

        return result