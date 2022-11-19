import re

# Pattern of fraction numbers e.g. 5/3
FRACTIONAL_PATTERN = re.compile('(\\d+/\\d+)')
# Pattern of numbers e.g. 2,930.34
NUMBER_PATTERN = re.compile('([+\\-]?(\\d{1,3}(,\\d{3})+|\\d+)(\\.\\d+)?)')
# Pattern of number and fraction numbers
NUMBER_AND_FRACTION_PATTERN = re.compile('(%s|%s)' % (FRACTIONAL_PATTERN.pattern, NUMBER_PATTERN.pattern))
# Pattern of numbers that following zeros under the decimal point. e.g., 0_250000000
FOLLOWING_ZERO_PATTERN = re.compile('(\\d+|\\d+_[0-9]*[1-9])_?(0+|0{4}\\d+)$')

# Map specifies how english words can be interpreted as a number
NUMBER_READINGS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
    'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000, 'million': 1000000, 'billion': 1000000000,

    'once': 1, 'twice': 2, 'thrice': 3, 'double': 2, 'triple': 3, 'quadruple': 4,
    'doubled': 2, 'tripled': 3, 'quadrupled': 4,

    'third': 3, 'forth': 4, 'fourth': 4, 'fifth': 5,
    'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10, 'eleventh': 11, 'twelfth': 12, 'thirteenth': 13,
    'fourteenth': 14, 'fifteenth': 15, 'sixteenth': 16, 'seventeenth': 17, 'eighteenth': 18, 'nineteenth': 19,
    'twentieth': 20, 'thirtieth': 30, 'fortieth': 40, 'fiftieth': 50, 'sixtieth': 60,
    'seventieth': 70, 'eightieth': 80, 'ninetieth': 90,
    'hundredth': 100, 'thousandth': 1000, 'millionth': 1000000,
    'billionth': 1000000000,

    'dozen': 12, 'half': 0.5, 'quarter': 0.25,
    'halved': 0.5, 'quartered': 0.25,
}

# List of multiples, that can be used as a unit. e.g. three half (3/2)
MULTIPLES = ['once', 'twice', 'thrice', 'double', 'triple', 'quadruple', 'dozen', 'half', 'quarter',
             'doubled', 'tripled', 'quadrupled', 'halved', 'quartered']

# Suffix of plural forms
PLURAL_FORMS = [('ies', 'y'), ('ves', 'f'), ('s', '')]

# Precedence of operators
OPERATOR_PRECEDENCE = {
    '^': 4,
    '*': 3,
    '/': 3,
    '+': 2,
    '-': 2,
    '=': 1
}


def find_numbers_in_text(text, append_number_token=True):
    numbers = []
    new_text = []

    # Replace "[NON-DIGIT][SPACEs].[DIGIT]" with "[NON-DIGIT][SPACEs]0.[DIGIT]"
    text = re.sub("([^\\d.,]+\\s*)(\\.\\d+)", "\\g<1>0\\g<2>", text)
    # Replace space between digits or digit and special characters like ',.' with "⌒" (to preserve original token id)
    text = re.sub("(\\d+)\\s+(\\.\\d+|,\\d{3}|\\d{3})", "\\1⌒\\2", text)

    # Original token index
    i = 0
    prev_token = None
    for token in text.split(' '):
        # Increase token id and record original token indices
        token_index = [i + j for j in range(token.count('⌒') + 1)]
        i = max(token_index) + 1

        # First, find the number patterns in the token
        token = token.replace('⌒', '')
        number_patterns = NUMBER_AND_FRACTION_PATTERN.findall(token)
        if number_patterns:
            for pattern in number_patterns:
                # Matched patterns, listed by order of occurrence.
                surface_form = pattern[0]
                surface_form = surface_form.replace(',', '')

                # Normalize the form: use the decimal point representation with 15-th position under the decimal point.
                is_fraction = '/' in surface_form
                value = eval(surface_form)
                if type(value) is float:
                    surface_form = FOLLOWING_ZERO_PATTERN.sub('\\1', '%.15f' % value)

                numbers.append(surface_form)

            # new_text.append(NUMBER_AND_FRACTION_PATTERN.sub(' \\1 %s ' % 'N_', token))
            # new_text.append('_%s %s' % (token, '[N]'))
            new_text.append('N_' + str(len(numbers)-1))
        else:
            # If there is no numbers in the text, then find the textual numbers.
            # Append the token first.
            # new_text.append(token)

            # Type indicator
            is_ordinal = False
            is_fraction = False
            is_single_multiple = False
            is_combined_multiple = False

            subtokens = re.split('[^a-zA-Z0-9]+', token.lower())  # Split hypen-concatnated tokens like twenty-three
            token_value = None
            for subtoken in subtokens:
                if not subtoken:
                    continue

                # convert to singular nouns
                for plural, singluar in PLURAL_FORMS:
                    if subtoken.endswith(plural):
                        subtoken = subtoken[:-len(plural)] + singluar
                        break

                if subtoken in NUMBER_READINGS:
                    if not token_value:
                        # If this is the first value in the token, then set it as it is.
                        token_value = NUMBER_READINGS[subtoken]

                        is_ordinal = subtoken[-2:] in ['rd', 'th']
                        is_single_multiple = subtoken in MULTIPLES

                        if is_ordinal and prev_token == 'a':
                            # Case like 'A third'
                            token_value = 1 / token_value
                    else:
                        # If a value was set before reading this subtoken,
                        # then treat it as multiples. (e.g. one-million, three-fifths, etc.)
                        followed_value = NUMBER_READINGS[subtoken]
                        is_single_multiple = False
                        is_ordinal = False

                        if followed_value >= 100 or subtoken == 'half':  # case of unit
                            token_value *= followed_value
                            is_combined_multiple = True
                        elif subtoken[-2:] in ['rd', 'th']:  # case of fractions
                            token_value /= followed_value
                            is_fraction = True
                        else:
                            token_value += followed_value

            # If a number is found.
            if token_value is not None:
                if type(token_value) is float:
                    surface_form = FOLLOWING_ZERO_PATTERN.sub('\\1', '%.15f' % token_value)
                else:
                    surface_form = str(token_value)

                numbers.append(surface_form)
                # new_text.append('_%s %s' % (token, '[N]'))
                new_text.append('N_' + str(len(numbers)-1))
            else:
                new_text.append(token)

        prev_token = token

    if append_number_token:
        text = ' '.join(new_text)

    return text, numbers