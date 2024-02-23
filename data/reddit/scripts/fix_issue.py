import sys
import re


def clean_text(text):
    text = re.sub(r'(. .)$', r'.', text)
    return text


path = sys.argv[1]

with open(path, 'r') as f:
    with open(path + '.cleaned', 'w') as g:
        for l in f:
            components = l.strip().split('\t')
            if len(components) != 3:
                continue
            cleaned = '\t'.join(
                [components[0]] + [clean_text(s.strip()) for s in components[1:]]
            )
            g.write(cleaned + '\n')
