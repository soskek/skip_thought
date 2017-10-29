import sys
import io

# use spacy 2.0

import spacy
nlp = spacy.load('en_core_web_sm', device=2)
text = []
for i_line, l in enumerate(io.open(sys.argv[1], encoding='utf8')):
    if i_line % 100000 == 0:
        sys.stderr.write('{} lines end\n'.format(i_line))
    l = l.strip()
    #"""
    if not l:
        print('')
        continue
    doc = nlp(l)
    tokens = [tok.orth_ for tok in doc]
    for ent in doc.ents:
        tokens[ent.start] = '{}@@{}'.format(
            '__'.join(tokens[ent.start: ent.end]),
            ent.label_)
        for i in range(ent.start + 1, ent.end):
            tokens[i] = None

    for sent in doc.sents:
        print(' '.join(tok for tok in tokens[sent.start: sent.end]
                       if tok is not None))
    """
    if l:
        text.append(l)
        continue
    doc = nlp('\n'.join(text))
    tokens = [tok.orth_ for tok in doc]
    for ent in doc.ents:
        tokens[ent.start] = '{}@@{}'.format(
            '__'.join(tokens[ent.start: ent.end]),
            ent.label_)
        for i in range(ent.start + 1, ent.end):
            tokens[i] = None

    for sent in doc.sents:
        print(' '.join(tok for tok in tokens[sent.start: sent.end]
                       if tok is not None))
    print('')
    text = []
    """
