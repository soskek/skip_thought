from __future__ import print_function
import io
import sys

for i_line, l in enumerate(io.open(sys.argv[1], encoding='utf8')):
    if i_line % 100000 == 0:
        sys.stderr.write('{} lines end\n'.format(i_line))
    l = l.strip().lower()
    if not l:
        print()
        continue

    out = []
    for tok in l.split():
        if '@@@' in tok:
            tok = tok.split('@@@')[0] + '@'
        if '@@' in tok:
            sp = tok.split('@@')
            if sp[-1]:
                tok = sp[-1]
                if len(sp[0]):
                    raw_tail_char = sp[0][-1]
                    tok = raw_tail_char + '@@' + tok
        out.append(tok)
    out = ' '.join(out)

    if not any(c.isdigit() or c.isalpha() for c in out):
        print()
        continue

    print(out)
