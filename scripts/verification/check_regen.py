import json

diff_but_partial_rule = 0
proper_partial = 0
total = 0
with open('data/splits/train.jsonl') as f:
    for line in f:
        total += 1
        row = json.loads(line)
        src = row.get('src_tokens', {})
        op = src.get('op')
        rule = row.get('rule_ids')
        if op == 'diff' and rule == 7:
            diff_but_partial_rule += 1
        if op == 'partial':
            proper_partial += 1

print('total rows:', total)
print('stale rows (op=diff, rule=7):', diff_but_partial_rule)
print('correct op=partial rows:', proper_partial)