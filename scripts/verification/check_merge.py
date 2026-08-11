import json

vocab = json.load(open('tokenizer/vocab.json'))
print('_version:', vocab.get('_version'))

flat = {}
dupes = []
for k, val in vocab.items():
    if k.startswith('_'):
        continue
    if isinstance(val, dict):
        for tok, tid in val.items():
            if tid in flat.values():
                existing = [t for t, i in flat.items() if i == tid]
                dupes.append((tok, tid, existing))
            flat[tok] = tid

print('OP:partial present:', 'OP:partial' in flat, '-> ID', flat.get('OP:partial'))
print('Total tokens:', len(flat))
print('Max ID:', max(flat.values()))
if dupes:
    print('DUPLICATE IDs FOUND (DO NOT COMMIT):')
    for tok, tid, existing in dupes:
        print(f'  {tok}={tid} collides with {existing}')
else:
    print('No duplicate IDs -- merge looks clean')