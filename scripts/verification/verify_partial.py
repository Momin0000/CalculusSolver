import sys
sys.path.insert(0, '.')
from problem_generator import generate_multivar_diff
from tokenizer.slang_serializer import serialize_slang_math
from inference.fallback_solver import FallbackSolver
import json

vocab = json.load(open('tokenizer/vocab.json'))
token_to_id = {}
for k, val in vocab.items():
    if not k.startswith('_') and isinstance(val, dict):
        token_to_id.update(val)

solver = FallbackSolver()
ok, fail = 0, 0
for i in range(200):
    src_terms, ans_terms, var, rule_id = generate_multivar_diff()
    src_op = {'op': 'partial', 'var': var, 'expr': src_terms[0]}
    tokens = serialize_slang_math(src_op)
    missing = [t for t in tokens if t not in token_to_id]
    if missing:
        print('MISSING:', missing)
        fail += 1
        continue
    try:
        solver.solve(src_op)
        ok += 1
    except Exception as e:
        print('SOLVER FAILED:', e)
        fail += 1

print(f'{ok}/{ok+fail} passed')