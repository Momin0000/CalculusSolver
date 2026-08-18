import json
import os
from typing import Any, Dict, List, Optional

import torch

from inference.grammar import NodeValidityPool, flatten_vocab, load_vocab, is_valid_prefix

# NOTE: NodeValidityPool, flatten_vocab, load_vocab, and is_valid_prefix now
# live in inference/grammar.py (torch-free) so the ONNX deployment path
# (deployment/onnx_beam_search.py) can reuse them without importing torch.
# This is a pure move -- no logic changed from the previous inline versions.


def beam_search(
    model,
    src_tokens: torch.Tensor,
    vocab_map: Dict[str, Any],
    beam_size: int = 5,
    max_len: int = 32,
    node_pool: Optional[NodeValidityPool] = None,
) -> Dict[str, Any]:
    """Beam search for CalculusSolverModel (tree-based, model/transformer.py).

    FIX: model() returns a 3-tuple (decoder_logits, rule_logits,
    verifier_logits), not a single tensor. Only decoder_logits is used for
    next-token selection here. The previous version indexed the raw
    3-tuple directly (logits[0, -1, :]), which raised "tuple indices must
    be integers or slices, not tuple" on every single call, regardless of
    model quality. Fixed by unpacking model_output[0] before indexing.
    """
    device = src_tokens.device
    vocab = vocab_map["token_to_id"]
    id_to_token = vocab_map["id_to_token"]
    bos_id = vocab["[BOS]"]
    eos_id = vocab["[EOS]"]

    if node_pool is None:
        node_pool = NodeValidityPool()

    vocab_size = max(id_to_token.keys()) + 1
    all_candidate_tokens = [id_to_token.get(idx, "[PAD]") for idx in range(vocab_size)]

    beams = [{"tokens": [bos_id], "score": 0.0, "finished": False}]
    completed = []

    for _ in range(max_len):
        candidates = []
        for beam in beams:
            if beam["finished"]:
                candidates.append(beam)
                continue

            current_tokens = beam["tokens"]
            token_strings = [id_to_token[t] for t in current_tokens]
            validity_tokens = (
                token_strings[1:]
                if token_strings and token_strings[0] == "[BOS]"
                else token_strings
            )

            tgt = torch.tensor([current_tokens], device=device)

            # FIX: unpack the tuple safely -- works whether model() returns
            # a single tensor or a (decoder_logits, rule_logits,
            # verifier_logits) tuple, so future model interface changes
            # won't silently reintroduce this same crash.
            model_output = model(src_tokens, tgt)
            decoder_logits = model_output[0] if isinstance(model_output, tuple) else model_output
            next_logits = decoder_logits[0, -1, :]

            mask = node_pool.mask(validity_tokens, all_candidate_tokens)
            invalid_mask = torch.tensor([not v for v in mask], device=device)
            safe_logits = next_logits.masked_fill(invalid_mask, float("-inf"))

            if torch.isinf(safe_logits).all():
                continue

            log_probs = torch.log_softmax(safe_logits, dim=-1)
            topk = torch.topk(log_probs, min(beam_size, safe_logits.size(0)))
            for score, token_id in zip(topk.values.tolist(), topk.indices.tolist()):
                new_tokens = current_tokens + [int(token_id)]
                finished = token_id == eos_id
                candidates.append({
                    "tokens": new_tokens,
                    "score": beam["score"] + float(score),
                    "finished": finished,
                })

        if not candidates:
            break

        beams = sorted(candidates, key=lambda x: x["score"], reverse=True)[:beam_size]
        if all(b["finished"] for b in beams):
            completed.extend(beams)
            break

    best = sorted(completed, key=lambda x: x["score"], reverse=True)[0] if completed else (
        beams[0] if beams else {"tokens": [bos_id], "score": 0.0, "finished": False}
    )

    status = "solved" if best["finished"] else "partial"
    return {"tokens": best["tokens"], "score": best["score"], "status": status}