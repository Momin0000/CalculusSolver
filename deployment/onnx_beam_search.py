"""
numpy/onnxruntime-only mirror of inference/beam_search.py -- mirrors that
file's beam_search() line-for-line in logic, but never imports torch.
This is the entire point of the Option A (ONNX) deployment path: the
production Vercel bundle only needs onnxruntime + numpy, not the full
PyTorch package, which is what pushed the old bundle over the ~250MB
serverless size limit.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort

from inference.grammar import NodeValidityPool


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def onnx_beam_search(
    session: ort.InferenceSession,
    src_tokens: List[int],
    vocab_map: Dict[str, Any],
    beam_size: int = 5,
    max_len: int = 32,
    node_pool: Optional[NodeValidityPool] = None,
) -> Dict[str, Any]:
    """Mirrors inference/beam_search.py::beam_search(), but calls the
    exported ONNX graph via onnxruntime instead of a torch.nn.Module."""
    vocab = vocab_map["token_to_id"]
    id_to_token = vocab_map["id_to_token"]
    bos_id = vocab["[BOS]"]
    eos_id = vocab["[EOS]"]

    if node_pool is None:
        node_pool = NodeValidityPool()

    vocab_size = max(id_to_token.keys()) + 1
    all_candidate_tokens = [id_to_token.get(idx, "[PAD]") for idx in range(vocab_size)]

    src_arr = np.array([src_tokens], dtype=np.int64)

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

            tgt_arr = np.array([current_tokens], dtype=np.int64)
            logits = session.run(
                ["logits"],
                {"src_seq": src_arr, "tgt_in_seq": tgt_arr},
            )[0]
            next_logits = logits[0, -1, :]

            mask = node_pool.mask(validity_tokens, all_candidate_tokens)
            safe_logits = next_logits.copy()
            safe_logits[[not v for v in mask]] = -np.inf

            if np.all(np.isinf(safe_logits)):
                continue

            log_probs = np.log(_softmax(safe_logits) + 1e-12)
            k = min(beam_size, safe_logits.shape[0])
            top_idx = np.argpartition(-log_probs, k - 1)[:k]
            top_idx = top_idx[np.argsort(-log_probs[top_idx])]

            for token_id in top_idx:
                token_id = int(token_id)
                score = float(log_probs[token_id])
                new_tokens = current_tokens + [token_id]
                finished = token_id == eos_id
                candidates.append({
                    "tokens": new_tokens,
                    "score": beam["score"] + score,
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