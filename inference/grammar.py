"""
Torch-free SLaNg grammar and vocab helpers, split out of inference/beam_search.py.

Purpose: the ONNX deployment path (deployment/onnx_beam_search.py,
deployment/onnx_solve.py) must never import torch -- that's the entire
point of Option A (ONNX export) as a fix for Vercel's ~250MB serverless
size limit. Everything in this file is pure Python / stdlib only, so both
the PyTorch path (inference/beam_search.py) and the ONNX path
(deployment/onnx_beam_search.py) can import it without pulling torch in.
"""

import json
from typing import Any, Dict, List


def is_valid_prefix(tokens: List[str]) -> bool:
    """Check if the given tokens form a valid prefix of a SLaNg AST.
    First token, if present, may be a RULE:xxx token (SimpleCalculusModel
    prepends one) -- skip it before running the AST grammar check."""
    if not tokens:
        return True

    check_tokens = tokens
    if tokens[0].startswith("RULE:"):
        check_tokens = tokens[1:]
        if not check_tokens:
            return True

    tokens = check_tokens

    def parse_term(index: int) -> dict:
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] != "NODE:TERM":
            return {"status": "invalid"}
        index += 1
        if index >= len(tokens):
            return {"status": "incomplete"}
        if not tokens[index].startswith("COEF:"):
            return {"status": "invalid"}
        index += 1
        while index < len(tokens):
            token = tokens[index]
            if token.startswith("VAR:"):
                index += 1
                if index >= len(tokens):
                    return {"status": "incomplete"}
                if not tokens[index].startswith("EXP:"):
                    return {"status": "invalid"}
                index += 1
                continue
            break
        return {"status": "complete", "next": index}

    def parse_term_list(index: int) -> dict:
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] == "STRUCT:CLOSE":
            return {"status": "complete", "next": index}
        current = index
        while True:
            node = parse_node(current)
            if node["status"] == "invalid":
                return {"status": "invalid"}
            if node["status"] == "incomplete":
                return {"status": "incomplete"}
            current = node["next"]
            if current >= len(tokens):
                return {"status": "incomplete"}
            if tokens[current] == "STRUCT:SEP":
                current += 1
                continue
            if tokens[current] == "STRUCT:CLOSE":
                return {"status": "complete", "next": current}
            return {"status": "invalid"}

    def parse_fraction(index: int) -> dict:
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] != "NODE:FRAC":
            return {"status": "invalid"}
        index += 1
        for expected in ["STRUCT:OPEN", "STRUCT:NUMI", "STRUCT:OPEN"]:
            if index >= len(tokens):
                return {"status": "incomplete"}
            if tokens[index] != expected:
                return {"status": "invalid"}
            index += 1
        numerator = parse_term_list(index)
        if numerator["status"] != "complete":
            return numerator
        index = numerator["next"]
        for expected in ["STRUCT:CLOSE", "STRUCT:SEP", "STRUCT:DENO", "STRUCT:OPEN"]:
            if index >= len(tokens):
                return {"status": "incomplete"}
            if tokens[index] != expected:
                return {"status": "invalid"}
            index += 1
        denominator = parse_term_list(index)
        if denominator["status"] != "complete":
            return denominator
        index = denominator["next"]
        for expected in ["STRUCT:CLOSE", "STRUCT:CLOSE"]:
            if index >= len(tokens):
                return {"status": "incomplete"}
            if tokens[index] != expected:
                return {"status": "invalid"}
            index += 1
        return {"status": "complete", "next": index}

    def parse_op_node(index: int) -> dict:
        if index >= len(tokens):
            return {"status": "incomplete"}
        token = tokens[index]
        if not isinstance(token, str) or not token.startswith("OP:"):
            return {"status": "invalid"}
        index += 1
        while (
            index < len(tokens)
            and isinstance(tokens[index], str)
            and tokens[index].startswith("OPVAR:")
        ):
            index += 1
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] != "STRUCT:OPEN":
            return {"status": "invalid"}
        index += 1
        seen_child = False
        while True:
            node = parse_node(index)
            if node["status"] == "invalid":
                return {"status": "invalid"}
            if node["status"] == "incomplete":
                return {"status": "incomplete"}
            seen_child = True
            index = node["next"]
            if index >= len(tokens):
                return {"status": "incomplete"}
            if tokens[index] == "STRUCT:SEP":
                index += 1
                continue
            if tokens[index] == "STRUCT:CLOSE":
                if not seen_child:
                    return {"status": "invalid"}
                index += 1
                return {"status": "complete", "next": index}
            return {"status": "invalid"}

    def parse_node(index: int) -> dict:
        if index >= len(tokens):
            return {"status": "incomplete"}
        token = tokens[index]
        if token == "NODE:TERM":
            return parse_term(index)
        if token == "NODE:FRAC":
            return parse_fraction(index)
        if isinstance(token, str) and token.startswith("OP:"):
            return parse_op_node(index)
        return {"status": "invalid"}

    result = parse_node(0)
    if result["status"] == "invalid":
        return False
    if result["status"] == "incomplete":
        return True
    return result["status"] == "complete" and result["next"] == len(tokens)


class NodeValidityPool:
    def __init__(self, script_path: str = "", num_workers: int = 1):
        pass

    def mask(self, tokens: List[str], candidate_tokens: List[str]) -> List[bool]:
        return [is_valid_prefix(tokens + [candidate]) for candidate in candidate_tokens]

    def close(self) -> None:
        pass


def flatten_vocab(vocab: Dict[str, Any]) -> Dict[str, int]:
    token_to_id = {}
    for key, value in vocab.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            token_to_id.update(value)
    return token_to_id


def load_vocab(vocab_path: str) -> Dict[str, Any]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    flat = flatten_vocab(raw)
    id_to_token = {idx: token for token, idx in flat.items()}
    return {
        "token_to_id": flat,
        "id_to_token": id_to_token,
        "special": raw.get("special_tokens", {}),
    }