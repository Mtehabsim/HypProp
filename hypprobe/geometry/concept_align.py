"""Locate concept tokens inside a saved sample, per role.

The tree probe needs, for each concept NODE of a prompt's ground-truth ontology,
a representation vector at (layer, role). Concepts are strings ("wumpus", "dog")
that BPE usually splits into several sub-tokens with leading-space markers
('Ġ' byte-level, '▁' SentencePiece). This module finds the token span(s) of each
concept by a sub-token sliding-window match over the saved ``tokens`` list —
tokenizer-free (no network on the DGX; we match on the stored strings directly)
with strict word-boundary guards so short real names ("cat") can't match inside
a longer token.

Roles (PREREGISTER3 §4):
  * ``premise`` — occurrences in the ``Every X is a Y`` premise block (the
    in-context definition site). Every node appears here, so this role yields a
    complete per-node cloud for tree decoding.
  * ``query``   — occurrences in the ``Question: … is a Z?`` clause (only the
    target concept + the entity's leaf appear here).
  * ``all``     — every prompt-side occurrence.

For a concept with several sub-tokens we return the LAST sub-token index by
default (the position where the word is fully disambiguated), with an option to
mean-pool the span.
"""

from __future__ import annotations

import numpy as np

_MARKERS = ("Ġ", "▁", "Ċ")
_PUNCT = set(".?!,:;)\"'")


def _clean(tok: str) -> str:
    return (tok.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")
            .strip().lower())


def _has_markers(tokens: list[str]) -> bool:
    return any(t.startswith(_MARKERS) or "Ġ" in t or "▁" in t for t in tokens)


def find_concept_spans(tokens: list[str], concepts: list[str]) -> dict[str, list[tuple[int, int]]]:
    """Return, per concept string, a list of ``(start, end)`` token spans (end exclusive).

    A span matches iff the concatenation of cleaned sub-tokens equals the
    (lowercased) concept AND it sits on word boundaries: the first sub-token is a
    word-start, and the token after the span is a word-start, punctuation, or the
    end of the sequence. When the tokenizer uses no space markers at all
    (word-level, e.g. mock data), every token is treated as a word-start and a
    single-token exact match is used.
    """
    cleaned = [_clean(t) for t in tokens]
    marked = _has_markers(tokens)

    def is_word_start(i: int) -> bool:
        if i == 0:
            return True
        if not marked:
            return True
        return tokens[i].startswith(_MARKERS)

    def boundary_after(j: int) -> bool:
        if j >= len(tokens):
            return True
        if not marked:
            return True
        if tokens[j].startswith(_MARKERS):
            return True
        c = cleaned[j]
        return bool(c) and c[0] in _PUNCT

    spans: dict[str, list[tuple[int, int]]] = {c: [] for c in concepts}
    lc = {c: c.lower() for c in concepts}
    max_sub = 8  # a concept won't span more than a handful of sub-tokens
    for i in range(len(tokens)):
        if not is_word_start(i) or not cleaned[i]:
            continue
        acc = ""
        for j in range(i, min(i + max_sub, len(tokens))):
            if j > i and is_word_start(j):
                break  # ran into the next word without a match
            acc += cleaned[j]
            for c in concepts:
                if acc == lc[c] and boundary_after(j + 1):
                    spans[c].append((i, j + 1))
    return spans


def _section_bounds(tokens: list[str]) -> int:
    """Index of the token that starts the 'Question:' clause (len(tokens) if none)."""
    cleaned = [_clean(t) for t in tokens]
    for i, c in enumerate(cleaned):
        if c.startswith("question"):
            return i
    return len(tokens)


def align_sample(sample: dict, role: str = "premise", pool: str = "last") -> dict:
    """Map each ground-truth concept node to a token index (or span mean) at a role.

    Returns ``{"node_ids": [...], "token_index": [...], "n_matched": int,
    "n_nodes": int}`` where ``token_index[k]`` is the chosen token position for
    node ``node_ids[k]``. Nodes with no matching token in the requested role are
    dropped (reported via ``n_matched``). ``pool='last'`` uses the last sub-token
    index; ``pool='mean'`` returns a list of span indices to be averaged by the
    caller.
    """
    tm = sample.get("tree_meta")
    if not tm:
        return {"node_ids": [], "token_index": [], "n_matched": 0, "n_nodes": 0}
    tokens = list(sample.get("tokens", []))
    names = tm["names"]
    prompt_len = int(sample.get("prompt_len", len(tokens)))
    q_start = _section_bounds(tokens)

    spans = find_concept_spans(tokens, names)

    def in_role(start: int, end: int) -> bool:
        if start >= prompt_len:          # prompt-side only (concept defs live there)
            return False
        if role == "premise":
            return end <= q_start
        if role == "query":
            return start >= q_start
        return True                       # 'all'

    node_ids, token_index, span_lists = [], [], []
    for nid, name in enumerate(names):
        occ = [(s, e) for (s, e) in spans.get(name, []) if in_role(s, e)]
        if not occ:
            continue
        # prefer the LAST occurrence (most contextualised); within it, last sub-tok
        s, e = occ[-1]
        node_ids.append(nid)
        token_index.append(e - 1 if pool == "last" else s)
        span_lists.append(list(range(s, e)))

    out = {"node_ids": node_ids, "token_index": token_index,
           "n_matched": len(node_ids), "n_nodes": len(names)}
    if pool == "mean":
        out["span_lists"] = span_lists
    return out


def concept_matrix(sample: dict, layer: int, role: str = "premise",
                   pool: str = "last"):
    """Return ``(X, node_ids, tree_distance_submatrix, node_depths)`` for one prompt.

    ``X`` is ``(n_matched, hidden)`` at ``layer`` for the matched concept nodes;
    the tree-distance submatrix and depths are restricted to those same nodes, so
    a decoder can regress decoded distances against ground-truth tree distances
    within this prompt. Returns ``None`` if fewer than 4 nodes matched (too few
    pairs to score).
    """
    al = align_sample(sample, role=role, pool=pool)
    if al["n_matched"] < 4:
        return None
    hidden = sample["hidden"]
    if hasattr(hidden, "numpy"):
        hidden = hidden.numpy()
    hidden = np.asarray(hidden, dtype=np.float64)
    L = min(layer, hidden.shape[0] - 1)
    h = hidden[L]  # (n_tok, hidden)

    idx = al["node_ids"]
    if pool == "mean" and "span_lists" in al:
        X = np.stack([h[sl].mean(axis=0) for sl in al["span_lists"]])
    else:
        X = np.stack([h[t] for t in al["token_index"]])

    tm = sample["tree_meta"]
    parent = tm["parent"]
    # exact tree distance over the matched nodes
    from ..data.prontoqa_tree import _tree_distance
    D_full = _tree_distance(parent)
    D = D_full[np.ix_(idx, idx)]
    depths = np.asarray([tm["depth"][i] for i in idx], dtype=float)
    return X, np.asarray(idx), D, depths
