"""CCS ontology utilities (ICD->CCS mapping + CCS hierarchy graph).

This module provides:
- ICD->CCS mapping
- CCS hierarchy edges (parent-child)
- Ancestor queries within <=h levels
- Subgraph edge extraction among a set of node ids

Tokenization strategy
---------------------
We use a *shared* integer token namespace for simplicity:
- ICD concepts get token ids first
- CCS concepts get token ids next
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


def normalize_icd_code(value: object) -> str:
    """Return an unprefixed, dot-free canonical ICD code."""

    code = str(value).strip().upper()
    if code.startswith("ICD:"):
        code = code[4:].strip()
    return code.replace(".", "")


def normalize_icd_version(value: object) -> str:
    """Normalize supported ICD version labels to ``9`` or ``10``."""

    version = str(value).strip().upper().replace("ICD-", "").replace("ICD", "")
    if version.endswith(".0"):
        version = version[:-2]
    if version not in {"9", "10"}:
        raise ValueError(f"unsupported ICD version {value!r}; expected 9 or 10")
    return version


def qualify_icd_code(code: object, version: Optional[object] = None) -> str:
    """Build the internal mapping key for an optionally versioned ICD code."""

    normalized = normalize_icd_code(code)
    if not normalized:
        return ""
    if version is None:
        return normalized
    return f"{normalize_icd_version(version)}:{normalized}"


def normalize_ccs_code(value: object) -> str:
    """Return an unprefixed canonical CCS code."""

    code = str(value).strip()
    if code.upper().startswith("CCS:"):
        code = code[4:].strip()
    return code


def canonicalize_ccs_id(value: object) -> str:
    """Return the shared external form for a CCS identifier."""

    code = normalize_ccs_code(value)
    return f"CCS:{code}" if code else ""


def _validate_tree_acyclic(parent_by_child: Dict[str, str]) -> None:
    """Reject cycles in a single-parent hierarchy."""

    complete: Set[str] = set()
    for start in parent_by_child:
        if start in complete:
            continue
        path: List[str] = []
        positions: Dict[str, int] = {}
        node = start
        while node in parent_by_child and node not in complete:
            if node in positions:
                cycle = path[positions[node] :] + [node]
                raise ValueError(f"CCS hierarchy contains a cycle: {' -> '.join(cycle)}")
            positions[node] = len(path)
            path.append(node)
            node = parent_by_child[node]
        complete.update(path)


@dataclass
class CCSOntology:
    # core mappings
    icd_to_ccs: Dict[str, str]                     # ICD string -> CCS string
    ccs_parent: Dict[str, str]                     # CCS child -> CCS parent (single-parent tree)

    # tokenization
    name_to_token: Dict[str, int]
    token_to_name: Dict[int, str]
    _icd_tokens: Set[int]
    _ccs_tokens: Set[int]
    _ccs_label_tokens: Set[int]
    icd_versioned: bool = False

    @property
    def icd_vocab_size(self) -> int:
        return len(self._icd_tokens)

    @property
    def ccs_vocab_size(self) -> int:
        return len(self._ccs_tokens)

    @property
    def ccs_label_vocab_size(self) -> int:
        """Number of direct ICD-mapped CCS prediction labels."""

        return len(self._ccs_label_tokens)

    @staticmethod
    def from_files(icd2ccs_path: str, ccs_hierarchy_path: str,
                   icd_col: str = 'icd', ccs_col: str = 'ccs',
                   child_col: str = 'child', parent_col: str = 'parent',
                   icd_version_col: Optional[str] = None) -> 'CCSOntology':
        """Load mapping and hierarchy from CSV files.

        ICD->CCS mapping file format (CSV):
        - columns: [icd_col, ccs_col]
        - optionally, ``icd_version_col`` containing 9 or 10

        CCS hierarchy file format (CSV):
        - columns: [child_col, parent_col]

        The hierarchy must be an acyclic tree with at most one parent per CCS node.
        """
        # Ontology identifiers are categorical strings. Letting pandas infer
        # numeric dtypes would destroy leading zeroes (for example ``001.0``).
        mdf = pd.read_csv(icd2ccs_path, dtype=str, keep_default_na=False)
        hdf = pd.read_csv(ccs_hierarchy_path, dtype=str, keep_default_na=False)

        for c in [icd_col, ccs_col]:
            if c not in mdf.columns:
                raise ValueError(f"ICD->CCS mapping file missing column '{c}'.")
        if icd_version_col is not None and icd_version_col not in mdf.columns:
            raise ValueError(f"ICD->CCS mapping file missing column '{icd_version_col}'.")
        for c in [child_col, parent_col]:
            if c not in hdf.columns:
                raise ValueError(f"CCS hierarchy file missing column '{c}'.")

        icd_to_ccs = {}
        mapping_versions = mdf[icd_version_col] if icd_version_col is not None else [None] * len(mdf)
        for i, c, version in zip(mdf[icd_col], mdf[ccs_col], mapping_versions):
            raw_icd = normalize_icd_code(i)
            ccs = normalize_ccs_code(c)
            if not raw_icd or not ccs or raw_icd.lower() == "nan" or ccs.lower() == "nan":
                continue
            try:
                icd = qualify_icd_code(raw_icd, version)
            except ValueError as exc:
                raise ValueError(f"invalid ICD mapping version for code {i!r}: {exc}") from exc
            previous = icd_to_ccs.get(icd)
            if previous is not None and previous != ccs:
                raise ValueError(
                    f"Conflicting ICD-to-CCS mappings for {icd!r}: "
                    f"{previous!r} and {ccs!r}."
                )
            icd_to_ccs[icd] = ccs
        ccs_parent = {}
        for ch, pa in zip(hdf[child_col], hdf[parent_col]):
            ch = normalize_ccs_code(ch)
            pa = normalize_ccs_code(pa)
            if ch and pa and ch.lower() != "nan" and pa.lower() != "nan":
                if ch == pa:
                    raise ValueError(f"CCS hierarchy contains a self-cycle at {ch!r}.")
                previous = ccs_parent.get(ch)
                if previous is not None and previous != pa:
                    raise ValueError(
                        f"Conflicting CCS parents for {ch!r}: "
                        f"{previous!r} and {pa!r}."
                    )
                ccs_parent[ch] = pa

        if not icd_to_ccs:
            raise ValueError("ICD-to-CCS mapping contains no valid records.")
        _validate_tree_acyclic(ccs_parent)

        # build token namespace
        icd_names = sorted(set(icd_to_ccs.keys()))
        direct_ccs_names = sorted(set(icd_to_ccs.values()))
        all_ccs_names = set(direct_ccs_names) | set(ccs_parent.keys()) | set(ccs_parent.values())
        ancestor_only_ccs_names = sorted(all_ccs_names - set(direct_ccs_names))
        # Prediction labels occupy the first contiguous CCS block; hierarchy-
        # only graph nodes follow and are not included in the label head.
        ccs_names = direct_ccs_names + ancestor_only_ccs_names

        name_to_token: Dict[str, int] = {}
        token_to_name: Dict[int, str] = {}

        cur = 0
        for name in icd_names:
            name_to_token[f"ICD:{name}"] = cur
            token_to_name[cur] = f"ICD:{name}"
            cur += 1
        for name in ccs_names:
            name_to_token[f"CCS:{name}"] = cur
            token_to_name[cur] = f"CCS:{name}"
            cur += 1

        icd_tokens = {name_to_token[f"ICD:{n}"] for n in icd_names}
        ccs_tokens = {name_to_token[f"CCS:{n}"] for n in ccs_names}
        ccs_label_tokens = {name_to_token[f"CCS:{n}"] for n in direct_ccs_names}

        return CCSOntology(
            icd_to_ccs=icd_to_ccs,
            ccs_parent=ccs_parent,
            name_to_token=name_to_token,
            token_to_name=token_to_name,
            _icd_tokens=icd_tokens,
            _ccs_tokens=ccs_tokens,
            _ccs_label_tokens=ccs_label_tokens,
            icd_versioned=icd_version_col is not None,
        )

    # -------------------------
    # Token helpers
    # -------------------------

    def icd_to_token(self, icd_code: str, version: Optional[object] = None) -> Optional[int]:
        if self.icd_versioned and version is None:
            raise ValueError("this ontology requires an ICD version for every lookup")
        icd_code = qualify_icd_code(icd_code, version if self.icd_versioned else None)
        key = f"ICD:{icd_code}"
        return self.name_to_token.get(key)

    def ccs_to_token(self, ccs_code: str) -> Optional[int]:
        ccs_code = normalize_ccs_code(ccs_code)
        key = f"CCS:{ccs_code}"
        return self.name_to_token.get(key)

    def token_is_icd(self, tok: int) -> bool:
        return int(tok) in self._icd_tokens

    def token_is_ccs(self, tok: int) -> bool:
        return int(tok) in self._ccs_tokens

    def token_to_ccs_code(self, tok: int) -> Optional[str]:
        name = self.token_to_name.get(int(tok))
        if name and name.startswith('CCS:'):
            return name.split(':', 1)[1]
        return None

    def token_to_icd_code(self, tok: int) -> Optional[str]:
        name = self.token_to_name.get(int(tok))
        if name and name.startswith('ICD:'):
            return name.split(':', 1)[1]
        return None

    # -------------------------
    # Mapping & ancestry
    # -------------------------

    def icd_tokens_to_ccs_tokens(self, icd_tokens: Iterable[int]) -> List[int]:
        """Map ICD token ids to CCS token ids (direct CCS)."""
        out = []
        for tok in icd_tokens:
            icd = self.token_to_icd_code(tok)
            if icd is None:
                continue
            ccs = self.icd_to_ccs.get(icd)
            if ccs is None:
                continue
            ccs_tok = self.ccs_to_token(ccs)
            if ccs_tok is not None:
                out.append(ccs_tok)
        return out

    def ancestors_within_h(self, ccs_tokens: Iterable[int], h: int = 2) -> List[int]:
        """Return CCS ancestors within <=h levels for given CCS tokens.

        Includes the input CCS tokens as level-0 nodes.
        """
        try:
            h = operator.index(h)
        except TypeError as exc:
            raise TypeError("h must be an integer.") from exc
        if h < 0:
            raise ValueError("h must be non-negative.")
        result: Set[int] = set(int(t) for t in ccs_tokens)
        frontier: Set[int] = set(int(t) for t in ccs_tokens)

        for _ in range(h):
            next_frontier: Set[int] = set()
            for tok in frontier:
                code = self.token_to_ccs_code(tok)
                if code is None:
                    continue
                parent = self.ccs_parent.get(code)
                if parent is None:
                    continue
                ptok = self.ccs_to_token(parent)
                if ptok is not None and ptok not in result:
                    next_frontier.add(ptok)
                    result.add(ptok)
            frontier = next_frontier
            if not frontier:
                break
        return sorted(result)

    # -------------------------
    # Graph edges
    # -------------------------

    def edges_among_nodes(self, node_ids: Iterable[int], undirected: bool = True, add_icd_ccs: bool = True) -> np.ndarray:
        """Extract ontology edges among the given nodes.

        We add two types of edges (treated as *untyped* adjacency downstream):
        1) CCS hierarchy edges: (child CCS) <-> (parent CCS)
        2) ICD-to-CCS edges: (ICD leaf) <-> (its mapped CCS)

        Parameters
        ----------
        node_ids:
            Token ids included in the visit graph.
        undirected:
            If True, symmetrize all edges.
        add_icd_ccs:
            If True, include ICD<->CCS mapping links when both endpoints appear in `node_ids`.

        Returns
        -------
        edge_index: np.ndarray of shape (2, E)
        """
        node_ids = [int(x) for x in node_ids]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node_ids must be unique.")
        node_set = set(node_ids)
        local_index = {token_id: index for index, token_id in enumerate(node_ids)}

        edges: List[Tuple[int, int]] = []

        # (A) CCS hierarchy edges among included CCS nodes
        for tok in node_ids:
            if not self.token_is_ccs(tok):
                continue
            code = self.token_to_ccs_code(tok)
            if code is None:
                continue
            parent = self.ccs_parent.get(code)
            if parent is None:
                continue
            ptok = self.ccs_to_token(parent)
            if ptok is None:
                continue
            if ptok in node_set:
                edges.append((local_index[tok], local_index[ptok]))
                if undirected:
                    edges.append((local_index[ptok], local_index[tok]))

        # (B) ICD <-> CCS leaf links
        if add_icd_ccs:
            for tok in node_ids:
                if not self.token_is_icd(tok):
                    continue
                icd = self.token_to_icd_code(tok)
                if icd is None:
                    continue
                ccs = self.icd_to_ccs.get(icd)
                if ccs is None:
                    continue
                ctok = self.ccs_to_token(ccs)
                if ctok is None:
                    continue
                if ctok in node_set:
                    edges.append((local_index[tok], local_index[ctok]))
                    if undirected:
                        edges.append((local_index[ctok], local_index[tok]))

        if len(edges) == 0:
            return np.zeros((2, 0), dtype=np.int64)

        edge_index = np.array(edges, dtype=np.int64).T
        return edge_index
