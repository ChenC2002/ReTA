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
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


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

    # reverse index
    _ccs_to_children: Dict[str, List[str]]

    @property
    def icd_vocab_size(self) -> int:
        return len(self._icd_tokens)

    @property
    def ccs_vocab_size(self) -> int:
        return len(self._ccs_tokens)

    @staticmethod
    def from_files(icd2ccs_path: str, ccs_hierarchy_path: str,
                   icd_col: str = 'icd', ccs_col: str = 'ccs',
                   child_col: str = 'child', parent_col: str = 'parent') -> 'CCSOntology':
        """Load mapping and hierarchy from CSV files.

        ICD->CCS mapping file format (CSV):
        - columns: [icd_col, ccs_col]

        CCS hierarchy file format (CSV):
        - columns: [child_col, parent_col]

        The hierarchy is assumed to be a tree (single parent per CCS node). If your CCS graph
        is a DAG, you can extend this to store multiple parents.
        """
        mdf = pd.read_csv(icd2ccs_path)
        hdf = pd.read_csv(ccs_hierarchy_path)

        for c in [icd_col, ccs_col]:
            if c not in mdf.columns:
                raise ValueError(f"ICD->CCS mapping file missing column '{c}'.")
        for c in [child_col, parent_col]:
            if c not in hdf.columns:
                raise ValueError(f"CCS hierarchy file missing column '{c}'.")

        icd_to_ccs = {str(i).strip().upper().replace('.', ''): str(c).strip() for i, c in zip(mdf[icd_col], mdf[ccs_col])}
        ccs_parent = {}
        for ch, pa in zip(hdf[child_col], hdf[parent_col]):
            ch = str(ch).strip()
            pa = str(pa).strip()
            if ch and pa:
                # keep first parent if duplicates exist
                if ch not in ccs_parent:
                    ccs_parent[ch] = pa

        # build token namespace
        icd_names = sorted(set(icd_to_ccs.keys()))
        ccs_names = sorted(set(icd_to_ccs.values()) | set(ccs_parent.keys()) | set(ccs_parent.values()))

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

        ccs_to_children: Dict[str, List[str]] = {}
        for ch, pa in ccs_parent.items():
            ccs_to_children.setdefault(pa, []).append(ch)

        return CCSOntology(
            icd_to_ccs=icd_to_ccs,
            ccs_parent=ccs_parent,
            name_to_token=name_to_token,
            token_to_name=token_to_name,
            _icd_tokens=icd_tokens,
            _ccs_tokens=ccs_tokens,
            _ccs_to_children=ccs_to_children,
        )

    # -------------------------
    # Token helpers
    # -------------------------

    def icd_to_token(self, icd_code: str) -> Optional[int]:
        icd_code = str(icd_code).strip().upper().replace('.', '')
        key = f"ICD:{icd_code}"
        return self.name_to_token.get(key)

    def ccs_to_token(self, ccs_code: str) -> Optional[int]:
        ccs_code = str(ccs_code).strip()
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
        node_set = set(node_ids)

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
                edges.append((tok, ptok))
                if undirected:
                    edges.append((ptok, tok))

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
                    edges.append((tok, ctok))
                    if undirected:
                        edges.append((ctok, tok))

        if len(edges) == 0:
            return np.zeros((2, 0), dtype=np.int64)

        edge_index = np.array(edges, dtype=np.int64).T
        return edge_index
