"""
Hierarchical type encoding using Trie structure.

Implements hierarchical one-hot encoding for circuit node types
using a Trie data structure.
"""

from typing import Dict, List, Tuple


class Trie:
    """Trie for hierarchical type encoding."""

    def __init__(self):
        self.children = {}
        self.children_order = {}
        self.branch_factor = 0
        self.is_leaf = False

    def add(self, hier: Tuple[str, ...]):
        """Add a hierarchical type to the trie."""
        node = self
        for char in hier:
            if char not in node.children:
                child = Trie()
                node.children[char] = child
                node.children_order[char] = node.branch_factor
                node.branch_factor += 1
            node = node.children[char]
        node.is_leaf = True

    def get_leaf_encodings(self) -> Dict[Tuple[str, ...], List[int]]:
        """Get one-hot encodings for all leaf nodes."""
        if self.is_leaf:
            return {}
        enc_dict = {}
        for char in self.children:
            child_enc = self.children[char].get_leaf_encodings()
            char_enc = self._one_hot(self.children_order[char], self.branch_factor)
            if child_enc:
                for sub_key, sub_enc in child_enc.items():
                    key = (char,) + sub_key
                    enc_dict[key] = char_enc + sub_enc
            else:
                enc_dict[(char,)] = char_enc
        return enc_dict

    @staticmethod
    def _one_hot(p: int, ncodes: int) -> List[int]:
        """Create one-hot encoding."""
        enc = [0] * ncodes
        enc[p] = 1
        return enc


def build_circuit_trie() -> Trie:
    """Build the standard circuit Trie for hierarchical type encoding."""
    trie = Trie()
    # MOSFET terminals
    for mos_type in ['N', 'P']:
        for term in ['D', 'G', 'S', 'B']:
            trie.add(('M', mos_type, term))
    # Two-terminal devices
    for dev in ['R', 'C', 'V', 'I']:
        for term in ['P', 'M']:
            trie.add((dev, term))
    # Voltage nodes
    trie.add(('VNode', 'GND'))
    trie.add(('VNode', 'NGND'))
    return trie
