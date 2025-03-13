from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from heapq import nlargest
from threading import Lock
from typing import Dict, List, Set

import numpy as np
from bpe_radix import RadixBalancedTree
from tqdm import tqdm


class RBTokenizer:
    def __init__(self, max_depth: int = 8, tech_terms: List[str] = None):
        self.rbt = RadixBalancedTree()
        self.merges: Dict[tuple, int] = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_tokens: Dict[str, int] = {}
        self.inverse_special: Dict[int, str] = {}
        self.max_depth = max_depth
        self.tech_terms: Set[bytes] = self._init_tech_terms(tech_terms or [])
        self._lock = Lock()

    def _init_tech_terms(self, terms: List[str]) -> Set[bytes]:
        return {term.encode("utf-8") for term in terms}

    def _count_pairs(self, seq: List[int]) -> Dict[tuple, int]:
        pairs = defaultdict(int)
        tech_window = 3
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            context = seq[max(0, i - tech_window) : min(len(seq), i + tech_window)]
            if any(self.vocab.get(tid, b"") in self.tech_terms for tid in context):
                pairs[pair] += 2
            else:
                pairs[pair] += 1
        return pairs

    def _premerge_technical_terms(self, text: str):
        for term_bytes in self.tech_terms:
            if not self.rbt.get_id(term_bytes):
                current_seq = list(term_bytes)
                while len(current_seq) > 1:
                    pairs = defaultdict(int)
                    for i in range(len(current_seq) - 1):
                        pairs[(current_seq[i], current_seq[i + 1])] += 1

                    if not pairs:
                        break

                    best_pair = max(pairs, key=lambda k: (pairs[k], -k[0], -k[1]))
                    new_id = 256 + len(self.merges)

                    merged_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
                    with self._lock:
                        self.merges[best_pair] = new_id
                        self.vocab[new_id] = merged_bytes
                        self.rbt._insert(merged_bytes, new_id)

                    current_seq = self._replace_pair(current_seq, best_pair, new_id)

    def _replace_pair(self, seq: List[int], pair: tuple, new_id: int) -> List[int]:
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
                new_seq.append(new_id)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        return new_seq

    def _parallel_batch_replace(
        self, sequences: List[List[int]], current_merges: List[tuple]
    ) -> List[List[int]]:
        replacements = {pair: self.merges[pair] for pair in current_merges}

        with ThreadPoolExecutor() as executor:
            futures = []
            for seq in sequences:
                futures.append(
                    executor.submit(self._replace_sequence, seq, replacements)
                )
            return [future.result() for future in futures]

    def _replace_sequence(self, seq: List[int], replacements: dict) -> List[int]:
        new_seq = []
        i = 0
        while i < len(seq):
            replaced = False
            for pair, new_id in replacements.items():
                if i <= len(seq) - 2 and (seq[i], seq[i + 1]) == pair:
                    new_seq.append(new_id)
                    i += 2
                    replaced = True
                    break
            if not replaced:
                new_seq.append(seq[i])
                i += 1
        return new_seq

    def _process_sequence(self, seq: List[int], pairs: List[tuple]) -> List[int]:
        arr = np.array(seq, dtype=np.uint32)
        mask = np.zeros_like(arr, dtype=bool)
        replacements = {}

        for pair in pairs:
            new_id = 256 + len(self.merges)
            pattern = np.array([pair[0], pair[1]], dtype=np.uint32)

            windows = np.lib.stride_tricks.sliding_window_view(arr, 2)
            matches = np.all(windows == pattern, axis=1)
            mask[:-1] |= matches

            replacements[pair] = new_id

        new_arr = []
        i = 0
        while i < len(arr):
            if i < len(mask) - 1 and mask[i]:
                pair = (arr[i], arr[i + 1])
                new_arr.append(replacements[pair])
                i += 2
            else:
                new_arr.append(arr[i])
                i += 1
        return new_arr

    def train(self, text: str, vocab_size: int, merge_batch_size: int = 32):
        text_bytes = list(text.encode("utf-8"))
        ids = [text_bytes]

        self._premerge_technical_terms(text)

        total_merges = vocab_size - 256 - len(self.merges)
        iterations = max(1, total_merges // merge_batch_size)

        with tqdm(total=iterations, desc="Training BPE") as pbar:
            for _ in range(iterations):
                batch_stats = defaultdict(int)
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self._count_pairs, seq) for seq in ids]
                    for future in futures:
                        for pair, count in future.result().items():
                            batch_stats[pair] += count

                top_pairs = nlargest(
                    merge_batch_size,
                    batch_stats.items(),
                    key=lambda x: (
                        x[1]
                        * (
                            2
                            if any(
                                self.vocab.get(p, b"") in self.tech_terms for p in x[0]
                            )
                            else 1
                        ),
                        -x[0][0],
                        -x[0][1],
                    ),
                )

                current_merges = {}
                for pair, _ in top_pairs:
                    new_id = 256 + len(self.merges)
                    merged_bytes = self.vocab[pair[0]] + self.vocab[pair[1]]

                    with self._lock:
                        self.merges[pair] = new_id
                        self.vocab[new_id] = merged_bytes
                        self.rbt._insert(merged_bytes, new_id)
                        current_merges[pair] = new_id

                ids = self._parallel_batch_replace(ids, current_merges)
                pbar.update(1)

    def encode(self, text: str) -> List[int]:
        byte_stream = text.encode("utf-8")
        ids = []
        i = 0

        while i < len(byte_stream):
            max_len = min(self.max_depth, len(byte_stream) - i)
            found = False

            for length in range(max_len, 0, -1):
                chunk = byte_stream[i : i + length]
                token_id = self.rbt.get_id(chunk)

                if token_id is not None:
                    ids.append(token_id)
                    i += length
                    found = True
                    break

            if not found:
                ids.append(byte_stream[i])
                i += 1

        return ids

    def decode(self, ids: List[int]) -> str:
        """Efficient decoding using vocabulary mapping"""
        byte_stream = bytearray()
        for token_id in ids:
            if token_id in self.vocab:
                byte_stream.extend(self.vocab[token_id])
            else:
                byte_stream.append(token_id)
        return byte_stream.decode("utf-8", errors="replace")

    def register_special_token(self, token: str, token_id: int):
        """Add special tokens to vocabulary"""
        token_bytes = token.encode("utf-8")
        with self._lock:
            self.special_tokens[token] = token_id
            self.inverse_special[token_id] = token
            self.vocab[token_id] = token_bytes
            self.rbt._insert(token_bytes, token_id)
