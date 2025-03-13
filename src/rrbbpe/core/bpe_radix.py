from collections import deque
from threading import Lock
from time import time


class CompressNode:
    __slots__ = ["prefix", "children", "count", "value", "last_accessed"]

    def __init__(self, prefix) -> None:
        self.prefix = prefix
        self.children = {}
        self.count = 0
        self.value = None
        self.last_accessed = time()


class RadixBalancedTree:
    def __init__(self, max_cache_size=1024):
        self.root = CompressNode(b"")
        self.total = 0
        self.merge_history = []
        self.max_cache_size = max_cache_size
        self.cache = deque(maxlen=max_cache_size)
        self.id_map = {}
        self.next_id = 256
        self.lock = Lock()
        self.byte_to_id = {bytes([i]): i for i in range(256)}

    def _insert(self, token_bytes, token_id):
        node = self.root
        i = 0
        while i < len(token_bytes):
            byte = token_bytes[i : i + 1]
            if byte in node.children:
                child = node.children[byte]
                prefix_len = len(child.prefix)

                if token_bytes[i : i + prefix_len] == child.prefix:
                    node = child
                    i += prefix_len
                else:
                    common_len = 0
                    max_len = min(prefix_len, len(token_bytes) - i)
                    while (
                        common_len < max_len
                        and child.prefix[common_len] == token_bytes[i + common_len]
                    ):
                        common_len += 1

                    split_node = CompressNode(child.prefix[:common_len])

                    child.prefix = child.prefix[common_len:]
                    split_node.children[child.prefix[:1]] = child

                    new_prefix = token_bytes[i + common_len :]
                    new_node = CompressNode(new_prefix)
                    new_node.value = token_id
                    split_node.children[new_prefix[:1]] = new_node

                    node.children[byte] = split_node
                    node = new_node
                    i += common_len
                    break
            else:
                new_node = CompressNode(token_bytes[i:])
                new_node.value = token_id
                node.children[byte] = new_node
                node = new_node
                i = len(token_bytes)
                break

        node.value = token_id
        self.id_map[token_id] = token_bytes
        return token_id

    def get_id(self, token_bytes):
        current_node = self.root
        i = 0

        while i < len(token_bytes):
            byte = token_bytes[i : i + 1]
            if byte not in current_node.children:
                return None

            child = current_node.children[byte]
            prefix = child.prefix

            if len(token_bytes) - i < len(prefix):
                return None

            if token_bytes[i : i + len(prefix)] != prefix:
                return None

            i += len(prefix)
            current_node = child

        if current_node.value:
            current_node.last_accessed = time()
            self._update_cache(current_node)

        return current_node.value

    def _update_cache(self, node):
        try:
            self.cache.remove(node)
        except Exception:
            pass

        self.cache.appendleft(node)
        while len(self.cache) > self.max_cache_size:
            removed = self.cache.pop()
            if removed.prefix and not removed.children:
                parent = self._find_parent(removed)
                if parent:
                    del parent.children[removed.prefi[0:1]]

    def _find_parent(self, target):
        queue = deque([(self.root, None)])
        while queue:
            node, parent = queue.popleft()
            for child in node.children.values():
                if child is target:
                    return parent
                queue.append((child, node))
        return None
