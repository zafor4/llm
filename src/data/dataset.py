"""Dataset preparation, curation, and PyTorch Causal Language Model Dataset."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from src.data.tokenizer import BaseTokenizer


# Built-in curated high-quality algorithmic Python corpus
# Cleanly licensed, educational algorithmic implementations with docstrings and typehints.
CURATED_ALGORITHMIC_TEMPLATES = [
    # 1. Fibonacci
    '''def fibonacci(n: int) -> int:
    """Compute the n-th Fibonacci number using dynamic programming.
    Time Complexity: O(n), Space Complexity: O(1).
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n in (0, 1):
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
''',
    # 2. Binary Search
    '''def binary_search(arr: list[int], target: int) -> int:
    """Find index of target in a sorted list using binary search.
    Returns -1 if target is not found.
    Time Complexity: O(log n), Space Complexity: O(1).
    """
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
''',
    # 3. Quick Sort
    '''def quick_sort(arr: list[int]) -> list[int]:
    """Sort an array using the divide-and-conquer quicksort algorithm.
    Time Complexity: O(n log n) average, O(n^2) worst case.
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
''',
    # 4. Merge Sort
    '''def merge_sort(arr: list[int]) -> list[int]:
    """Sort a list using merge sort algorithm.
    Time Complexity: O(n log n), Space Complexity: O(n).
    """
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
''',
    # 5. Greatest Common Divisor
    '''def gcd(a: int, b: int) -> int:
    """Compute Greatest Common Divisor using Euclidean algorithm.
    Time Complexity: O(log(min(a, b))).
    """
    while b != 0:
        a, b = b, a % b
    return abs(a)
''',
    # 6. Sieve of Eratosthenes
    '''def sieve_of_eratosthenes(limit: int) -> list[int]:
    """Find all prime numbers up to a given limit.
    Time Complexity: O(n log log n).
    """
    if limit < 2:
        return []
    primes = [True] * (limit + 1)
    primes[0] = primes[1] = False
    for p in range(2, int(limit**0.5) + 1):
        if primes[p]:
            for i in range(p * p, limit + 1, p):
                primes[i] = False
    return [p for p in range(2, limit + 1) if primes[p]]
''',
    # 7. Palindrome Checker
    '''def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome ignoring cases and punctuation."""
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]
''',
    # 8. Matrix Multiplication
    '''def matrix_multiply(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Multiply two 2D matrices A and B.
    Dimensions: A is (m x k), B is (k x n). Output is (m x n).
    """
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B:
        raise ValueError("Inner matrix dimensions must match")
    result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
    return result
''',
    # 9. Breadth First Search
    '''def bfs_shortest_path(graph: dict[str, list[str]], start: str, goal: str) -> list[str]:
    """Find shortest path in an unweighted graph using BFS."""
    from collections import deque
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return []
''',
    # 10. Depth First Search
    '''def dfs_traversal(graph: dict[str, list[str]], start: str) -> list[str]:
    """Traverse graph using Depth First Search and return visit order."""
    visited = []
    def _dfs(node: str) -> None:
        if node not in visited:
            visited.append(node)
            for neighbor in graph.get(node, []):
                _dfs(neighbor)
    _dfs(start)
    return visited
''',
    # 11. Stack implementation
    '''class Stack:
    """LIFO Stack data structure implemented via Python list."""
    def __init__(self):
        self._items = []

    def push(self, item) -> None:
        self._items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._items[-1]

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def size(self) -> int:
        return len(self._items)
''',
    # 12. Queue implementation
    '''class Queue:
    """FIFO Queue data structure implemented via collections.deque."""
    from collections import deque
    def __init__(self):
        from collections import deque
        self._items = deque()

    def enqueue(self, item) -> None:
        self._items.append(item)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._items.popleft()

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def size(self) -> int:
        return len(self._items)
''',
    # 13. Least Common Multiple
    '''def lcm(a: int, b: int) -> int:
    """Calculate the Least Common Multiple of two integers."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)
''',
    # 14. Knapsack Problem
    '''def knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:
    """Solve the 0/1 Knapsack problem using dynamic programming.
    Time Complexity: O(n * W), Space Complexity: O(n * W).
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][capacity]
''',
    # 15. Longest Common Subsequence
    '''def longest_common_subsequence(text1: str, text2: str) -> int:
    """Compute the length of the longest common subsequence of two strings."""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if text1[i] == text2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[m][n]
'''
]


def generate_synthetic_code_corpus(num_samples: int = 1200, seed: int = 42) -> List[str]:
    """Generate diverse variations of Python algorithms with randomized docstrings and parameters."""
    rng = random.Random(seed)
    corpus = []
    
    variable_names = [
        ("arr", "nums", "data", "items", "seq"),
        ("target", "val", "key", "query", "needle"),
        ("result", "ans", "res", "output", "out"),
        ("low", "left", "start", "lo"),
        ("high", "right", "end", "hi"),
    ]
    
    for i in range(num_samples):
        base_snippet = rng.choice(CURATED_ALGORITHMIC_TEMPLATES)
        # Systematic variation: slight comment injection, variable renaming, or docstring variations
        variant = base_snippet
        if rng.random() > 0.5:
            variant = f"# Algorithmic Example #{i}\n" + variant
        if rng.random() > 0.6:
            variant = variant.replace("arr", rng.choice(variable_names[0]))
        corpus.append(variant.strip())
    
    return corpus


def build_and_split_corpus(
    corpus: List[str],
    train_ratio: float = 0.85,
    val_ratio: float = 0.075,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Split corpus into train, val, and test partitions deterministically."""
    rng = random.Random(seed)
    shuffled = list(corpus)
    rng.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_data = shuffled[:n_train]
    val_data = shuffled[n_train : n_train + n_val]
    test_data = shuffled[n_train + n_val :]
    
    return train_data, val_data, test_data


class CausalLMDataset(Dataset):
    """PyTorch Dataset for Next-Token Prediction in Causal Language Modeling.
    
    Takes a 1D tensor or list of token IDs and slices into chunks of length (block_size + 1),
    where x = chunk[:-1] and y = chunk[1:].
    """

    def __init__(self, token_ids: List[int], block_size: int, stride: Optional[int] = None):
        self.block_size = block_size
        self.stride = stride or block_size  # Non-overlapping by default, or sliding window
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        
        # Calculate number of available chunks
        total_len = len(self.tokens)
        if total_len <= block_size:
            # Pad if shorter than block_size + 1
            padded = torch.zeros(block_size + 1, dtype=torch.long)
            padded[:total_len] = self.tokens
            self.tokens = padded
            self.num_chunks = 1
        else:
            self.num_chunks = max(1, (total_len - block_size - 1) // self.stride + 1)

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        chunk = self.tokens[start : start + self.block_size + 1]
        if len(chunk) < self.block_size + 1:
            pad_size = (self.block_size + 1) - len(chunk)
            chunk = torch.cat([chunk, torch.zeros(pad_size, dtype=torch.long)])
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
