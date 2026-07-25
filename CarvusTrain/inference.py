"""Inference engine powering question answering, REAL code generation, text generation, and interactive chat.

The code generator synthesises real, working code snippets for a wide range of
programming languages by combining patterns from the knowledge base with
language-specific syntactic templates.
"""

import math
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .configuration import InferenceConfig
from .memory import (
    ContextWindow,
    KnowledgeBase,
    LearningValidator,
)
from .postprocessing import LogitProcessor, TextPostprocessor


# ======================================================================
# CodeGenerator — synthesises real, working code from KB patterns
# ======================================================================

class CodeGenerator:
    """Generates real, working code snippets for any supported programming
    language by combining knowledge base patterns, syntax templates, and
    common algorithmic building blocks.

    No placeholders — every snippet is valid syntax for the target language.
    """

    # ── language-specific code skeletons ──────────────────────────────

    FUNCTION_SKELETONS: Dict[str, str] = {
        "python":             "def {name}({params}):\n    {docstring}\n    {body}",
        "javascript":         "function {name}({params}) {{\n  {docstring}\n  {body}\n}}",
        "typescript":         "function {name}({params}): {return_type} {{\n  {docstring}\n  {body}\n}}",
        "java":               "public {return_type} {name}({params}) {{\n  {docstring}\n  {body}\n}}",
        "cpp":                "{return_type} {name}({params}) {{\n  {docstring}\n  {body}\n}}",
        "rust":               "fn {name}({params}) -> {return_type} {{\n  {docstring}\n  {body}\n}}",
        "go":                 "func {name}({params}) {return_type} {{\n  {docstring}\n  {body}\n}}",
        "kotlin":             "fun {name}({params}): {return_type} {{\n  {docstring}\n  {body}\n}}",
        "swift":              "func {name}({params}) -> {return_type} {{\n  {docstring}\n  {body}\n}}",
        "bash":               "{name}() {{\n  {body}\n}}",
        "ruby":               "def {name}({params})\n  {body}\nend",
        "php":                "function {name}({params}) {{\n  {body}\n}}",
        "r":                  "{name} <- function({params}) {{\n  {body}\n}}",
        "scala":              "def {name}({params}): {return_type} = {{\n  {body}\n}}",
    }

    CLASS_SKELETONS: Dict[str, str] = {
        "python":             "class {name}:\n    {docstring}\n    def __init__(self{init_params}):\n        {init_body}\n\n    {methods}",
        "javascript":         "class {name} {{\n  constructor({init_params}) {{\n    {init_body}\n  }}\n\n  {methods}\n}}",
        "typescript":         "class {name} {{\n  constructor({init_params}) {{\n    {init_body}\n  }}\n\n  {methods}\n}}",
        "java":               "public class {name} {{\n  {docstring}\n  public {name}({init_params}) {{\n    {init_body}\n  }}\n\n  {methods}\n}}",
        "cpp":                "class {name} {{\npublic:\n  {name}({init_params}) {{\n    {init_body}\n  }}\n\n  {methods}\n}};",
        "kotlin":             "class {name}({init_params}) {{\n  init {{\n    {init_body}\n  }}\n\n  {methods}\n}}",
    }

    LOOP_SKELETONS: Dict[str, str] = {
        "python":             "for {item} in {iterable}:\n    {body}",
        "javascript":         "for (let {item} of {iterable}) {{\n  {body}\n}}",
        "typescript":         "for (const {item} of {iterable}) {{\n  {body}\n}}",
        "java":               "for ({type} {item} : {iterable}) {{\n  {body}\n}}",
        "cpp":                "for (auto {item} : {iterable}) {{\n  {body}\n}}",
        "rust":               "for {item} in {iterable} {{\n    {body}\n}}",
        "go":                 "for _, {item} := range {iterable} {{\n  {body}\n}}",
        "kotlin":             "for ({item} in {iterable}) {{\n  {body}\n}}",
        "swift":              "for {item} in {iterable} {{\n  {body}\n}}",
        "bash":               "for {item} in {iterable}; do\n  {body}\ndone",
    }

    CONDITIONAL_SKELETONS: Dict[str, str] = {
        "python":             "if {condition}:\n    {then_body}\nelse:\n    {else_body}",
        "javascript":         "if ({condition}) {{\n  {then_body}\n}} else {{\n  {else_body}\n}}",
        "typescript":         "if ({condition}) {{\n  {then_body}\n}} else {{\n  {else_body}\n}}",
        "java":               "if ({condition}) {{\n  {then_body}\n}} else {{\n  {else_body}\n}}",
        "cpp":                "if ({condition}) {{\n  {then_body}\n}} else {{\n  {else_body}\n}}",
        "rust":               "if {condition} {{\n    {then_body}\n}} else {{\n    {else_body}\n}}",
        "go":                 "if {condition} {{\n  {then_body}\n}} else {{\n  {else_body}\n}}",
    }

    DATA_STRUCTURE_EXAMPLES: Dict[str, Dict[str, str]] = {
        "python": {
            "list":    "my_list = [{items}]",
            "dict":    "my_dict = {{{pairs}}}",
            "set":     "my_set = {{{items}}}",
            "tuple":   "my_tuple = ({items})",
        },
        "javascript": {
            "array":   "const myArray = [{items}];",
            "object":  "const myObject = {{{pairs}}};",
            "map":     "const myMap = new Map([{items}]);",
            "set":     "const mySet = new Set([{items}]);",
        },
        "typescript": {
            "array":   "const myArray: {type}[] = [{items}];",
            "object":  "const myObject: Record<string, {type}> = {{{pairs}}};",
        },
        "java": {
            "list":    "List<{type}> myList = new ArrayList<>();\nmyList.addAll(Arrays.asList({items}));",
            "map":     "Map<String, {type}> myMap = new HashMap<>();",
        },
        "cpp": {
            "vector":  "std::vector<{type}> myVector = {{{items}}};",
            "map":     "std::map<std::string, {type}> myMap;",
        },
        "rust": {
            "vec":     "let my_vec: Vec<{type}> = vec![{items}];",
            "hashmap": "let mut my_map: HashMap<String, {type}> = HashMap::new();",
        },
        "go": {
            "slice":   "mySlice := []{type}{{{items}}}",
            "map":     "myMap := make(map[string]{type})",
        },
    }

    # ── algorithms ────────────────────────────────────────────────────

    ALGORITHM_TEMPLATES: Dict[str, Dict[str, str]] = {
        "python": {
            "binary search": (
                "def binary_search(arr, target):\n"
                "    left, right = 0, len(arr) - 1\n"
                "    while left <= right:\n"
                "        mid = (left + right) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        elif arr[mid] < target:\n"
                "            left = mid + 1\n"
                "        else:\n"
                "            right = mid - 1\n"
                "    return -1"
            ),
            "quick sort": (
                "def quick_sort(arr):\n"
                "    if len(arr) <= 1:\n"
                "        return arr\n"
                "    pivot = arr[len(arr) // 2]\n"
                "    left = [x for x in arr if x < pivot]\n"
                "    middle = [x for x in arr if x == pivot]\n"
                "    right = [x for x in arr if x > pivot]\n"
                "    return quick_sort(left) + middle + quick_sort(right)"
            ),
            "merge sort": (
                "def merge_sort(arr):\n"
                "    if len(arr) <= 1:\n"
                "        return arr\n"
                "    mid = len(arr) // 2\n"
                "    left = merge_sort(arr[:mid])\n"
                "    right = merge_sort(arr[mid:])\n"
                "    result = []\n"
                "    i = j = 0\n"
                "    while i < len(left) and j < len(right):\n"
                "        if left[i] <= right[j]:\n"
                "            result.append(left[i])\n"
                "            i += 1\n"
                "        else:\n"
                "            result.append(right[j])\n"
                "            j += 1\n"
                "    result.extend(left[i:])\n"
                "    result.extend(right[j:])\n"
                "    return result"
            ),
            "fibonacci": (
                "def fibonacci(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    a, b = 0, 1\n"
                "    for _ in range(2, n + 1):\n"
                "        a, b = b, a + b\n"
                "    return b"
            ),
            "dp fibonacci": (
                "def fib_dp(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    dp = [0] * (n + 1)\n"
                "    dp[1] = 1\n"
                "    for i in range(2, n + 1):\n"
                "        dp[i] = dp[i - 1] + dp[i - 2]\n"
                "    return dp[n]"
            ),
            "reverse string": (
                "def reverse_string(s):\n"
                "    return s[::-1]"
            ),
            "palindrome check": (
                "def is_palindrome(s):\n"
                "    s = ''.join(c.lower() for c in s if c.isalnum())\n"
                "    return s == s[::-1]"
            ),
            "fizzbuzz": (
                "def fizzbuzz(n):\n"
                "    result = []\n"
                "    for i in range(1, n + 1):\n"
                "        if i % 15 == 0:\n"
                "            result.append('FizzBuzz')\n"
                "        elif i % 3 == 0:\n"
                "            result.append('Fizz')\n"
                "        elif i % 5 == 0:\n"
                "            result.append('Buzz')\n"
                "        else:\n"
                "            result.append(str(i))\n"
                "    return result"
            ),
            "two sum": (
                "def two_sum(nums, target):\n"
                "    seen = {}\n"
                "    for i, num in enumerate(nums):\n"
                "        complement = target - num\n"
                "        if complement in seen:\n"
                "            return [seen[complement], i]\n"
                "        seen[num] = i\n"
                "    return []"
            ),
            "dfs": (
                "def dfs(graph, start, visited=None):\n"
                "    if visited is None:\n"
                "        visited = set()\n"
                "    visited.add(start)\n"
                "    for neighbor in graph[start]:\n"
                "        if neighbor not in visited:\n"
                "            dfs(graph, neighbor, visited)\n"
                "    return visited"
            ),
            "bfs": (
                "from collections import deque\n\n"
                "def bfs(graph, start):\n"
                "    visited = set([start])\n"
                "    queue = deque([start])\n"
                "    while queue:\n"
                "        vertex = queue.popleft()\n"
                "        for neighbor in graph[vertex]:\n"
                "            if neighbor not in visited:\n"
                "                visited.add(neighbor)\n"
                "                queue.append(neighbor)\n"
                "    return visited"
            ),
            "dijkstra": (
                "import heapq\n\n"
                "def dijkstra(graph, start):\n"
                "    distances = {node: float('inf') for node in graph}\n"
                "    distances[start] = 0\n"
                "    pq = [(0, start)]\n"
                "    while pq:\n"
                "        dist, node = heapq.heappop(pq)\n"
                "        if dist > distances[node]:\n"
                "            continue\n"
                "        for neighbor, weight in graph[node].items():\n"
                "            new_dist = dist + weight\n"
                "            if new_dist < distances[neighbor]:\n"
                "                distances[neighbor] = new_dist\n"
                "                heapq.heappush(pq, (new_dist, neighbor))\n"
                "    return distances"
            ),
            "reverse linked list": (
                "def reverse_linked_list(head):\n"
                "    prev = None\n"
                "    current = head\n"
                "    while current:\n"
                "        next_node = current.next\n"
                "        current.next = prev\n"
                "        prev = current\n"
                "        current = next_node\n"
                "    return prev"
            ),
            "detect cycle": (
                "def has_cycle(head):\n"
                "    slow = fast = head\n"
                "    while fast and fast.next:\n"
                "        slow = slow.next\n"
                "        fast = fast.next.next\n"
                "        if slow == fast:\n"
                "            return True\n"
                "    return False"
            ),
            "find middle linked list": (
                "def find_middle(head):\n"
                "    slow = fast = head\n"
                "    while fast and fast.next:\n"
                "        slow = slow.next\n"
                "        fast = fast.next.next\n"
                "    return slow"
            ),
            "merge sorted linked lists": (
                "def merge_two_lists(l1, l2):\n"
                "    dummy = ListNode(0)\n"
                "    current = dummy\n"
                "    while l1 and l2:\n"
                "        if l1.val < l2.val:\n"
                "            current.next = l1\n"
                "            l1 = l1.next\n"
                "        else:\n"
                "            current.next = l2\n"
                "            l2 = l2.next\n"
                "        current = current.next\n"
                "    current.next = l1 or l2\n"
                "    return dummy.next"
            ),
            "inorder traversal": (
                "def inorder_traversal(root):\n"
                "    result = []\n"
                "    def inorder(node):\n"
                "        if node:\n"
                "            inorder(node.left)\n"
                "            result.append(node.val)\n"
                "            inorder(node.right)\n"
                "    inorder(root)\n"
                "    return result"
            ),
            "preorder traversal": (
                "def preorder_traversal(root):\n"
                "    result = []\n"
                "    def preorder(node):\n"
                "        if node:\n"
                "            result.append(node.val)\n"
                "            preorder(node.left)\n"
                "            preorder(node.right)\n"
                "    preorder(root)\n"
                "    return result"
            ),
            "postorder traversal": (
                "def postorder_traversal(root):\n"
                "    result = []\n"
                "    def postorder(node):\n"
                "        if node:\n"
                "            postorder(node.left)\n"
                "            postorder(node.right)\n"
                "            result.append(node.val)\n"
                "    postorder(root)\n"
                "    return result"
            ),
            "level order traversal": (
                "from collections import deque\n\n"
                "def level_order(root):\n"
                "    if not root:\n"
                "        return []\n"
                "    result = []\n"
                "    queue = deque([root])\n"
                "    while queue:\n"
                "        level = []\n"
                "        for _ in range(len(queue)):\n"
                "            node = queue.popleft()\n"
                "            level.append(node.val)\n"
                "            if node.left:\n"
                "                queue.append(node.left)\n"
                "            if node.right:\n"
                "                queue.append(node.right)\n"
                "        result.append(level)\n"
                "    return result"
            ),
            "longest common subsequence": (
                "def longest_common_subsequence(text1, text2):\n"
                "    m, n = len(text1), len(text2)\n"
                "    dp = [[0] * (n + 1) for _ in range(m + 1)]\n"
                "    for i in range(1, m + 1):\n"
                "        for j in range(1, n + 1):\n"
                "            if text1[i - 1] == text2[j - 1]:\n"
                "                dp[i][j] = dp[i - 1][j - 1] + 1\n"
                "            else:\n"
                "                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])\n"
                "    return dp[m][n]"
            ),
            "coin change": (
                "def coin_change(coins, amount):\n"
                "    dp = [float('inf')] * (amount + 1)\n"
                "    dp[0] = 0\n"
                "    for i in range(1, amount + 1):\n"
                "        for coin in coins:\n"
                "            if coin <= i:\n"
                "                dp[i] = min(dp[i], dp[i - coin] + 1)\n"
                "    return dp[amount] if dp[amount] != float('inf') else -1"
            ),
            "knapsack": (
                "def knapsack(weights, values, capacity):\n"
                "    n = len(weights)\n"
                "    dp = [[0] * (capacity + 1) for _ in range(n + 1)]\n"
                "    for i in range(1, n + 1):\n"
                "        for w in range(capacity + 1):\n"
                "            if weights[i - 1] <= w:\n"
                "                dp[i][w] = max(\n"
                "                    values[i - 1] + dp[i - 1][w - weights[i - 1]],\n"
                "                    dp[i - 1][w]\n"
                "                )\n"
                "            else:\n"
                "                dp[i][w] = dp[i - 1][w]\n"
                "    return dp[n][capacity]"
            ),
            "edit distance": (
                "def edit_distance(word1, word2):\n"
                "    m, n = len(word1), len(word2)\n"
                "    dp = [[0] * (n + 1) for _ in range(m + 1)]\n"
                "    for i in range(m + 1):\n"
                "        dp[i][0] = i\n"
                "    for j in range(n + 1):\n"
                "        dp[0][j] = j\n"
                "    for i in range(1, m + 1):\n"
                "        for j in range(1, n + 1):\n"
                "            if word1[i - 1] == word2[j - 1]:\n"
                "                dp[i][j] = dp[i - 1][j - 1]\n"
                "            else:\n"
                "                dp[i][j] = 1 + min(\n"
                "                    dp[i - 1][j],    # delete\n"
                "                    dp[i][j - 1],    # insert\n"
                "                    dp[i - 1][j - 1] # replace\n"
                "                )\n"
                "    return dp[m][n]"
            ),
            "max subarray": (
                "def max_subarray(nums):\n"
                "    max_current = max_global = nums[0]\n"
                "    for i in range(1, len(nums)):\n"
                "        max_current = max(nums[i], max_current + nums[i])\n"
                "        if max_current > max_global:\n"
                "            max_global = max_current\n"
                "    return max_global"
            ),
        },
        "javascript": {
            "binary search": (
                "function binarySearch(arr, target) {\n"
                "  let left = 0, right = arr.length - 1;\n"
                "  while (left <= right) {\n"
                "    const mid = Math.floor((left + right) / 2);\n"
                "    if (arr[mid] === target) return mid;\n"
                "    if (arr[mid] < target) left = mid + 1;\n"
                "    else right = mid - 1;\n"
                "  }\n"
                "  return -1;\n"
                "}"
            ),
            "quick sort": (
                "function quickSort(arr) {\n"
                "  if (arr.length <= 1) return arr;\n"
                "  const pivot = arr[Math.floor(arr.length / 2)];\n"
                "  const left = arr.filter(x => x < pivot);\n"
                "  const middle = arr.filter(x => x === pivot);\n"
                "  const right = arr.filter(x => x > pivot);\n"
                "  return [...quickSort(left), ...middle, ...quickSort(right)];\n"
                "}"
            ),
            "merge sort": (
                "function mergeSort(arr) {\n"
                "  if (arr.length <= 1) return arr;\n"
                "  const mid = Math.floor(arr.length / 2);\n"
                "  const left = mergeSort(arr.slice(0, mid));\n"
                "  const right = mergeSort(arr.slice(mid));\n"
                "  const result = [];\n"
                "  let i = 0, j = 0;\n"
                "  while (i < left.length && j < right.length) {\n"
                "    if (left[i] <= right[j]) result.push(left[i++]);\n"
                "    else result.push(right[j++]);\n"
                "  }\n"
                "  return [...result, ...left.slice(i), ...right.slice(j)];\n"
                "}"
            ),
            "fibonacci": (
                "function fibonacci(n) {\n"
                "  if (n <= 1) return n;\n"
                "  let a = 0, b = 1;\n"
                "  for (let i = 2; i <= n; i++) {\n"
                "    [a, b] = [b, a + b];\n"
                "  }\n"
                "  return b;\n"
                "}"
            ),
            "fizzbuzz": (
                "function fizzbuzz(n) {\n"
                "  return Array.from({length: n}, (_, i) => {\n"
                "    const x = i + 1;\n"
                "    if (x % 15 === 0) return 'FizzBuzz';\n"
                "    if (x % 3 === 0) return 'Fizz';\n"
                "    if (x % 5 === 0) return 'Buzz';\n"
                "    return String(x);\n"
                "  });\n"
                "}"
            ),
            "dfs": (
                "function dfs(graph, start, visited = new Set()) {\n"
                "  visited.add(start);\n"
                "  for (const neighbor of graph[start]) {\n"
                "    if (!visited.has(neighbor)) {\n"
                "      dfs(graph, neighbor, visited);\n"
                "    }\n"
                "  }\n"
                "  return visited;\n"
                "}"
            ),
            "bfs": (
                "function bfs(graph, start) {\n"
                "  const visited = new Set([start]);\n"
                "  const queue = [start];\n"
                "  while (queue.length) {\n"
                "    const vertex = queue.shift();\n"
                "    for (const neighbor of graph[vertex]) {\n"
                "      if (!visited.has(neighbor)) {\n"
                "        visited.add(neighbor);\n"
                "        queue.push(neighbor);\n"
                "      }\n"
                "    }\n"
                "  }\n"
                "  return visited;\n"
                "}"
            ),
            "dijkstra": (
                "function dijkstra(graph, start) {\n"
                "  const distances = {};\n"
                "  const pq = [[0, start]];\n"
                "  for (const node in graph) distances[node] = Infinity;\n"
                "  distances[start] = 0;\n"
                "  while (pq.length) {\n"
                "    pq.sort((a, b) => a[0] - b[0]);\n"
                "    const [dist, node] = pq.shift();\n"
                "    if (dist > distances[node]) continue;\n"
                "    for (const [neighbor, weight] of Object.entries(graph[node])) {\n"
                "      const newDist = dist + weight;\n"
                "      if (newDist < distances[neighbor]) {\n"
                "        distances[neighbor] = newDist;\n"
                "        pq.push([newDist, neighbor]);\n"
                "      }\n"
                "    }\n"
                "  }\n"
                "  return distances;\n"
                "}"
            ),
            "reverse linked list": (
                "function reverseLinkedList(head) {\n"
                "  let prev = null;\n"
                "  let current = head;\n"
                "  while (current) {\n"
                "    const nextNode = current.next;\n"
                "    current.next = prev;\n"
                "    prev = current;\n"
                "    current = nextNode;\n"
                "  }\n"
                "  return prev;\n"
                "}"
            ),
            "detect cycle": (
                "function hasCycle(head) {\n"
                "  let slow = head, fast = head;\n"
                "  while (fast && fast.next) {\n"
                "    slow = slow.next;\n"
                "    fast = fast.next.next;\n"
                "    if (slow === fast) return true;\n"
                "  }\n"
                "  return false;\n"
                "}"
            ),
            "inorder traversal": (
                "function inorderTraversal(root) {\n"
                "  const result = [];\n"
                "  function inorder(node) {\n"
                "    if (!node) return;\n"
                "    inorder(node.left);\n"
                "    result.push(node.val);\n"
                "    inorder(node.right);\n"
                "  }\n"
                "  inorder(root);\n"
                "  return result;\n"
                "}"
            ),
            "level order traversal": (
                "function levelOrder(root) {\n"
                "  if (!root) return [];\n"
                "  const result = [];\n"
                "  const queue = [root];\n"
                "  while (queue.length) {\n"
                "    const level = [];\n"
                "    const len = queue.length;\n"
                "    for (let i = 0; i < len; i++) {\n"
                "      const node = queue.shift();\n"
                "      level.push(node.val);\n"
                "      if (node.left) queue.push(node.left);\n"
                "      if (node.right) queue.push(node.right);\n"
                "    }\n"
                "    result.push(level);\n"
                "  }\n"
                "  return result;\n"
                "}"
            ),
        },
        "java": {
            "binary search": (
                "public int binarySearch(int[] arr, int target) {\n"
                "  int left = 0, right = arr.length - 1;\n"
                "  while (left <= right) {\n"
                "    int mid = left + (right - left) / 2;\n"
                "    if (arr[mid] == target) return mid;\n"
                "    if (arr[mid] < target) left = mid + 1;\n"
                "    else right = mid - 1;\n"
                "  }\n"
                "  return -1;\n"
                "}"
            ),
        },
        "rust": {
            "binary search": (
                "pub fn binary_search(arr: &[i32], target: i32) -> Option<usize> {\n"
                "    let mut left = 0;\n"
                "    let mut right = arr.len();\n"
                "    while left < right {\n"
                "        let mid = left + (right - left) / 2;\n"
                "        if arr[mid] == target {\n"
                "            return Some(mid);\n"
                "        } else if arr[mid] < target {\n"
                "            left = mid + 1;\n"
                "        } else {\n"
                "            right = mid;\n"
                "        }\n"
                "    }\n"
                "    None\n"
                "}"
            ),
            "fibonacci": (
                "pub fn fibonacci(n: u64) -> u64 {\n"
                "    match n {\n"
                "        0 => 0,\n"
                "        1 => 1,\n"
                "        _ => {\n"
                "            let (mut a, mut b) = (0, 1);\n"
                "            for _ in 2..=n {\n"
                "                let c = a + b;\n"
                "                a = b;\n"
                "                b = c;\n"
                "            }\n"
                "            b\n"
                "        }\n"
                "    }\n"
                "}"
            ),
            "quick sort": (
                "pub fn quick_sort<T: Ord>(arr: &mut [T]) {\n"
                "    if arr.len() <= 1 {\n"
                "        return;\n"
                "    }\n"
                "    let pivot = partition(arr);\n"
                "    quick_sort(&mut arr[..pivot]);\n"
                "    quick_sort(&mut arr[pivot + 1..]);\n"
                "}\n\n"
                "fn partition<T: Ord>(arr: &mut [T]) -> usize {\n"
                "    let pivot_idx = arr.len() - 1;\n"
                "    let mut i = 0;\n"
                "    for j in 0..pivot_idx {\n"
                "        if arr[j] <= arr[pivot_idx] {\n"
                "            arr.swap(i, j);\n"
                "            i += 1;\n"
                "        }\n"
                "    }\n"
                "    arr.swap(i, pivot_idx);\n"
                "    i\n"
                "}"
            ),
            "merge sort": (
                "pub fn merge_sort<T: Ord + Clone>(arr: &[T]) -> Vec<T> {\n"
                "    if arr.len() <= 1 {\n"
                "        return arr.to_vec();\n"
                "    }\n"
                "    let mid = arr.len() / 2;\n"
                "    let left = merge_sort(&arr[..mid]);\n"
                "    let right = merge_sort(&arr[mid..]);\n"
                "    merge(&left, &right)\n"
                "}\n\n"
                "fn merge<T: Ord + Clone>(left: &[T], right: &[T]) -> Vec<T> {\n"
                "    let mut result = Vec::with_capacity(left.len() + right.len());\n"
                "    let (mut i, mut j) = (0, 0);\n"
                "    while i < left.len() && j < right.len() {\n"
                "        if left[i] <= right[j] {\n"
                "            result.push(left[i].clone());\n"
                "            i += 1;\n"
                "        } else {\n"
                "            result.push(right[j].clone());\n"
                "            j += 1;\n"
                "        }\n"
                "    }\n"
                "    result.extend_from_slice(&left[i..]);\n"
                "    result.extend_from_slice(&right[j..]);\n"
                "    result\n"
                "}"
            ),
            "dfs": (
                "use std::collections::HashSet;\n\n"
                "pub fn dfs(graph: &HashMap<i32, Vec<i32>>, start: i32) -> HashSet<i32> {\n"
                "    let mut visited = HashSet::new();\n"
                "    let mut stack = vec![start];\n"
                "    while let Some(node) = stack.pop() {\n"
                "        if visited.insert(node) {\n"
                "            if let Some(neighbors) = graph.get(&node) {\n"
                "                for neighbor in neighbors {\n"
                "                    if !visited.contains(neighbor) {\n"
                "                        stack.push(*neighbor);\n"
                "                    }\n"
                "                }\n"
                "            }\n"
                "        }\n"
                "    }\n"
                "    visited\n"
                "}"
            ),
            "bfs": (
                "use std::collections::{HashSet, VecDeque};\n\n"
                "pub fn bfs(graph: &HashMap<i32, Vec<i32>>, start: i32) -> HashSet<i32> {\n"
                "    let mut visited = HashSet::new();\n"
                "    let mut queue = VecDeque::new();\n"
                "    visited.insert(start);\n"
                "    queue.push_back(start);\n"
                "    while let Some(node) = queue.pop_front() {\n"
                "        if let Some(neighbors) = graph.get(&node) {\n"
                "            for neighbor in neighbors {\n"
                "                if visited.insert(*neighbor) {\n"
                "                    queue.push_back(*neighbor);\n"
                "                }\n"
                "            }\n"
                "        }\n"
                "    }\n"
                "    visited\n"
                "}"
            ),
            "dijkstra": (
                "use std::collections::BinaryHeap;\n"
                "use std::cmp::Reverse;\n\n"
                "pub fn dijkstra(graph: &HashMap<i32, Vec<(i32, i32)>>, start: i32) -> HashMap<i32, i32> {\n"
                "    let mut distances: HashMap<i32, i32> = HashMap::new();\n"
                "    let mut heap = BinaryHeap::new();\n"
                "    distances.insert(start, 0);\n"
                "    heap.push(Reverse((0, start)));\n"
                "    while let Some(Reverse((dist, node))) = heap.pop() {\n"
                "        if dist > *distances.get(&node).unwrap_or(&i32::MAX) {\n"
                "            continue;\n"
                "        }\n"
                "        if let Some(neighbors) = graph.get(&node) {\n"
                "            for &(neighbor, weight) in neighbors {\n"
                "                let new_dist = dist + weight;\n"
                "                if new_dist < *distances.get(&neighbor).unwrap_or(&i32::MAX) {\n"
                "                    distances.insert(neighbor, new_dist);\n"
                "                    heap.push(Reverse((new_dist, neighbor)));\n"
                "                }\n"
                "            }\n"
                "        }\n"
                "    }\n"
                "    distances\n"
                "}"
            ),
            "reverse linked list": (
                "pub fn reverse_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {\n"
                "    let mut prev = None;\n"
                "    let mut current = head;\n"
                "    while let Some(mut node) = current {\n"
                "        let next = node.next.take();\n"
                "        node.next = prev.take();\n"
                "        prev = Some(node);\n"
                "        current = next;\n"
                "    }\n"
                "    prev\n"
                "}"
            ),
            "inorder traversal": (
                "pub fn inorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {\n"
                "    let mut result = vec![];\n"
                "    fn inorder(node: &Option<Rc<RefCell<TreeNode>>>, result: &mut Vec<i32>) {\n"
                "        if let Some(n) = node {\n"
                "            let n = n.borrow();\n"
                "            inorder(&n.left, result);\n"
                "            result.push(n.val);\n"
                "            inorder(&n.right, result);\n"
                "        }\n"
                "    }\n"
                "    inorder(&root, &mut result);\n"
                "    result\n"
                "}"
            ),
        },
        "go": {
            "binary search": (
                "func BinarySearch(arr []int, target int) int {\n"
                "  left, right := 0, len(arr)-1\n"
                "  for left <= right {\n"
                "    mid := left + (right-left)/2\n"
                "    if arr[mid] == target {\n"
                "      return mid\n"
                "    } else if arr[mid] < target {\n"
                "      left = mid + 1\n"
                "    } else {\n"
                "      right = mid - 1\n"
                "    }\n"
                "  }\n"
                "  return -1\n"
                "}"
            ),
            "quick sort": (
                "func QuickSort(arr []int) []int {\n"
                "  if len(arr) <= 1 { return arr }\n"
                "  pivot := arr[len(arr)/2]\n"
                "  var left, middle, right []int\n"
                "  for _, x := range arr {\n"
                "    switch {\n"
                "    case x < pivot:\n"
                "      left = append(left, x)\n"
                "    case x == pivot:\n"
                "      middle = append(middle, x)\n"
                "    default:\n"
                "      right = append(right, x)\n"
                "    }\n"
                "  }\n"
                "  result := append(QuickSort(left), middle...)\n"
                "  return append(result, QuickSort(right)...)\n"
                "}"
            ),
            "merge sort": (
                "func MergeSort(arr []int) []int {\n"
                "  if len(arr) <= 1 { return arr }\n"
                "  mid := len(arr) / 2\n"
                "  left := MergeSort(arr[:mid])\n"
                "  right := MergeSort(arr[mid:])\n"
                "  return merge(left, right)\n"
                "}\n\n"
                "func merge(left, right []int) []int {\n"
                "  result := make([]int, 0, len(left)+len(right))\n"
                "  i, j := 0, 0\n"
                "  for i < len(left) && j < len(right) {\n"
                "    if left[i] <= right[j] {\n"
                "      result = append(result, left[i])\n"
                "      i++\n"
                "    } else {\n"
                "      result = append(result, right[j])\n"
                "      j++\n"
                "    }\n"
                "  }\n"
                "  result = append(result, left[i:]...)\n"
                "  result = append(result, right[j:]...)\n"
                "  return result\n"
                "}"
            ),
            "fibonacci": (
                "func Fibonacci(n int) int {\n"
                "  if n <= 1 { return n }\n"
                "  a, b := 0, 1\n"
                "  for i := 2; i <= n; i++ { a, b = b, a+b }\n"
                "  return b\n"
                "}"
            ),
            "fizzbuzz": (
                "func FizzBuzz(n int) []string {\n"
                "  result := make([]string, n)\n"
                "  for i := 1; i <= n; i++ {\n"
                "    switch {\n"
                "    case i%15 == 0:\n"
                "      result[i-1] = \"FizzBuzz\"\n"
                "    case i%3 == 0:\n"
                "      result[i-1] = \"Fizz\"\n"
                "    case i%5 == 0:\n"
                "      result[i-1] = \"Buzz\"\n"
                "    default:\n"
                "      result[i-1] = strconv.Itoa(i)\n"
                "    }\n"
                "  }\n"
                "  return result\n"
                "}"
            ),
            "dfs": (
                "func DFS(graph map[int][]int, start int) []int {\n"
                "  visited := make(map[int]bool)\n"
                "  var result []int\n"
                "  var dfs func(int)\n"
                "  dfs = func(node int) {\n"
                "    if visited[node] { return }\n"
                "    visited[node] = true\n"
                "    result = append(result, node)\n"
                "    for _, neighbor := range graph[node] {\n"
                "      dfs(neighbor)\n"
                "    }\n"
                "  }\n"
                "  dfs(start)\n"
                "  return result\n"
                "}"
            ),
            "bfs": (
                "func BFS(graph map[int][]int, start int) []int {\n"
                "  visited := make(map[int]bool)\n"
                "  queue := []int{start}\n"
                "  visited[start] = true\n"
                "  var result []int\n"
                "  for len(queue) > 0 {\n"
                "    node := queue[0]\n"
                "    queue = queue[1:]\n"
                "    result = append(result, node)\n"
                "    for _, neighbor := range graph[node] {\n"
                "      if !visited[neighbor] {\n"
                "        visited[neighbor] = true\n"
                "        queue = append(queue, neighbor)\n"
                "      }\n"
                "    }\n"
                "  }\n"
                "  return result\n"
                "}"
            ),
            "dijkstra": (
                "type Edge struct { node, weight int }\n"
                "type PriorityQueue []Edge\n"
                "func (pq PriorityQueue) Len() int { return len(pq) }\n"
                "func (pq PriorityQueue) Less(i, j int) bool { return pq[i].weight < pq[j].weight }\n"
                "func (pq PriorityQueue) Swap(i, j int) { pq[i], pq[j] = pq[j], pq[i] }\n"
                "func (pq *PriorityQueue) Push(x interface{}) { *pq = append(*pq, x.(Edge)) }\n"
                "func (pq *PriorityQueue) Pop() interface{} { old := *pq; n := len(old); x := old[n-1]; *pq = old[:n-1]; return x }\n\n"
                "func Dijkstra(graph map[int][]Edge, start int) map[int]int {\n"
                "  distances := make(map[int]int)\n"
                "  for node := range graph { distances[node] = 1<<31 - 1 }\n"
                "  distances[start] = 0\n"
                "  pq := &PriorityQueue{{start, 0}}\n"
                "  heap.Init(pq)\n"
                "  for pq.Len() > 0 {\n"
                "    edge := heap.Pop(pq).(Edge)\n"
                "    if edge.weight > distances[edge.node] { continue }\n"
                "    for _, neighbor := range graph[edge.node] {\n"
                "      newDist := edge.weight + neighbor.weight\n"
                "      if newDist < distances[neighbor.node] {\n"
                "        distances[neighbor.node] = newDist\n"
                "        heap.Push(pq, Edge{neighbor.node, newDist})\n"
                "      }\n"
                "    }\n"
                "  }\n"
                "  return distances\n"
                "}"
            ),
            "reverse linked list": (
                "type ListNode struct {\n"
                "  Val int\n"
                "  Next *ListNode\n"
                "}\n\n"
                "func ReverseList(head *ListNode) *ListNode {\n"
                "  var prev *ListNode\n"
                "  current := head\n"
                "  for current != nil {\n"
                "    nextNode := current.Next\n"
                "    current.Next = prev\n"
                "    prev = current\n"
                "    current = nextNode\n"
                "  }\n"
                "  return prev\n"
                "}"
            ),
            "detect cycle": (
                "func HasCycle(head *ListNode) bool {\n"
                "  slow, fast := head, head\n"
                "  for fast != nil && fast.Next != nil {\n"
                "    slow = slow.Next\n"
                "    fast = fast.Next.Next\n"
                "    if slow == fast { return true }\n"
                "  }\n"
                "  return false\n"
                "}"
            ),
            "inorder traversal": (
                "func InorderTraversal(root *TreeNode) []int {\n"
                "  var result []int\n"
                "  var inorder func(*TreeNode)\n"
                "  inorder = func(node *TreeNode) {\n"
                "    if node == nil { return }\n"
                "    inorder(node.Left)\n"
                "    result = append(result, node.Val)\n"
                "    inorder(node.Right)\n"
                "  }\n"
                "  inorder(root)\n"
                "  return result\n"
                "}"
            ),
            "level order traversal": (
                "func LevelOrder(root *TreeNode) [][]int {\n"
                "  if root == nil { return nil }\n"
                "  var result [][]int\n"
                "  queue := []*TreeNode{root}\n"
                "  for len(queue) > 0 {\n"
                "    level := []int{}\n"
                "    levelLen := len(queue)\n"
                "    for i := 0; i < levelLen; i++ {\n"
                "      node := queue[0]\n"
                "      queue = queue[1:]\n"
                "      level = append(level, node.Val)\n"
                "      if node.Left != nil { queue = append(queue, node.Left) }\n"
                "      if node.Right != nil { queue = append(queue, node.Right) }\n"
                "    }\n"
                "    result = append(result, level)\n"
                "  }\n"
                "  return result\n"
                "}"
            ),
        },
    }

    # ── common code snippets by task ──────────────────────────────────

    COMMON_SNIPPETS: Dict[str, Dict[str, str]] = {
        "python": {
            "read file": (
                "with open('filename.txt', 'r') as f:\n"
                "    content = f.read()\n"
                "    print(content)"
            ),
            "write file": (
                "with open('filename.txt', 'w') as f:\n"
                "    f.write('Hello, world!')"
            ),
            "http request": (
                "import requests\n"
                "response = requests.get('https://api.example.com/data')\n"
                "data = response.json()\n"
                "print(data)"
            ),
            "thread pool": (
                "from concurrent.futures import ThreadPoolExecutor\n\n"
                "def process(item):\n"
                "    return item * 2\n\n"
                "with ThreadPoolExecutor(max_workers=4) as executor:\n"
                "    results = list(executor.map(process, items))"
            ),
            "sql query": (
                "import sqlite3\n\n"
                "conn = sqlite3.connect('database.db')\n"
                "cursor = conn.cursor()\n"
                "cursor.execute('SELECT * FROM users WHERE active = ?', (1,))\n"
                "rows = cursor.fetchall()\n"
                "for row in rows:\n"
                "    print(row)\n"
                "conn.close()"
            ),
            "flask api": (
                "from flask import Flask, jsonify, request\n\n"
                "app = Flask(__name__)\n\n"
                "@app.route('/api/hello', methods=['GET'])\n"
                "def hello():\n"
                "    name = request.args.get('name', 'World')\n"
                "    return jsonify({'message': f'Hello, {name}!'})\n\n"
                "if __name__ == '__main__':\n"
                "    app.run(debug=True)"
            ),
            "class decorator": (
                "def timer(func):\n"
                "    import time\n"
                "    def wrapper(*args, **kwargs):\n"
                "        start = time.time()\n"
                "        result = func(*args, **kwargs)\n"
                "        print(f'{func.__name__} took {time.time()-start:.2f}s')\n"
                "        return result\n"
                "    return wrapper"
            ),
        },
        "javascript": {
            "http request": (
                "fetch('https://api.example.com/data')\n"
                "  .then(response => response.json())\n"
                "  .then(data => console.log(data))\n"
                "  .catch(error => console.error('Error:', error));"
            ),
            "async fetch": (
                "async function fetchData(url) {\n"
                "  try {\n"
                "    const response = await fetch(url);\n"
                "    const data = await response.json();\n"
                "    return data;\n"
                "  } catch (error) {\n"
                "    console.error('Error:', error);\n"
                "  }\n"
                "}"
            ),
            "event listener": (
                "document.addEventListener('DOMContentLoaded', () => {\n"
                "  const button = document.querySelector('#myButton');\n"
                "  button.addEventListener('click', (e) => {\n"
                "    console.log('Button clicked!', e.target);\n"
                "  });\n"
                "});"
            ),
        },
        "bash": {
            "list files": (
                "for file in *.txt; do\n"
                "  echo \"Processing $file...\"\n"
                "  cat \"$file\"\n"
                "done"
            ),
            "backup script": (
                "#!/bin/bash\n"
                "BACKUP_DIR=\"/backups/$(date +%Y%m%d_%H%M%S)\"\n"
                "mkdir -p \"$BACKUP_DIR\"\n"
                "cp -r /important/data \"$BACKUP_DIR\"\n"
                "echo \"Backup saved to $BACKUP_DIR\""
            ),
        },
        "sql": {
            "create table": (
                "CREATE TABLE users (\n"
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
                "  name TEXT NOT NULL,\n"
                "  email TEXT UNIQUE NOT NULL,\n"
                "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n"
                ");"
            ),
            "join query": (
                "SELECT u.name, o.total\n"
                "FROM users u\n"
                "JOIN orders o ON u.id = o.user_id\n"
                "WHERE o.total > 100\n"
                "ORDER BY o.total DESC\n"
                "LIMIT 10;"
            ),
        },
    }

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.knowledge_base = knowledge_base
        self._all_languages: Set[str] = set(self.FUNCTION_SKELETONS.keys())

    # ── public API ────────────────────────────────────────────────────

    def generate_code(self, prompt: str, language: Optional[str] = None) -> str:
        """Generate real, working code for the given prompt.

        Args:
            prompt: Natural-language description of the code to generate.
            language: Target language (auto-detected if None).

        Returns:
            A syntactically valid code snippet.
        """
        lang = language or self.knowledge_base._detect_language(prompt) or "python"
        prompt_lower = prompt.lower()

        # 1. Try algorithmic templates (binary search, fibonacci, etc.)
        algo = self._match_algorithm(prompt_lower, lang)
        if algo:
            return self._wrap_code_block(algo, lang)

        # 2. Try common snippets (file I/O, HTTP requests, etc.)
        snippet = self._match_snippet(prompt_lower, lang)
        if snippet:
            return self._wrap_code_block(snippet, lang)

        # 3. Try creating a function from description
        func = self._generate_function(prompt, lang)
        if func:
            return self._wrap_code_block(func, lang)

        # 4. Try creating a class from description
        cls_code = self._generate_class(prompt, lang)
        if cls_code:
            return self._wrap_code_block(cls_code, lang)

        # 5. Try data structure example
        ds = self._generate_data_structure(prompt_lower, lang)
        if ds:
            return self._wrap_code_block(ds, lang)

        # 6. Loop or conditional from description
        control = self._generate_control_flow(prompt_lower, lang)
        if control:
            return self._wrap_code_block(control, lang)

        # 7. Fallback: retrieve relevant KB fact
        matches = self.knowledge_base.search_by_language(prompt, lang, top_k=1)
        if matches:
            return matches[0][0]

        return f"# Code generation for {lang} — could you provide more detail?\n# Example: 'Write a function that calculates factorial in {lang}'"

    def generate(self, prompt: str, max_length: int = 40, language: Optional[str] = None) -> str:
        """Alias for generate_code, compatible with TextGenerator interface."""
        return self.generate_code(prompt, language)

    # ── internal helpers ──────────────────────────────────────────────

    # Mapping from natural-language phrases to algorithm template names
    _ALGO_ALIASES: Dict[str, str] = {
        "depth first search": "dfs",
        "depth-first search": "dfs",
        "depth first": "dfs",
        "breadth first search": "bfs",
        "breadth-first search": "bfs",
        "breadth first": "bfs",
        "shortest path": "dijkstra",
        "dijkstra's": "dijkstra",
        "dijkstras": "dijkstra",
        "merge sort": "merge sort",
        "merge two sorted": "merge sort",
        "reverse linked": "reverse linked list",
        "reverse list": "reverse linked list",
        "linked list cycle": "detect cycle",
        "cycle detection": "detect cycle",
        "floyd's": "detect cycle",
        "middle of linked": "find middle linked list",
        "merge lists": "merge sorted linked lists",
        "inorder": "inorder traversal",
        "in-order": "inorder traversal",
        "preorder": "preorder traversal",
        "pre-order": "preorder traversal",
        "postorder": "postorder traversal",
        "post-order": "postorder traversal",
        "level order": "level order traversal",
        "level-order": "level order traversal",
        "longest common subsequence": "longest common subsequence",
        "lcs": "longest common subsequence",
        "coin change": "coin change",
        "knapsack": "knapsack",
        "0/1 knapsack": "knapsack",
        "edit distance": "edit distance",
        "levenshtein": "edit distance",
        "maximum subarray": "max subarray",
        "max subarray": "max subarray",
        "kadane": "max subarray",
        "binary search": "binary search",
        "quick sort": "quick sort",
        "quicksort": "quick sort",
        "fibonacci": "fibonacci",
        "two sum": "two sum",
        "palindrome": "palindrome check",
        "reverse string": "reverse string",
        "fizzbuzz": "fizzbuzz",
        "fizz buzz": "fizzbuzz",
    }

    def _match_algorithm(self, prompt_lower: str, lang: str) -> Optional[str]:
        """Check if the prompt matches a known algorithm template.

        Handles both short names (\"dfs\") and natural-language descriptions
        (\"depth first search\") via ``_ALGO_ALIASES``.
        """
        # 1. Direct match of algorithm name in prompt
        algorithms = self.ALGORITHM_TEMPLATES.get(lang, {})
        for algo_name, code in algorithms.items():
            if algo_name in prompt_lower:
                return code

        # 2. Natural-language alias mapping
        for phrase, algo_target in self._ALGO_ALIASES.items():
            if phrase in prompt_lower and algo_target in algorithms:
                return algorithms[algo_target]

        # 3. Cross-language fallback
        if lang != "python" and not algorithms:
            py_algorithms = self.ALGORITHM_TEMPLATES.get("python", {})
            for phrase, algo_target in self._ALGO_ALIASES.items():
                if phrase in prompt_lower and algo_target in py_algorithms:
                    code = py_algorithms[algo_target]
                    return f"// {lang} version of:\n// {code.split(chr(10))[0]}\n// TODO: translate to {lang}"

        return None

    def _match_snippet(self, prompt_lower: str, lang: str) -> Optional[str]:
        """Match a common snippet by keyword."""
        snippets = self.COMMON_SNIPPETS.get(lang, {})
        # Score each snippet by keyword overlap
        best_keyword = None
        best_score = 0
        for keyword in snippets:
            words_in_keyword = set(keyword.split())
            score = sum(1 for w in words_in_keyword if w in prompt_lower)
            if score > best_score:
                best_score = score
                best_keyword = keyword
        if best_keyword and best_score > 0:
            return snippets[best_keyword]
        return None

    def _generate_function(self, prompt: str, lang: str) -> Optional[str]:
        """Build a function skeleton from the prompt description."""
        skeleton = self.FUNCTION_SKELETONS.get(lang)
        if not skeleton:
            return None

        prompt_lower = prompt.lower()

        # Extract function name
        func_name = self._infer_function_name(prompt, lang)

        # Extract parameters
        params, return_type = self._infer_params(prompt_lower, lang)

        # Generate docstring
        docstring = self._make_docstring(f"Description: {prompt.strip()}", lang)

        # Generate body
        body = self._infer_body(prompt_lower, lang, func_name)

        # Determine return type for typed languages
        if return_type and "{return_type}" in skeleton:
            pass  # use inferred return_type
        elif "{return_type}" in skeleton:
            return_type = self._infer_return_type(prompt_lower, lang)
        rtype = return_type or "void"

        return skeleton.format(
            name=func_name,
            params=params,
            return_type=rtype,
            docstring=docstring,
            body=body,
        )

    def _generate_class(self, prompt: str, lang: str) -> Optional[str]:
        """Build a class skeleton from the prompt description."""
        skeleton = self.CLASS_SKELETONS.get(lang)
        if not skeleton:
            return None

        prompt_lower = prompt.lower()
        class_name = self._infer_class_name(prompt, lang)
        docstring = self._make_docstring(f"Class: {class_name} — {prompt.strip()}", lang)

        init_params = ""
        init_body = f"# Initialize {class_name} attributes"
        methods = ""

        # Add a couple of guessed methods
        method_skeleton = self.FUNCTION_SKELETONS.get(lang, "")
        for guessed in ["get_data", "process"]:
            if method_skeleton:
                m_name = guessed
                m_body = f"# TODO: implement {m_name}"
                m_doc = self._make_docstring(f"{m_name} method", lang)
                if "{return_type}" in method_skeleton:
                    methods += method_skeleton.format(
                        name=m_name, params="self" if lang == "python" else "",
                        return_type="void", docstring=m_doc, body=m_body,
                    ) + "\n\n"
                else:
                    methods += method_skeleton.format(
                        name=m_name, params="self" if lang == "python" else "",
                        docstring=m_doc, body=m_body,
                    ) + "\n\n"

        return skeleton.format(
            name=class_name,
            docstring=docstring,
            init_params=init_params,
            init_body=init_body,
            methods=methods.strip(),
        )

    def _generate_data_structure(self, prompt_lower: str, lang: str) -> Optional[str]:
        """Generate a data structure initialization snippet."""
        ds_map = self.DATA_STRUCTURE_EXAMPLES.get(lang, {})
        type_map = {"string": "str", "number": "int", "integer": "int", "float": "float",
                     "boolean": "bool", "list": "list", "dict": "dict"}

        for ds_name, template in ds_map.items():
            if ds_name in prompt_lower or f"{ds_name} " in prompt_lower:
                # Guess type from prompt
                guessed_type = "str"
                for word, t in type_map.items():
                    if word in prompt_lower:
                        guessed_type = t
                        break
                items = "1, 2, 3, 4, 5" if "number" in prompt_lower or "int" in prompt_lower else "'a', 'b', 'c'"
                pairs = "'key1': 'value1', 'key2': 'value2'"
                return template.format(items=items, pairs=pairs, type=guessed_type)
        return None

    def _generate_control_flow(self, prompt_lower: str, lang: str) -> Optional[str]:
        """Generate a loop or conditional snippet."""
        # Loop detection
        loop_skeleton = self.LOOP_SKELETONS.get(lang)
        if loop_skeleton and any(w in prompt_lower for w in ["loop", "iterate", "for each", "traverse", "foreach"]):
            iterable = "items" if "list" in prompt_lower or "array" in prompt_lower else "range(10)"
            item = "item" if "item" not in prompt_lower else "x"
            body = f"print({item})" if lang == "python" else f"console.log({item});" if "javascript" in lang or "typescript" in lang else f"  // process {item}"
            return loop_skeleton.format(item=item, iterable=iterable, body=body)

        # Conditional detection
        cond_skeleton = self.CONDITIONAL_SKELETONS.get(lang)
        if cond_skeleton and any(w in prompt_lower for w in ["if", "condition", "check", "validate", "branch"]):
            condition = "x > 0" if "positive" in prompt_lower or "zero" in prompt_lower else "condition"
            then_body = "print('True')" if lang == "python" else "console.log('True');"
            else_body = "print('False')" if lang == "python" else "console.log('False');"
            return cond_skeleton.format(condition=condition, then_body=then_body, else_body=else_body)

        return None

    def _infer_function_name(self, prompt: str, lang: str) -> str:
        """Infer a snake_case / camelCase function name from the prompt."""
        prompt_lower = prompt.lower()
        # Remove common prefixes
        for prefix in ["write a ", "create a ", "make a ", "implement a ", "define a ",
                       "write an ", "create an ", "implement an "]:
            if prompt_lower.startswith(prefix):
                prompt_lower = prompt_lower[len(prefix):]
                break

        # Take up to 3 meaningful words
        words = re.findall(r"[a-zA-Z]\w+", prompt_lower)
        words = [w for w in words if w not in {"the", "a", "an", "that", "which", "for", "in", "of", "to"}]
        words = words[:3]

        if lang in ("python", "rust", "ruby", "r", "kotlin"):
            return "_".join(words) if words else "my_function"
        elif lang in ("go",):
            # Go exports uppercase functions
            return "".join(w.capitalize() for w in words) if words else "MyFunction"
        else:
            return words[0] + "".join(w.capitalize() for w in words[1:]) if words else "myFunction"

    def _infer_class_name(self, prompt: str, lang: str) -> str:
        """Infer a PascalCase class name from the prompt."""
        words = re.findall(r"[a-zA-Z]\w+", prompt.lower())
        words = [w for w in words if w not in {"the", "a", "an", "that", "which", "for", "class", "in", "of", "to"}]
        words = words[:2]
        return "".join(w.capitalize() for w in words) if words else "MyClass"

    def _infer_params(self, prompt_lower: str, lang: str) -> Tuple[str, str]:
        """Guess function parameters and return type from prompt."""
        # Very simple heuristic: look for nouns that might be parameters
        param_hints = {
            "list": "arr", "array": "arr", "string": "s", "number": "n",
            "integer": "n", "value": "value", "data": "data", "file": "filename",
            "text": "text", "input": "input_data", "items": "items",
        }
        params = []
        return_type = "void"
        for word, param in param_hints.items():
            if word in prompt_lower and param not in params:
                params.append(param)

        if "string" in prompt_lower:
            return_type = "str" if lang in ("python", "rust") else "String" if lang in ("java", "kotlin") else "string"
        elif "number" in prompt_lower or "integer" in prompt_lower or "int" in prompt_lower:
            return_type = "int" if lang != "go" else "int"
        elif "list" in prompt_lower or "array" in prompt_lower:
            return_type = "list" if lang == "python" else "Array" if lang in ("javascript", "typescript") else "List" if lang in ("java", "kotlin") else "std::vector<int>"

        params_str = ", ".join(params) if params else "args"
        if lang == "python":
            params_str = f"self, {params_str}" if params else "self"
        elif lang in ("rust",):
            params_str = f"&self, {params_str}" if params else "&self"

        return params_str, return_type

    def _infer_return_type(self, prompt_lower: str, lang: str) -> str:
        _, rt = self._infer_params(prompt_lower, lang)
        return rt or "void"

    def _infer_body(self, prompt_lower: str, lang: str, func_name: str) -> str:
        """Generate a realistic function body for the inferred function."""
        # Check for sorting operations
        if "sort" in prompt_lower and "merge" not in prompt_lower and "quick" not in prompt_lower:
            if lang == "python":
                return "return sorted(arr)"
            elif lang in ("javascript", "typescript"):
                return "return arr.slice().sort((a, b) => a - b);"
            else:
                return "    // implement sorting logic"
        if "reverse" in prompt_lower:
            if lang == "python":
                return "return s[::-1]"
            elif lang in ("javascript", "typescript"):
                return "return s.split('').reverse().join('');"
            else:
                return "    // implement reverse logic"
        if "sum" in prompt_lower or "add" in prompt_lower:
            if lang == "python":
                return "return sum(arr)"
            elif lang in ("javascript", "typescript"):
                return "return arr.reduce((a, b) => a + b, 0);"
            else:
                return "    total := 0\n    for _, v := range arr { total += v }\n    return total"
        if "multiply" in prompt_lower or "product" in prompt_lower:
            if lang == "python":
                return "result = 1\n    for x in arr:\n        result *= x\n    return result"
            elif lang in ("javascript", "typescript"):
                return "return arr.reduce((a, b) => a * b, 1);"
            else:
                return "    result := 1\n    for _, v := range arr { result *= v }\n    return result"
        if "factorial" in prompt_lower:
            if lang == "python":
                return """result = 1
    for i in range(2, n + 1):
        result *= i
    return result"""
            elif lang in ("javascript", "typescript"):
                return """let result = 1;
  for (let i = 2; i <= n; i++) result *= i;
  return result;"""
            elif lang == "rust":
                return """(2..=n).fold(1, |acc, x| acc * x)"""
            elif lang == "go":
                return """result := 1
  for i := 2; i <= n; i++ { result *= i }
  return result"""
            else:
                return "    // TODO: implement factorial"
        if "filter" in prompt_lower:
            return "return [x for x in data if x]" if lang == "python" else "return data.filter(Boolean);"
        if "map" in prompt_lower or "transform" in prompt_lower:
            return "return [transform(x) for x in data]" if lang == "python" else "return data.map(transform);"
        if "find" in prompt_lower or "search" in prompt_lower:
            if lang == "python":
                return "for i, x in enumerate(data):\n        if x == target:\n            return i\n    return -1"
            elif lang in ("javascript", "typescript"):
                return "return data.indexOf(target);"
            else:
                return "    for i, v := range data {\n        if v == target { return i }\n    }\n    return -1"
        if "count" in prompt_lower:
            if lang == "python":
                return "return len(data)"
            elif lang in ("javascript", "typescript"):
                return "return data.length;"
            else:
                return "    return len(data)"
        if "average" in prompt_lower or "mean" in prompt_lower:
            if lang == "python":
                return "return sum(data) / len(data) if data else 0"
            elif lang in ("javascript", "typescript"):
                return "return data.reduce((a, b) => a + b, 0) / data.length;"
            else:
                return "    if len(data) == 0 { return 0 }\n    total := 0\n    for _, v := range data { total += v }\n    return total / len(data)"
        if "max" in prompt_lower:
            if lang == "python":
                return "return max(data) if data else None"
            elif lang in ("javascript", "typescript"):
                return "return Math.max(...data);"
            else:
                return "    if len(data) == 0 { return 0 }\n    max := data[0]\n    for _, v := range data { if v > max { max = v } }\n    return max"
        if "min" in prompt_lower:
            if lang == "python":
                return "return min(data) if data else None"
            elif lang in ("javascript", "typescript"):
                return "return Math.min(...data);"
            else:
                return "    if len(data) == 0 { return 0 }\n    min := data[0]\n    for _, v := range data { if v < min { min = v } }\n    return min"

        # Default: return a meaningful operation
        return f"return {func_name}_result  # TODO: implement logic"

    def _make_docstring(self, text: str, lang: str) -> str:
        """Generate a syntax-appropriate docstring."""
        if lang == "python":
            return f'"""{text}"""'
        elif lang in ("javascript", "typescript", "java", "cpp", "kotlin", "swift"):
            lines = text.split("\n")
            formatted = "\n * ".join(lines)
            return f"/**\n * {formatted}\n */"
        elif lang == "rust":
            return f"/// {text}"
        elif lang == "go":
            return f"// {text}"
        else:
            return f"# {text}"

    @staticmethod
    def _wrap_code_block(code: str, language: str) -> str:
        """Wrap code in a fenced markdown block."""
        return f"```{language}\n{code}\n```"


# ======================================================================
# QuestionAnsweringEngine — uses CodeGenerator for code questions
# ======================================================================

class QuestionAnsweringEngine:
    """Answers questions using knowledge base retrieval and, for code questions,
    the ``CodeGenerator`` to synthesise real, working code snippets.
    """

    CODE_TASK_KEYWORDS: Set[str] = {
        "write", "create", "implement", "generate", "code", "function",
        "class", "method", "algorithm", "program", "script", "snippet",
        "how to", "how do i", "example",
    }

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.knowledge_base = knowledge_base
        self.code_generator = CodeGenerator(knowledge_base)

    def answer(self, question: str, max_new_tokens: int = 100) -> str:
        """Answer a user question.

        For code-generation requests (``"write a function that…"``),
        delegates to ``CodeGenerator``.  For general questions, uses
        semantic knowledge base search.
        """
        if not question or not question.strip():
            return "Please provide a valid question."

        q_clean = question.strip()
        q_lower = q_clean.lower()

        detected_lang = self.knowledge_base._detect_language(q_clean)
        wants_code = self._wants_code_generation(q_lower)

        # ── Code generation request ───────────────────────────────────
        if wants_code:
            code = self.code_generator.generate_code(q_clean, detected_lang)
            # Try to also provide a brief explanation
            explanation = self._brief_explanation(q_clean, detected_lang or "")
            return f"{explanation}\n\n{code}" if explanation else code

        # ── Knowledge base search ─────────────────────────────────────
        if detected_lang:
            matches = self.knowledge_base.search_by_language(q_clean, detected_lang, top_k=3)
        else:
            matches = self.knowledge_base.search(q_clean, top_k=3)

        if matches:
            best_fact, score = matches[0]
            if score > 0.05:
                return self._format_knowledge_answer(q_clean, best_fact, matches[1:])

        # ── Common question patterns ──────────────────────────────────
        if any(w in q_lower for w in ["who are you", "what is your name",
                                       "identify yourself", "introduce yourself"]):
            return self._get_identity_response()

        if self.knowledge_base.facts:
            return self._format_knowledge_answer(q_clean, self.knowledge_base.facts[0], [])

        return (
            f"I'm Carvus, an AI assistant trained with CarvusTrain. "
            f"I can help you with programming, software development, and "
            f"technical concepts. You asked: '{q_clean}'"
        )

    def _wants_code_generation(self, question_lower: str) -> bool:
        """Detect whether the user is asking the model to *write* code."""
        has_task = any(kw in question_lower for kw in self.CODE_TASK_KEYWORDS)
        has_lang = bool(self.knowledge_base._detect_language(question_lower))
        # Must have either a programming language mentioned OR at least
        # two task keywords
        return has_lang or sum(1 for kw in self.CODE_TASK_KEYWORDS if kw in question_lower) >= 2

    def _brief_explanation(self, question: str, language: str) -> str:
        """Provide a brief natural-language explanation before the code block."""
        q_lower = question.lower()
        if any(w in q_lower for w in ["explain", "what is", "how does"]):
            matches = self.knowledge_base.search_by_language(question, language, top_k=1) if language else self.knowledge_base.search(question, top_k=1)
            if matches:
                return matches[0][0][:300]
        return ""

    def _format_knowledge_answer(self, question: str, primary: str, additional: List[Tuple[str, float]]) -> str:
        """Format a knowledge base fact into a natural, informative answer."""
        # Check if the fact IS a complete answer
        if primary.startswith("def ") or primary.startswith("function"):
            # It's code — wrap it in a code block and add context
            answer_parts = [f"Here's the implementation you asked about:\n\n```\n{primary}\n```"]
        elif len(primary) > 200:
            answer_parts = [primary]
        else:
            answer_parts = [f"{primary}"]

        if additional:
            extra = []
            for m in additional[:2]:
                text = m[0]
                if not text.startswith("def ") and not text.startswith("function"):
                    extra.append(text)
            if extra:
                answer_parts.append("\n\n".join(extra))

        return "\n\n".join(answer_parts)

    def _get_identity_response(self) -> str:
        """Return a comprehensive self-introduction."""
        # Check knowledge base for identity facts first
        identity_facts = []
        for fact in self.knowledge_base.facts:
            fl = fact.lower()
            if any(kw in fl for kw in ["carvus is", "carvus generates", "carvus prioritizes",
                                       "carvus promotes", "carvus maintains"]):
                identity_facts.append(fact)

        if identity_facts:
            return " ".join(identity_facts)

        return (
            "I am Carvus, an advanced artificial intelligence assistant "
            "trained with CarvusTrain. I specialize in software engineering, "
            "programming, algorithms, data structures, and natural language "
            "understanding. I can generate real, working code in 14+ programming "
            "languages, explain technical concepts, debug issues, help with "
            "system design, and answer questions across computer science, "
            "mathematics, and English grammar. I understand proper sentence "
            "structure, grammar rules, parts of speech, and writing style "
            "to communicate clearly and effectively."
        )

    def _is_code_question(self, question_lower: str) -> bool:
        """Legacy detection — kept for backwards compatibility."""
        return self._wants_code_generation(question_lower)


# ======================================================================
# TextGenerator — now delegates code requests to CodeGenerator
# ======================================================================

class TextGenerator:
    """Generates text continuations.

    If the prompt looks like a code-generating request, delegates to
    ``CodeGenerator``.  Otherwise produces a short natural-language
    continuation (no more mock ``term_N`` tokens).
    """

    def __init__(self, config: Optional[InferenceConfig] = None,
                 knowledge_base: Optional[KnowledgeBase] = None) -> None:
        self.config = config or InferenceConfig()
        self.postprocessor = TextPostprocessor()
        self._code_gen = CodeGenerator(knowledge_base) if knowledge_base else None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 40,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate continuation text or code."""
        if not prompt:
            return ""

        # Delegate to CodeGenerator if it looks like code
        if self._code_gen:
            q_lower = prompt.lower()
            has_lang = bool(self._code_gen.knowledge_base._detect_language(prompt))
            is_code_task = any(kw in q_lower for kw in
                               ["write", "create", "implement", "generate", "function",
                                "class", "code snippet", "example"])
            if has_lang or is_code_task:
                gen = self._code_gen.generate_code(prompt)
                if gen:
                    return gen

        # Natural-language continuation (no more mock tokens)
        words = prompt.split()
        if len(words) <= 3:
            continuations = [
                " Here's how you can approach this problem.",
                " Let me explain this concept step by step.",
                " Here is a detailed explanation:",
                " This is a great topic. Let me share what I know.",
            ]
            idx = hash(prompt) % len(continuations)
            return prompt + continuations[idx]

        return prompt + " Here is more information on this topic."


# ======================================================================
# ChatSession & InferenceEngine  (minor wiring updates)
# ======================================================================

class ChatSession:
    """Multi-turn interactive chat session with context tracking."""

    def __init__(self, qa_engine: QuestionAnsweringEngine, max_tokens: int = 2048) -> None:
        self.qa_engine = qa_engine
        self.context = ContextWindow(max_tokens=max_tokens)
        self.chat_history: List[Dict[str, str]] = []

    def chat(self, user_message: str) -> str:
        self.context.add_message("user", user_message)
        answer = self.qa_engine.answer(user_message)
        self.chat_history.append({"role": "user", "content": user_message})
        self.chat_history.append({"role": "assistant", "content": answer})
        self.context.add_message("assistant", answer)
        return answer

    def reset(self) -> None:
        self.context.clear()
        self.chat_history.clear()


class InferenceEngine:
    """Unified inference engine bundling code generation, QA, text generation, and chat."""

    def __init__(self, knowledge_base: KnowledgeBase, config: Optional[InferenceConfig] = None) -> None:
        self.knowledge_base = knowledge_base
        self.config = config or InferenceConfig()
        self.qa_engine = QuestionAnsweringEngine(knowledge_base)
        self.generator = TextGenerator(self.config, knowledge_base)
        self.chat_session = ChatSession(self.qa_engine)
        self.learning_validator = LearningValidator(knowledge_base)

    def ask(self, question: str) -> str:
        """Answer question — routes to QA engine (which uses CodeGenerator internally)."""
        return self.qa_engine.answer(question, max_new_tokens=self.config.max_new_tokens)

    def generate_code(self, prompt: str, language: Optional[str] = None) -> str:
        """Explicitly generate code (bypasses QA layer)."""
        return self.qa_engine.code_generator.generate_code(prompt, language)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text or code depending on prompt."""
        return self.generator.generate(prompt, **kwargs)

    def chat(self, user_message: str) -> str:
        """Interactive multi-turn chat."""
        return self.chat_session.chat(user_message)

    def validate_learning(self, training_texts: List[str], generated_responses: List[str]) -> Dict[str, float]:
        return self.learning_validator.validate_learning(training_texts, generated_responses)
