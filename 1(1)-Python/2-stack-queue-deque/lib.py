from __future__ import annotations
from collections import deque


"""
TODO:
- rotate_and_remove 구현하기 
"""


def create_circular_queue(n: int) -> deque[int]:
    """1부터 n까지의 숫자로 deque를 생성합니다."""
    return deque(range(1, n + 1))

def rotate_and_remove(queue: deque[int], k: int) -> int:
    """
    큐에서 k번째 원소를 제거하고 반환합니다.
    1. k번째 원소를 맨 앞에 오게 deque를 회전시킴
    2. 가장 앞의 값을 제거 후 반환
    """
    queue.rotate(-k+1)
    return queue.popleft()
