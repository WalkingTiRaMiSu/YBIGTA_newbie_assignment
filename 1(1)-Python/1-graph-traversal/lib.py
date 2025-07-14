from __future__ import annotations
import copy
from collections import deque
from collections import defaultdict
from typing import DefaultDict, List


"""
TODO:
- __init__ 구현하기
- add_edge 구현하기
- dfs 구현하기 (재귀 또는 스택 방식 선택)
- bfs 구현하기
"""


class Graph:
    def __init__(self, n: int) -> None:
        """
        그래프 초기화
        n: 정점의 개수 (1번부터 n번까지)
        """
        self.n = n
        self.graph: DefaultDict[int, List[int]] = defaultdict(list)

    
    def add_edge(self, u: int, v: int) -> None:
        """
        양방향 간선 추가
        """
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def dfs(self, start: int) -> list[int]:
        """
        깊이 우선 탐색 (DFS)
        
        구현 방법 선택:
        1. 재귀 방식: 함수 내부에서 재귀 함수 정의하여 구현
        2. 스택 방식: 명시적 스택을 사용하여 반복문으로 구현

        <재귀 방식>
        1. start부터 출발
        2. start와 이웃된 노드 중 방문하지 않은 노드가 있다면 방문
        3. 그 노드와 이웃된 노드 중 방문하지 않은 노드 탐색 후 방문
        4. 3 반복, 종료 후 2로 돌아감
        """
        visited = []
        def find_neighbor(n):
            visited.append(n)
            for neighbor in sorted(self.graph[n]):
                if neighbor not in visited:
                    find_neighbor(neighbor)
        find_neighbor(start)
        return visited
    
    def bfs(self, start: int) -> list[int]:
        """
        너비 우선 탐색 (BFS)
        큐를 사용하여 구현\n
        같은 레벨의 노드들을 방문하는 것이 포인트!

        1. 루트를 visited와 queue에 저장
        2. queue의 가장 왼쪽 값을 꺼내어 그 노드의 이웃 탐색
        3. 방문한 적 없는 노드는 visited와 queue에 저장
        4. 2,3 반복
        """
        visited = [start]
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in sorted(self.graph[node]):
                if neighbor not in visited:
                    visited.append(neighbor)
                    queue.append(neighbor)
        return visited
    
    def search_and_print(self, start: int) -> None:
        """
        DFS와 BFS 결과를 출력
        """
        dfs_result = self.dfs(start)
        bfs_result = self.bfs(start)
        
        print(' '.join(map(str, dfs_result)))
        print(' '.join(map(str, bfs_result)))
