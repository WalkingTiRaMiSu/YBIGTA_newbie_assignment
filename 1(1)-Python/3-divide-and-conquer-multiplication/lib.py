from __future__ import annotations
import copy


"""
TODO:
- __setitem__ 구현하기
- __pow__ 구현하기 (__matmul__을 활용해봅시다)
- __repr__ 구현하기
"""


class Matrix:
    MOD = 1000

    def __init__(self, matrix: list[list[int]]) -> None:
        self.matrix = matrix

    @staticmethod
    def full(n: int, shape: tuple[int, int]) -> Matrix:
        '''
        shape를 n으로 채운 행렬 생성
        '''
        return Matrix([[n] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        '''
        shape를 0으로 채운 행렬 생성
        '''
        return Matrix.full(0, shape)

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        '''
        shape를 1로 채운 행렬 생성
        '''
        return Matrix.full(1, shape)

    @staticmethod
    def eye(n: int) -> Matrix:
        '''
        크기가 n인 단위행렬 생성
        '''
        matrix = Matrix.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1
        return matrix

    @property
    def shape(self) -> tuple[int, int]:
        '''
        행렬 크기 리턴
        '''
        return (len(self.matrix), len(self.matrix[0]))

    def clone(self) -> Matrix:
        '''
        행렬 복사본 생성
        '''
        return Matrix(copy.deepcopy(self.matrix))

    def __getitem__(self, key: tuple[int, int]) -> int:
        '''
        인덱스로 호출하기
        '''
        return self.matrix[key[0]][key[1]]

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        '''
        인덱스로 불러온 값 바꾸기
        '''
        self.matrix[key[0]][key[1]] = value % Matrix.MOD

    def __matmul__(self, matrix: Matrix) -> Matrix:
        '''
        행렬 곱 @ 정의
        '''
        x, m = self.shape
        m1, y = matrix.shape
        assert m == m1

        result = self.zeros((x, y))

        for i in range(x):
            for j in range(y):
                for k in range(m):
                    result[i, j] += self[i, k] * matrix[k, j]

        return result

    def __pow__(self, n: int) -> Matrix:
        '''
        행렬 제곱(matrix**n) 정의하기\n
        분할 정복 사용
        '''
        result = Matrix.eye(self.shape[0])
        origin = self.clone()

        while n > 0:
            if n % 2 == 1:
                result = result @ origin
            origin = origin @ origin
            n //= 2

        return result


    def __repr__(self) -> str:
        '''
        행렬 출력
        '''
        rows = [" ".join(str(cell % self.MOD) for cell in row) for row in self.matrix]
        return "\n".join(rows)