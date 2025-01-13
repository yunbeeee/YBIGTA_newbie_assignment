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
        return Matrix([[n] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(0, shape)

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(1, shape)

    @staticmethod
    def eye(n: int) -> Matrix:
        matrix = Matrix.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1
        return matrix

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.matrix), len(self.matrix[0]))

    def clone(self) -> Matrix:
        return Matrix(copy.deepcopy(self.matrix))

    def __getitem__(self, key: tuple[int, int]) -> int:
        return self.matrix[key[0]][key[1]]

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        # 구현하세요!
        """
        행렬의 특정 위치에 값 설정.

        Args:
            key (tuple[int, int]): 값을 설정할 위치(행, 열)
            value (int): 설정할 값(MOD로 나눈 나머지)

        Returns:
            None
        """
        row, col = key
        self.matrix[row][col] = value % self.MOD

    def __matmul__(self, matrix: Matrix) -> Matrix:
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
        # 구현하세요!
        """
        행렬 거듭제곱 연산 과정.

        Args:
            n (int): 거듭제곱의 지수.

        Returns:
            Matrix: 거듭제곱 결과 행렬.
        """
        if n == 0:
            return Matrix.eye(self.shape[0]) 
        if n == 1:
            result = self
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i, j] %= self.MOD
            return result
        
        half = self ** (n // 2)
        result = half @ half
        if n % 2 == 1:
            result = result @ self
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] %= self.MOD
        return result


    def __repr__(self) -> str:
        # 구현하세요!
        """
        행렬의 내용을 문자열로 보기 좋게 표현.

        Returns:
            str: 행렬의 각 행을 문자열로 변환하여 '\n'로 연결한 결과.
        """
        line = [' '.join(map(str, row)) for row in self.matrix]
        return '\n'.join(line)