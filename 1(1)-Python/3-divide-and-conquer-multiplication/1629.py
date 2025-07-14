# lib.py의 Matrix 클래스를 참조하지 않음
import sys


"""
TODO:
- fast_power 구현하기 
"""


def fast_power(base: int, exp: int, mod: int) -> int:
    """
    빠른 거듭제곱 알고리즘 구현
    분할 정복을 이용, 시간복잡도 고민!
    아래의 논리를 재귀적으로 적용
    1. 홀수일 경우 제곱 형태를 취한 후 base 한 번 더 곱함 그 후 나머지
    2. 짝수일 경우 제곱 형태를 취한 후 나머지
    """
    if exp == 1:
        return base % mod
    else:
        temp = fast_power(base, exp//2, mod)
        if exp % 2 == 0:
            return (temp * temp) % mod
        else:
            return (temp * temp * base) % mod

def main() -> None:
    A: int
    B: int
    C: int
    A, B, C = map(int, input().split()) # 입력 고정
    
    result: int = fast_power(A, B, C) # 출력 형식
    print(result) 

if __name__ == "__main__":
    main()
