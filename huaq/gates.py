import numpy as np
import cmath
from typing import List

from . import exceptions as ex

INVSQRT2 = 1 / cmath.sqrt(2)

# there is technically a method in numpy that does this but it really didn't like outputting
# a square matrix and always made rows of matrix vectors??? its nonsense
def tensor_product(A: np.matrix, B:np.matrix):
    tensor_product_flat = np.zeros((A.shape[0] * B.shape[0], A.shape[0] * B.shape[0]), complex) 

    # Compute the tensor product manually
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(B.shape[0]):
                for l in range(B.shape[1]):
                    tensor_product_flat[i * B.shape[0] + k][j * B.shape[1] + l] = A.item(i, j) * B.item(k, l)

    return tensor_product_flat

class Gate:
    def __init__(self, time: int, matrix, qubits: List[int], name: str):
        self.t = time
        self.mat = matrix
        self.qbts = qubits
        self.n = name
    
    # later functionality
    # def __add__(self, other):
    #     if other.t == self.t:
    #         if np.union1d(other.qbts, self.qbts) < len(other.qbts) + len(self.qbts):
    #             # uh oh something went fuckity wuckity, ops on overlapping gates
    #             # gates on the same qubits cannot be simultaneous
    #             raise ex.DoubleOperationException()
    #         else:
    #             pass


# single qubit X (not)
class X(Gate):
    def __init__(self, time: int, qubit: int):
        mat = np.matrix([[0, 1],
                         [1, 0]]).astype(complex)

        super().__init__(time, mat, [qubit], "X")

# single qubit Y
class Y(Gate):
    def __init__(self, time: int, qubit: int):
        mat = np.matrix([[0, -1j],
                         [1j, 0]]).astype(complex)

        super().__init__(time, mat, [qubit], "Y")

# single qubit Z
class Z(Gate):
    def __init__(self, time:int, qubit: int):
        mat = np.matrix([[1, 0],
                         [0, -1]]).astype(complex)

        super().__init__(time, mat, [qubit], "Z")

# single qubit H
class H(Gate):
    def __init__(self, time: int, qubit: int):
        mat = np.matrix([[INVSQRT2, INVSQRT2],
                         [INVSQRT2, -1 * INVSQRT2]]).astype(complex)

        super().__init__(time, mat, [qubit], "H")

# two qubit CNOT
class CNot(Gate):
    def __init__(self, time: int, qubits: tuple[int, int]):
        mat = np.matrix([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]]).astype(complex)

        super().__init__(time, mat, list(qubits), "CX")
