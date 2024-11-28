import numpy as np
import cmath
from typing import List, Union
from deprecated import deprecated

from .exceptions import *
from .types import Var

INVSQRT2 = 1 / cmath.sqrt(2)

@deprecated("use tensor_product instead")
def tensor_prod(A: np.matrix, B:np.matrix):
    tensor_product_flat = np.zeros((A.shape[0] * B.shape[0], A.shape[0] * B.shape[0]), complex) 

    # Compute the tensor product manually
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(B.shape[0]):
                for l in range(B.shape[1]):
                    tensor_product_flat[i * B.shape[0] + k][j * B.shape[1] + l] = A.item(i, j) * B.item(k, l)

    return tensor_product_flat

def tensor_product(A: np.matrix, B: np.matrix):
    prod = np.zeros((A.shape[0] * B.shape[0], A.shape[1] * B.shape[1]), complex)

    for i in range(A.shape[0] * B.shape[0]):
        for j in range(A.shape[1] * B.shape[1]):
            prod[i, j] = A.item(i // B.shape[0], j // B.shape[1]) * B.item(i % B.shape[0], j % B.shape[1])

    return prod

# general gate class
class Gate:
    def __init__(self, time: int, matrix, qubits: List[int], name: str):
        self.t = time
        self.mat = matrix
        self.qbts = qubits
        self.n = name

        self.precomputed = True
        self.entangling = False

    def __init_postcomputed__(self, time: int, qubits: List[int], name: str, variables: List[Var]):
        # if it is not precomputed, it will be related to a Var object, needs a postcompute method
        self.t = time
        self.qbts = qubits
        self.n = name + "(" + ", ".join([v.name for v in variables]) + ")"
        self.vars = variables
        
        self.precomputed = False

    def post_compute(self, value, ret: bool = False):
        raise NotImplementedError()

    def __str__(self):
        return f"{self.n}(t={self.t}, qubits={self.qbts})"
    
    def __add__(self, other): # literally gate_kron
        if self.t == other.t:
            if len(np.union1d(self.qbts, other.qbts)) < len(self.qbts) + len(other.qbts):
                # same time, overlapping qubits
                raise DoubleOperationException()
            elif abs(min(other.qbts) - max(self.qbts)) == 1:
                # you can just take the tensor product of the two gates
                return gate_kron(self, other, self.t)
            elif abs(min(other.qbts) - max(self.qbts)) != 1:
                lower_gate = self if max(self.qbts) < min(other.qbts) else other

                while True:
                    if min(other.qbts) - max(lower_gate.qbts) == 1:
                        return gate_kron(self, other, self.t)
                    else:
                        lower_gate = gate_kron(lower_gate, I(lower_gate.t, min(lower_gate.qbts) + 1), lower_gate.t)

# tensor product of two gates, kronecker product of their matrices
def gate_kron(g1: Gate, g2: Gate, t):
    g = Gate(t, tensor_product(g1.mat, g2.mat), g1.qbts + g2.qbts, f"{g1.n} {g2.n}")
    if g1.entangling or g2.entangling:
        g.entangling = True
    return g

@deprecated("use id_gate_kron instead")
def identity_gate_kron(g1: Gate, below: bool, t):
    # tensor product is easier to compute by just duplication the matrix for the corners of a larger square matrix

    mat = np.matrix(np.zeros((2 * g1.mat.shape[0], 2 * g1.mat.shape[1]), complex))

    for i in range(g1.mat.shape[0]):
        for j in range(g1.mat.shape[1]):
            mat[i, j] = g1.mat.item(i, j)
            mat[i + g1.mat.shape[0], j + g1.mat.shape[1]] = g1.mat.item(i, j)

    if below:
        return Gate(t, mat, g1.qbts + [max(g1.qbts) + 1], f"{g1.n} I")
    else:
        return Gate(t, mat, g1.qbts + [min(g1.qbts) - 1], f"I {g1.n}")

def id_gate_kron(g1: Gate, qubit: int, t):
    mat = np.matrix(np.zeros((2 * g1.mat.shape[0], 2 * g1.mat.shape[1]), complex))

    for i in range(g1.mat.shape[0]):
        for j in range(g1.mat.shape[1]):
            mat[i, j] = g1.mat.item(i, j)
            mat[i + g1.mat.shape[0], j + g1.mat.shape[1]] = g1.mat.item(i, j)

    g = Gate(t, mat, g1.qbts + [qubit], f"{g1.n} I")
    if g1.entangling:
        g.entangling = True
    return g

class I(Gate):
    def __init__(self, time: int, qubit: int):
        mat = np.matrix([[1, 0],
                         [0, 1]]).astype(complex)

        super().__init__(time, mat, [qubit], "I")

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

# single qubit rotation around x
class Rx(Gate):
    def __init__(self, time: int, qubit: int, angle: Union[Var, float]):
        if isinstance(angle, int) or angle.precomputed:
            a = angle if isinstance(angle, float) else angle.value
            mat = np.matrix([[cmath.cos(a / 2), -1j * cmath.sin(a / 2)],
                             [-1j * cmath.sin(a / 2), cmath.cos(a / 2)]]).astype(complex)

            super().__init__(time, mat, [qubit], f"Rx({angle})")
        else:
            super().__init_postcomputed__(time, [qubit], "Rx", [angle])

    def post_compute(self, value: float, ret: bool = False):
        mat = np.matrix([[cmath.cos(value / 2), -1j * cmath.sin(value / 2)],
                              [-1j * cmath.sin(value / 2), cmath.cos(value / 2)]]).astype(complex)

        if ret:
            return Gate(self.t, mat, self.qbts, self.n)
        else:
            self.mat = mat
# single qubit rotation around y
class Ry(Gate):
    def __init__(self, time: int, qubit: int, angle: Union[Var, float]):
        if isinstance(angle, int) or angle.precomputed:
            a = angle if isinstance(angle, float) else angle.value
            mat = np.matrix([[cmath.cos(a / 2), -1 * cmath.sin(a / 2)],
                             [cmath.sin(a / 2), cmath.cos(a / 2)]]).astype(complex)

            super().__init__(time, mat, [qubit], f"Ry({angle})")
        else:
            super().__init_postcomputed__(time, [qubit], "Ry", [angle])

    def post_compute(self, value: float, ret: bool = False):
        mat = np.matrix([[cmath.cos(value / 2), -1 * cmath.sin(value / 2)],
                              [cmath.sin(value / 2), cmath.cos(value / 2)]]).astype(complex)

        if ret:
            return Gate(self.t, mat, self.qbts, self.n)
        else:
            self.mat = mat

# single qubit rotation around z
class Rz(Gate):
    def __init__(self, time: int, qubit: int, angle: Union[Var, float]):
        if isinstance(angle, int) or angle.precomputed:
            a = angle if isinstance(angle, float) else angle.value
            nege = cmath.exp(-1j * a / 2)
            pose = cmath.exp(1j * a / 2)

            mat = np.matrix([[nege, 0],
                             [0, pose]]).astype(complex)
            
            super().__init__(time, mat, [qubit], f"Rz({angle})")
        else:
            super().__init_postcomputed__(time, [qubit], "Rz", [angle])

    def post_compute(self, value: float, ret: bool = False):
        nege = cmath.exp(-1j * value / 2)
        pose = cmath.exp(1j * value / 2)

        mat = np.matrix([[nege, 0],
                             [0, pose]]).astype(complex)

        if ret:
            return Gate(self.t, mat, self.qbts, self.n)
        else:
            self.mat = mat

# single qubit S
class S(Gate):
    def __init__(self, time: int, qubit: int):
        mat = np.matrix([[1, 0],
                         [0, 1j]]).astype(complex)
        super().__init__(time, mat, [qubit], "H")

# square root of an X/NOT gate
class sqX(Gate):
    def __init__(self, time: int, qubit: int):
        mat = np.matrix([[1, -1j],
                         [-1j, 1]]).astype(complex)
        super().__init__(time, mat, [qubit], "sqX")

# two qubit CNOT
class CNot(Gate):
    def __init__(self, time: int, qubits: tuple[int, int]):
        mat = np.matrix([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]]).astype(complex)

        super().__init__(time, mat, list(qubits), "CX")
        
        self.entangling = True

# two qubit CZ
class CZ(Gate):
    def __init__(self, time: int, qubits: tuple[int, int]):
        mat = np.matrix([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, -1]]).astype(complex)

        super().__init__(time, mat, list(qubits), "CZ")

        self.entangling = True

# two qubit SWAP
class Swap(Gate):
    def __init__(self, time: int, qubits: tuple[int, int]):
        mat = np.matrix([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]]).astype(complex)

        super().__init__(time, mat, list(qubits), "SWAP")

# two qubit XX
class XX(Gate):
    def __init__(self, time: int, qubits: tuple[int, int]):
        mat = np.matrix([[0, 0, 0, 1],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [1, 0, 0, 0]]).astype(complex)

        super().__init__(time, mat, list(qubits), "XX")

# two qubit exponentiated XX
class eXX(Gate):
    def __init__(self, time:int, qubits: tuple[int, int], exponent: Union[Var, float]):
        if isinstance(exponent, int) or exponent.precomputed:
            exp = exponent if isinstance(exponent, float) else exponent.value
            f = cmath.exp(1j * exp * np.pi / 2)
            c = f * cmath.cos(exp * np.pi / 2)
            s = -1j * f * cmath.sin(exp * np.pi / 2)
            mat = np.matrix([[c, 0, 0, s],
                             [0, c, s, 0],
                             [0, s, c, 0],
                             [s, 0, 0, c]]). astype(complex)

            super().__init__(time, mat, list(qubits), f"eXX({exponent})")
        else:
            super().__init_postcomputed__(time, list(qubits), "eXX", [exponent])

        self.entangling = True

    def post_compute(self, value: float, ret: bool = False):
        f = cmath.exp(1j * value * np.pi / 2)
        c = f * cmath.cos(value * np.pi / 2)
        s = -1j * f * cmath.sin(value * np.pi / 2)
        mat = np.matrix([[c, 0, 0, s],
                             [0, c, s, 0],
                             [0, s, c, 0],
                             [s, 0, 0, c]]).astype(complex)

        if ret:
            return Gate(self.t, mat, self.qbts, self.n)
        else:
            self.mat = mat

        
