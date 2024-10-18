from huaq.gates import H, X, Z, CNot
from huaq.circuit import Circuit

c = Circuit(2)

gates = [
    H(0, 0),
    H(0, 1),

    CNot(1, (0, 1)),

    Z(2, 0),
    X(2, 1),
]

c.add_gates(gates)

print(c.run())
