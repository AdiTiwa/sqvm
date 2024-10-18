from huaq.gates import H, X, Z, CNot, eXX
from huaq.circuit import Circuit, Sweep

c = Circuit(2)
c.add_variable("theta")

gates = [
    H(0, 0),

    eXX(1, (0, 1), c.get_variable("theta")),

    Z(2, 0),
    X(2, 1),
]

c.add_gates(gates)

print(c.run(theta=Sweep(0, 2, 10)))
