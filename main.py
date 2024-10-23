import numpy as np

from huaq.gates import H, X, Z, CNot, eXX
from huaq.circuit import Circuit, Sweep

c = Circuit(3)
c.add_variable('theta')

gates = [
    eXX(1, (0, 1), c.get_variable('theta')),

    X(2, 0),
    Z(2, 1),
]

c.add_gates(gates)

result = c.run(log=True, theta=Sweep(0, 2*np.pi, 100))
result.plot()
