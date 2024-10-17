import numpy as np

from huaq.circuit import Circuit
from huaq.gates import H

circuit = Circuit(1)

circuit.add_gate(H(0, 0))

print(circuit.run())
