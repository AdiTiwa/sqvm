import numpy as np

from huaq.gates import H, X, Z, CNot, eXX
from huaq.circuit import Circuit

c = Circuit(2)

c.add_gates([H(0, 0), H(1, 1), CNot(1, (0, 1)), H(2, 0), H(2, 1)])

c.run()
