import numpy as np

from typing import List

from .gates import *
from .exceptions import *
from .visualization import *
from .types import *
from .utils import *

class TimeSlice:
    def __init__(self, time: int, gates: List[Gate]):
        self.time = time
        self.gates = gates

    def sort(self):
        self.gates = sorted(self.gates, key=lambda x: x.qbts[0])

class Circuit:
    def __init__(self, qubits):
        self.qubits = [[1, 0] for _ in range(qubits)]
        self.entangled = [[] for _ in range(qubits)]

        self.program: List[TimeSlice] = []

    # TODO: Variables
    def run(self):
        for t in self.program: 
            for g in t.gates:
                if any([self.entangled[g.qbts[i]] != [] for i in range(len(g.qbts))]):
                    # entangled gate, tensor product all gates that involve entangled qubits and then combine and dot
                    from functools import reduce

                    entangled_qubits = reduce(np.union1d, (*[self.entangled[g.qbts[i]] for i in range(len(g.qbts))], g.qbts)) # holy mother of one-liners


                elif len(g.qbts) != 1:
                    # multi-qubit gate, combine states and then dot
                    state = np.concatenate((*[self.qubits[g.qbts[i]] for i in range(len(g.qbts))],))
                    state = np.dot(state, g.mat)

                    # take apart state vector into individual qubits
                    for q in g.qbts:
                        self.qubits[q] = state[:2]
                        state = state[2:]

                    if g.entangling:
                        for q in g.qbts:
                            self.entangled[q] = [q for q in g.qbts if q != q]
                else:
                    # not entangled or multi-qubit gate, implies single qubit gate?
                    self.qubits[g.qbts[0]] = np.dot(self.qubits[g.qbts[0]], g.mat)
