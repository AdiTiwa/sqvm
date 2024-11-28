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

    def append(self, gate: Gate):
        self.gates.append(gate)
        self.sort()

    def __str__(self):
        return f"TimeSlice({self.t}, {g.__str__() for g in self.gates})"

class Circuit:
    def __init__(self, qubits):
        self.num_qubits = qubits

        self.program: List[TimeSlice] = []

    def add_gate(self, gate: Gate):
        t = gate.t
        if not any([t == ts.time for ts in self.program]):
            self.program.append(TimeSlice(t, [gate]))
        else:
            for ts in self.program:
                if ts.time == t:
                    ts.append(gate)
                    break

    def add_gates(self, gates: List[Gate]):
        for gate in gates:
            self.add_gate(gate)

    # TODO: Variables
    def run(self):
        self.qubits = [[1, 0] for _ in range(self.num_qubits)]
        self.entangled = [[] for _ in range(self.num_qubits)]

        for t in self.program:
            queue = t.gates

            while queue != []:
                g = queue.pop(0)

                if any([self.entangled[g.qbts[i]] != [] for i in range(len(g.qbts))]):
                    # entangled gate, tensor product all gates that involve entangled qubits and then combine and dot
                    from functools import reduce

                    entangled_qubits = reduce(np.union1d, (*[self.entangled[g.qbts[i]] for i in range(len(g.qbts))], g.qbts)) # holy mother of one-liners

                    biggate: Gate = g
                    for gother in queue:
                        if any([q in entangled_qubits for q in gother.qbts]):
                            biggate = gate_kron(biggate, gother, t.time) # tensor product all gates that involve entangled qubits
                            queue.remove(gother)
                    
                    qubits_not_used = np.setdiff1d(entangled_qubits, biggate.qbts)
                    while qubits_not_used != []:
                        # if the big gate doesn't involve all entangled qubits, pad with identity gates
                        biggate = id_gate_kron(biggate, qubits_not_used[0], t.time)
                        qubits_not_used = np.setdiff1d(entangled_qubits, biggate.qbts)

                    state = np.concatenate((*[self.qubits[biggate.qbts[i]] for i in range(len(biggate.qbts))],))

                    state = np.dot(state, biggate.mat)

                    for q in biggate.qbts:
                        self.qubits[q] = state[:2]
                        state = state[2:]

                    if biggate.entangling:
                        for q in biggate.qbts:
                            self.entangled[q] = np.union1d(self.entangled[q], [q for q in biggate.qbts if q != q])

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
                            self.entangled[q] = np.union1d(self.entangled[q], [q for q in g.qbts if q != q])
                else:
                    # not entangled or multi-qubit gate, implies single qubit gate?
                    self.qubits[g.qbts[0]] = np.dot(self.qubits[g.qbts[0]], g.mat)
