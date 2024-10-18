import numpy as np

from .gates import *
from .exceptions import *

def debug_log(string: str, log: bool = False):
    if log:
        print(string)

def compile_gates(gates: List[Gate], qubits: int, log: bool = False) -> List[Gate]:
    i = 0
    while i < len(gates) - 1:
        if gates[i].t == gates[i + 1].t:
            if len(np.union1d(gates[i].qbts, gates[i + 1].qbts)) < len(gates[i].qbts) + len(gates[i + 1].qbts):
                # same time, overlapping qubits
                raise DoubleOperationException()
            elif min(gates[i + 1].qbts) - max(gates[i].qbts) == 1:
                # you can just take the tensor product of the two gates
                g1 = gates.pop(i)
                g2 = gates.pop(i)

                gates.insert(i, gate_kron(g1, g2, g1.t))
                debug_log(f"merged {g1}, {g2}, i={i}", log)
            elif min(gates[i + 1].qbts) - max(gates[i].qbts) != 1:
                # distance is greater than 1, need to insert identity gate
                g1 = gates.pop(i)

                gates.insert(i, gate_kron(g1, I(g1.t, min(g1.qbts) + 1), g1.t))
                debug_log(f"merged {g1}, I, i={i}", log)
        else:
            if min(gates[i].qbts) != 0:
                g1 = gates.pop(i)
                gates.insert(i, gate_kron(I(g1.t, 0), g1, g1.t)) # pad down to qubit 0
                debug_log(f"merged I, {g1}, i={i}", log)
            elif max(gates[i].qbts) != qubits - 1:
                g1 = gates.pop(i)
                gates.insert(i, gate_kron(g1, I(g1.t, qubits - 1), g1.t)) # pad up to qubit qubits - 1
                debug_log(f"merged {g1}, I, i={i}", log)
            else:
                i += 1
    return gates

class Circuit: 
    def __init__(self, qubits: int):
        self.qubits = list(range(qubits))
        self.gates = []

    def _sort_gates(self):
        gs = sorted(self.gates, key=lambda x: x.t)
        self.gates = []

        for i in range(gs[-1].t + 1):
            simultaneous_gates = [gate for gate in gs if gate.t == i]
            self.gates += sorted(simultaneous_gates, key=lambda x: min(x.qbts))

    def add_gate(self, gate: Gate):
        if len(np.union1d(gate.qbts, self.qubits)) != len(self.qubits):
            raise QubitOutOfRange() # if there isn't a complete overlap of the qubits there is a weird qubit somewhere
        self.gates.append(gate)
        self._sort_gates()

    def add_gates(self, gates: List[Gate]):
        for gate in gates:
            self.add_gate(gate) 

    def run(self, prob: bool = True, log: bool = False):
        # since compiled gates are already ordered in time, all we'd need to do is take tensor products of simultaneous gates
        # and then multiply the matrices together

        state = np.zeros(2**len(self.qubits))
        state[0] = 1 # |0> state for all qubits

        program = compile_gates(self.gates, len(self.qubits), log=log) 
        for gate in program:
            state = np.dot(state, gate.mat)

        if prob:
            return np.abs(np.asarray(state).reshape(-1)) ** 2 # take the absolute value sqauared for probablity density
        return state
