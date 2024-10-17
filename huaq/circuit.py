import numpy as np

from .gates import *
from .exceptions import *

def compile_gates(gates: List[Gate], i = 0) -> List[Gate]:
    # this function will take a list of gates and compile them into a list of gates that are non-overlapping
    if i == len(gates) - 1:
        return gates

    if gates[i].t == gates[i + 1].t:
        if np.union1d(gates[i].qbts, gates[i + 1].qbts) < len(gates[i].qbts) + len(gates[i + 1].qbts):
            # gates overlapping qubits
            raise DoubleOperationException()
        else:
            gates[i].mat = tensor_product(gates[i].mat, gates[i + 1].mat)
            gates[i].qbts = np.union1d(gates[i].qbts, gates[i + 1].qbts)
            gates.pop(i + 1)
            return compile_gates(gates, i)

class Circuit: 
    def __init__(self, qubits: int):
        self.qubits = list(range(qubits))
        self.gates = []

    def add_gate(self, gate: Gate):
        if len(np.union1d(gate.qbts, self.qubits)) != len(self.qubits):
            raise QubitOutOfRange() # if there isn't a complete overlap of the qubits there is a weird qubits somewhere
        self.gates.append(gate)
        self.gates = sorted(self.gates, key=lambda x: x.t)

    def run(self):
        # since the gates are already ordered in time, all we'd need to do is take tensor products of simultaneous gates
        # and then multiply the matrices together

        initial_state = np.zeros(2**len(self.qubits))
        initial_state[0] = 1 # |0> state for all qubits

        program = compile_gates(self.gates)
        for gate in program:
            initial_state = np.dot(gate.mat, initial_state)

        return initial_state
