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

class Sweep:
    def __init__(self, start: float, end: float, steps: int):
        self.start = start
        self.end = end
        self.steps = steps

        self.current = start
        self.step = (end - start) / steps
        self.name: str = ''

    def next(self):
        self.current += self.step
        if self.current >= self.end:
            self.reset()
        return self.current
    
    def reset(self):
        self.current = self.start

class Circuit: 
    def __init__(self, qubits: int):
        self.qubits = list(range(qubits))
        self.gates = []
        self.variables = []

        self.precomputed = True

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

        if not gate.precomputed:
            self.precomputed = False

        self._sort_gates()

    def add_gates(self, gates: List[Gate]):
        for gate in gates:
            self.add_gate(gate)

    def add_fixed_variable(self, name: str, value: int = 0):
        self.variables.append(Var(name, value))

    def add_variable(self, name: str):
        v = Var(name, None)
        self.variables.append(Var(name, None, precomputed=False))
    
    def get_variable(self, name: str) -> Var:
        for var in self.variables:
            if var.name == name:
                return var
        raise MissingVariable()

    def singleton(self, prob: bool = True, log: bool = False, **kwargs):
        # since compiled gates are already ordered in time, all we'd need to do is take tensor products of simultaneous gates
        # and then multiply the matrices together

        state = np.zeros(2**len(self.qubits))
        state[0] = 1 # |0> state for all qubits

        if not self.precomputed:
            for var in self.variables:
                if var.name not in kwargs and (var.precomputed is False or var.value is None):
                    raise MissingVariable()
                var.value = kwargs[var.name]

            for gate in self.gates:
                if not gate.precomputed:
                    gate.post_compute()

        program = compile_gates(self.gates, len(self.qubits), log=log) 
        for gate in program:
            state = np.dot(state, gate.mat)

        if prob:
            return np.abs(np.asarray(state).reshape(-1)) ** 2 # take the absolute value sqauared for probablity density
        return state
    
    def run(self, prob:bool = True, log: bool = False, **kwargs):
        sweeps = [] 
        iterations = 1
        variable_state = {}

        for k, v in kwargs.items():
            if isinstance(v, Sweep):
                iterations *= v.steps
                v.name = k
                sweeps.append(v)
            else:
                variable_state[k] = v

        results = np.zeros((iterations, 2**len(self.qubits)))
        
        for i in range(iterations):
            for sweep in sweeps:
                variable_state[sweep.name] = sweep.next()

            results[i] = self.singleton(prob=prob, log=log, **variable_state)

        return results
