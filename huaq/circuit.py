import numpy as np

from .gates import *
from .exceptions import *
from .visualization import *
from .types import *
from .utils import *

def debug_log(string: str, log: bool = False):
    if log:
        print(string)

def compile_gates(gates: List[Gate], qubits: int, log: bool = False) -> List[Gate]:
    i = 0
    completed = False
    while i < len(gates):
        if i < len(gates) - 1 and gates[i].t == gates[i + 1].t:
            completed = False
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

                gates.insert(i, identity_gate_kron(g1, True, g1.t))
                debug_log(f"merged {g1}, I, i={i}", log)
        elif completed:
            if min(gates[i].qbts) != 0:
                g1 = gates.pop(i)
                gates.insert(i, identity_gate_kron(g1, False, g1.t)) # pad down to qubit 0
                debug_log(f"merged I, {g1}, i={i}", log)
            elif max(gates[i].qbts) != qubits - 1:
                g1 = gates.pop(i)
                gates.insert(i, identity_gate_kron(g1, True, g1.t)) # pad up to qubit qubits - 1
                debug_log(f"merged {g1}, I, i={i}", log)
            else:
                debug_log(f"completed gate {gates[i]}, t={i}", log)
                i += 1
        else:
            completed = True

    if log:
        print("---")
        print(f"compilation complete with i={i}")
        for gate in gates:
            print(gate)
    return gates

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

    def singleton(self, log: bool = False, **kwargs):
        # since compiled gates are already ordered in time, all we'd need to do is take tensor products of simultaneous gates
        # and then multiply the matrices together

        state = np.zeros(2**len(self.qubits))
        state[0] = 1 # |0> state for all qubits

        if not self.precomputed:
            for var in self.variables:
                if var.name not in kwargs and (var.precomputed is False or var.value is None):
                    raise MissingVariable()

            for gate in self.gates:
                if not gate.precomputed:
                    gate.post_compute(kwargs[gate.vars[0].name])
                    debug_log(f"post computed {gate} with v={kwargs[gate.vars[0].name]}", log)

        program = compile_gates(self.gates.copy(), len(self.qubits), log=log)
        
        for gate in program:
            state = np.dot(state, gate.mat)

        return ProbabilityDensity(state, kwargs)
    
    def run(self, log: bool = False, **kwargs):
        ranges = [] 
        iterations = 1
        variable_state = {}

        for k, v in kwargs.items():
            if isinstance(v, Range):
                iterations *= len(v.space)
                v.set_name(k)
                ranges.append(v)
            else:
                variable_state[k] = v

        results = Result([], list(kwargs.keys()))
        
        possible_range_values = cartesian_product(*[r.space for r in ranges])
        for i in range(iterations):
            for idx, s in enumerate(possible_range_values[i]):
                variable_state[ranges[idx].name] = s
                debug_log(f"running with {variable_state}", log)

            results.append(self.singleton(log=log, **variable_state))

        return results
