import numpy as np

from .gates import *
from .exceptions import *
from .visualization import *
from .types import *
from .utils import *

def debug_log(string: str, log: bool = False):
    if log:
        print(string)

# compile a list of gates sorted by time and subsorted by qubit into a list of single gates at different times
def compile_gates(gates: List[Gate], qubits: int, log: bool = False) -> List[Gate]:
    program = []

    i = 0
    t = -1
    sim = []
    while i < len(gates):
        if gates[i].t != t:
            if len(sim) > 0:
                program.append(compile_simultaneous_gates(sim, qubits, log=log))
                sim = []
            t = gates[i].t
        sim.append(gates[i])
        i += 1

    if log:
        print("---")
        print(f"compilation complete with i={i}")
        for gate in gates:
            print(gate)
    return gates

# this function assumes that all the gates are in the same time, a la the timeslice struct
def compile_simultaneous_gates(gs: List[Gate], qubits: int, log: bool = False) -> Gate:
    i = 0
    completed = False
    gates = gs.copy()
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
                break
        else:
            completed = True 

    return gates[-1]

# this class is a wrapper for a list of gates that happen at the same time
class TimeSlice:
    def __init__(self, t: int, gates: List[Gate], precomputed: bool = False):
        self.t = t
        self.gates = gates
        self.precomputed = precomputed

        self.dependencies = []

        for gate in gates:
            if not gate.precomputed:
                for var in gate.vars:
                    self.dependencies.append(var)

        self.unitary = None

    def append(self, gate: Gate):
        self.gates.append(gate)
        
        if not gate.precomputed:
            self.precomputed = False
            for var in gate.vars:
                self.dependencies.append(var)

        self._sort_gates()

    def _sort_gates(self):
        gs = sorted(self.gates, key=lambda x: x.t)
        self.gates = []

        for i in range(gs[-1].t + 1):
            simultaneous_gates = [gate for gate in gs if gate.t == i]
            self.gates += sorted(simultaneous_gates, key=lambda x: min(x.qbts))


    def compile(self, qubits: int, log: bool = False, **kwargs):
        postcomputed_gates = []

        # this function allows the injection of postcomputed gates into the list of gates and practically means that gates that don't rely
        # on a parameter can be computed once and never again throughout the course of the program
        if not self.precomputed:
            debug_log(f"this is not precomputed need to inject {self.dependencies}", log)
            if not all([var.name in kwargs for var in self.dependencies]):
                raise MissingVariable()
            
            for gate in self.gates:
                if not gate.precomputed:
                    postcomputed_gates.append(gate.post_compute(kwargs[gate.vars[0].name], ret=True))
                    debug_log(f"post computed {gate} with v={kwargs[gate.vars[0].name]}", log)
                else:
                    postcomputed_gates.append(gate)
        else:
            postcomputed_gates = self.gates.copy()

        self.unitary = compile_simultaneous_gates(postcomputed_gates, qubits, log=log)

        return self

    def __str__(self):
        return f"TimeSlice({self.t}, {g.__str__() for g in self.gates})"

class Circuit: 
    def __init__(self, qubits: int):
        self.qubits = list(range(qubits))
        self.gates = []
        self.variables = []

        self.precomputed = True
        self.compiled = False
        self.compiled_gates = []
    
    # since the gates are already sorted by qubit in the TimeSlice class, we can just sort by time
    def _sort_gates(self):
        self.gates = sorted(self.gates, key=lambda x: x.t)

    def add_gate(self, gate: Gate):
        if len(np.union1d(gate.qbts, self.qubits)) != len(self.qubits):
            raise QubitOutOfRange() # if there isn't a complete overlap of the qubits there is a weird qubit somewhere
        
        t = gate.t
        if not any([t == ts.t for ts in self.gates]):
            # if there isn't a timeslice at time t, create one
            self.gates.append(TimeSlice(t, [gate], precomputed=gate.precomputed))
        else:
            # if there is a timeslice at time t, append the gate to the list of gates at time t
            for ts in self.gates:
                if ts.t == t:
                    ts.append(gate)
                    if not gate.precomputed:
                        ts.precomputed = False
                    break

        self._sort_gates()
    
    # getters and setters etc
    def add_gates(self, gates: List[Gate]):
        for gate in gates:
            self.add_gate(gate)

    def add_fixed_variable(self, name: str, value: int = 0):
        self.variables.append(Var(name, value))

    def add_variable(self, name: str):
        self.variables.append(Var(name, None, precomputed=False))
    
    def get_variable(self, name: str) -> Var:
        for var in self.variables:
            if var.name == name:
                return var
        raise MissingVariable()

    # this function runs the circuit a single time with parameters as floats
    def singleton(self, log: bool = False, **kwargs):
        # since compiled gates are already ordered in time, all we'd need to do is take tensor products of simultaneous gates
        # and then multiply the matrices together

        state = np.zeros(2**len(self.qubits))
        state[0] = 1 # |0> state for all qubits

        program = []

        for ts in self.compiled_gates:
            if not ts.precomputed: # if the gates depend on a variable, inject the variable and compute the tensor product
                debug_log(f"compiling {ts} with v={kwargs[ts.dependencies[0].name]}", log)
                ts.compile(len(self.qubits), log=log, **kwargs)
            program.append(ts.unitary)

        debug_log(f"running program {program}", log)
        
        for gate in program:
            state = np.dot(state, gate.mat)

        return ProbabilityDensity(state, kwargs)
    
    def run(self, log: bool = False, **kwargs):
        ranges = [] 
        iterations = 1
        variable_state = {}
        
        if not self.compiled:
            for ts in self.gates:
                if ts.precomputed:
                    # if the gates are static/don't depend on a variable, compile them once and never again
                    self.compiled_gates.append(ts.compile(len(self.qubits), log=log, **kwargs))
                else:
                    # else we need to deal with them on every singleton :D
                    self.compiled_gates.append(ts)

            self.compiled = True

        for k, v in kwargs.items():
            if isinstance(v, Range):
                iterations *= len(v.space)
                v.set_name(k)
                ranges.append(v)
            else:
                variable_state[k] = v

        results = Result([], list(kwargs.keys()))
        
        # take all possible combinations of the ranges (sweeps or whatever) and run the circuit with those values
        possible_range_values = cartesian_product(*[r.space for r in ranges])
        for i in range(iterations):
            for idx, s in enumerate(possible_range_values[i]):
                variable_state[ranges[idx].name] = s
                debug_log(f"running with {variable_state}", log)

            # use singleton to run the circuit with the values as floats
            results.append(self.singleton(log=log, **variable_state))

        return results
