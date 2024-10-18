class DoubleOperationException(Exception):
    "Calling two gates simultaneously on the same qubit is not possible."
    pass

class QubitOutOfRange(Exception):
    "Qubit index out of range. Check if the gate is within the qubit range of the circuit."
    pass

class MissingVariable(Exception):
    "Variable not found provided in the run method or not initialized."
    pass
