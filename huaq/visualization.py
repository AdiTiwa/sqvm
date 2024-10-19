import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import math as m

from typing import Union

from .types import *

DISPLAY_PRECISION = 4
PERCENT_PRECISION = 2
AS_PERCENT = False

# immutable class to represent a result of a quantum circuit, recording the state and the variables
class ProbabilityDensity:
    def __init__(self, state, variables: dict[str, float] = {}):
        self.state = state
        self.prob = np.abs(np.asarray(state).reshape(-1)) ** 2 # take the absolute value sqauared for probablity density

        self.output_range = []
        for i in range(len(self.prob)): # generate the output range of qubits, counting in binary lmao
            self.output_range.append(f'|{bin(i)[2:].zfill(int(m.log(len(self.prob),2)))}>')

        self.variables = variables

    def simulate(self, iterations: int):
        output = []
        for _ in range(iterations):
            output.append(np.random.choice(self.output_range, p=self.prob))
        return output

    def plot(self):
        fig, ax = plt.subplots() 

        ax.bar(self.output_range, self.prob)

        plt.show()

    def variable(self, name: str):
        return self.variables[name]

    def __str__(self): # nasty string formatting
        variables = ", ".join([f"{k}={round(v, DISPLAY_PRECISION)}" for k, v in self.variables.items()]) 
        output = ", ".join(f"{o}:{round(self.prob[idx], DISPLAY_PRECISION)}" for idx, o in enumerate(self.output_range))
        if AS_PERCENT:
            output = ", ".join(f"{o}:{round(self.prob[idx] * 100, PERCENT_PRECISION)}%" for idx, o in enumerate(self.output_range))
        return f"for {variables}, {output})"

    def __repr__(self):
        return self.__str__()

class Result:
    def __init__(self, probabilities: list[ProbabilityDensity], variables: list[Union[str, Sweep]]):
        self.probabilities = probabilities
        self.variables = variables

    def where(self, **kwargs):
        for prob in self.probabilities:
            if all([prob.variable(k) == v for k, v in kwargs.items()]):
                return prob
        raise ValueError("No results found for these variables")

    def append(self, prob: ProbabilityDensity):
        self.probabilities.append(prob)

    def plot(self):
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.1)

        ax.bar(self.probabilities[0].output_range, self.probabilities[0].prob)
        
        iteration = Slider(plt.axes([0.2, 0.02, 0.7, 0.02]), 'Iteration', 1, len(self.probabilities), valinit=0, valstep=1)

        def update(val):
            ax.clear()
            ax.bar(self.probabilities[int(iteration.val) - 1].output_range, self.probabilities[int(iteration.val) - 1].prob)
            plt.draw()

        iteration.on_changed(update)

        plt.show()

    def __getitem__(self, idx):
        return self.probabilities[idx]
    
    def __iter__(self):
        return iter(self.probabilities)

    def __len__(self):
        return len(self.probabilities)
    
    def __str__(self):
        return f"For {self.__len__()} iterations: \n" + "\n".join([str(p) for p in self.probabilities])
