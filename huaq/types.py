from typing import Any, Callable

# precomputed or hydrated gate parameters
class Var:
    def __init__(self, name: str, value, precomputed: bool = True):
        self.name = name
        self.value = value
        self.precomputed = precomputed

# base class for all wrapper classes with a range of values for a variable
class Range:
    def __init__(self, space: list[float]):
        self.space = space

        self.name: str = ''

    def set_name(self, name: str):
        self.name = name

    def __len__(self):
        return len(self.space)

# class to handle variables that need to change for many iterations
class Sweep(Range):
    def __init__(self, start: float, end: float, steps: int):
        self.start = start
        self.end = end
        self.steps = steps

        self.step = (end - start) / steps

        self.values = []
        current = start
        for _ in range(steps):
            self.values.append(current)
            current += self.step

        super().__init__(self.values)
        del self.values

# class to handle variables that come from processing data
class Processor(Range):
    def __init__(self, process: Callable[[Any], float], inputs: list[Any]):
        self.process = process
        self.inputs = inputs

        self.values = []
        for _ in range(len(self.values)):
            self.values.append(self.process(self.values))

        super().__init__(self.values)
        del self.values
