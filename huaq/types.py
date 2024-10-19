# precomputed or hydrated gate parameters
class Var:
    def __init__(self, name: str, value, precomputed: bool = True):
        self.name = name
        self.value = value
        self.precomputed = precomputed

# class to handle variables that need to change for many iterations
class Sweep:
    def __init__(self, start: float, end: float, steps: int):
        self.start = start
        self.end = end
        self.steps = steps

        self.current = start
        self.step = (end - start) / steps
        self.name: str = ''

        self.values = []
        for _ in range(steps):
            self.values.append(self.current)
            self.current += self.step

    def next(self):
        self.current += self.step
        if self.current - self.step * 0.01 > self.end:
            self.reset()
        return self.current
    
    def reset(self):
        self.current = self.start

