class Pipeline:
    def __init__(self, steps):
        self.steps = list(steps)

    def __rshift__(self, other):
        return Pipeline([*self.steps, other])

    def run(self):
        data = None
        for step in self.steps:
            data = step.transform(data)
        return data

    def __repr__(self):
        return " >> ".join(step.name for step in self.steps)
