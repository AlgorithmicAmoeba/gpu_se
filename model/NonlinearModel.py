class NonlinearModel:
    """Base class for nonlinear models"""
    def DEs(self, Xs, inputs):
        raise NotImplementedError

    def step(self, dt, inputs):
        raise NotImplementedError

    def outputs(self, Xs, inputs):
        raise NotImplementedError