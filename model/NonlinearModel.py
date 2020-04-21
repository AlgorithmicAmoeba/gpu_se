class NonlinearModel:
    """Base class for nonlinear models"""
    def DEs(self, inputs):
        raise NotImplementedError

    def step(self, dt, inputs):
        raise NotImplementedError

    def outputs(self, inputs):
        raise NotImplementedError
