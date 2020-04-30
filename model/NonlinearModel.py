class NonlinearModel:
    """Base class for nonlinear models"""
    def DEs(self, inputs):
        """Contains the differential and algebraic equations for the system model.

        Parameters
        ----------
        inputs : array-like
        """
        raise NotImplementedError

    def step(self, dt, inputs):
        """Updates the model with inputs

        Parameters
        ----------
        dt : float
        inputs : array-like
        """
        raise NotImplementedError

    def outputs(self, inputs):
        """Returns all the outputs (state and calculated)

        Parameters
        ----------
        inputs : array-like
        """
        raise NotImplementedError
