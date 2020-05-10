import numpy
import model.NonlinearModel


class TankModel(model.NonlinearModel):
    def __init__(self, X0, t0=0, simple=False):
        self.X = numpy.array(X0)
        self.t = t0
        self.simple = simple

    def DEs(self, inputs):
        """Contains the differential and algebraic equations for the system model.

        Parameters
        ----------
        inputs : ndarray
            The inputs to the system at the current time

        Returns
        -------
        dX : array_like
            The differential changes to the state variables
        """
        h, = self.X
        F_in, = inputs  # Normally on the order of (0, 0.1)

        k = 0.1
        A = 2
        if self.simple:
            dh = (F_in - k*h*A)/A
        else:
            dh = (F_in - k * numpy.sqrt(h * A)) / A

        dX = numpy.array([dh])

        return dX

    def step(self, dt, inputs):
        """Updates the model with inputs

        Parameters
        ----------
        inputs
        dt : float
            Time since previous step

        """
        self.t += dt
        dX = self.DEs(inputs)
        self.X += dX*dt
        return self.outputs(inputs)

    def outputs(self, inputs):
        """Returns all the outputs (state and calculated)

        Returns
        -------
        outputs : array_like
            List of all the outputs from the model
        """

        outs = numpy.array(self.X[0])
        _ = inputs
        return outs.copy()
