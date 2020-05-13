import numpy
import model.NonlinearModel


class LinkedTanks(model.NonlinearModel):
    def __init__(self, X0, t0=0, linear=False):
        self.X = numpy.array(X0)
        self.t = t0
        self.linear = linear

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
        h1, h2 = self.X
        F1_in, F2_in = inputs  # Normally on the order of (0, 0.1)

        k1, k2, k_link = 0.1, 0.3, 0.05
        A1, A2 = 2, 8

        F_1to2 = k_link * (h1 - h2)
        if self.linear:
            dh1 = (F1_in - k1*h1*A1 - F_1to2)/A1
        else:
            dh1 = (F1_in - k1 * numpy.sqrt(h1 * A1) + F_1to2) / A1

        dh2 = (F2_in - k2*h2*A2)/A2

        dX = numpy.array([dh1, dh2])

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

        outs = numpy.array(self.X)
        _ = inputs
        return outs.copy()
