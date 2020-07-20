import numpy
import model.NonlinearModel


class DiagTank(model.NonlinearModel):
    def __init__(self, X0, t0=0):
        self.X = numpy.array(X0)
        self.t = t0

        self.linear_tank = model.TankModel(numpy.atleast_1d(X0[0]), linear=True)
        self.nonlinear_tank = model.TankModel(numpy.atleast_1d(X0[1]), linear=False)

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

        dh_lin = self.linear_tank.DEs(numpy.atleast_1d(inputs[0]))
        dh_nonlin = self.linear_tank.DEs(numpy.atleast_1d(inputs[1]))

        dX = numpy.hstack([dh_lin, dh_nonlin])

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
