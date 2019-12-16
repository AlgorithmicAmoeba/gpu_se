import numpy
import model


class CSTRModel(model.NonlinearModel):
    """A nonlinear model of a CSTR with an exothermic, irreversible reaction
    :math:`A \\rightarrow B`. The only manipulated variable is the heat added
    to the reactor Q.

    Parameters
    ----------
    X0 : array_like
        Initial states

    t0 : float, optional
        Initial time.
        Defaults to zero

    Attributes
    -----------
    X : array_like
        Array of current state

    t : float
        Current time
    """
    def __init__(self, X0, t0=0):
        self.X = numpy.array(X0)
        self.t = t0

    def DEs(self, Xs, inputs):
        """Contains the differential and algebraic equations for the system model.

        Parameters
        ----------
        Xs : ndarray
            The states of the system at the current time

        inputs : ndarray
            The inputs to the system at the current time

        Returns
        -------
        dX : array_like
            The differential changes to the state variables
        """
        Ca, T = [max(0, N) for N in Xs]
        Q, F = inputs  # Normally on the order of (0, 0.1)

        V, Ca0, dH, E, rho, R, Ta0, k0, Cp = 5, 1, -4.78e4, 8.314e4, 1e3, 8.314, 310, 72e7, 0.239

        D = F/V
        rate = k0*numpy.exp(-E/R/T)*Ca

        dCa = D*(Ca0 - Ca) - rate
        dT = D*(Ta0 - T) - dH/rho/Cp*rate + Q/rho/Cp/V

        dX = numpy.array([dCa, dT])

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
        dX = self.DEs(self.X, inputs)
        self.X += dX*dt
        return self.outputs(self.X, inputs)

    def outputs(self, Xs, inputs):
        """Returns all the outputs (state and calculated)

        Returns
        -------
        outputs : array_like
            List of all the outputs from the model
        """

        outs = Xs
        _ = inputs
        return outs.copy()
