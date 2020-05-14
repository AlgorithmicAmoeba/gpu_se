# Contains code for the system model
import numpy
import model


class Bioreactor(model.NonlinearModel):
    """A nonlinear model of the system

    Parameters
    ----------
    X0 : array_like
        Initial states

    t : float, optional
        Initial time.
        Defaults to zero

    Attributes
    -----------
    X : array_like
        Array of current state

    t : float
        Initial time

    rate_matrix_inv : 2d array_like
        The inverse of the rate matrix.
        Placed here so that it is only calculated once
    """
    def __init__(self, X0, t=0):
        self.X = numpy.array(X0)
        self.t = t

        gamma, beta = 1.8, 0.1
        rate_matrix = numpy.array([[1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 1],
                                   [-6, 4, 7/3, 2, -6*gamma],
                                   [0, 12, -1, 0, 6*beta]])
        self.rate_matrix_inv = numpy.linalg.inv(rate_matrix)
        self.high_N = True
        self.V = 1  # L

    def DEs(self, inputs):
        """Contains the differential and algebraic equations for the system model.
        The rate equations defined in the matrix `rate_matrix` are described by: \n
        1) glucose + 2*CO2 + 6*ATP --> 2*FA + 2*water
        2) glucose --> 6*CO2 + 12*NADH + 4*ATP (TCA)
        3) NADH + 0.5*O2 -> 7/3 ATP (Respiration)
        4) glucose -> 2*ethanol + 2*CO2 + 2*ATP
        5) glucose + 6*gamma*ATP --> 6*biomass + 6*beta*NADH

        where the unknowns are: rFAp, rTCA, rResp, rEp, rXp

        Parameters
        ----------
        inputs : ndarray
            The inputs to the system at the current time

        Returns
        -------
        dX : array_like
            The differential changes to the state variables
        """
        Ng, Nx, Nfa, Ne, _ = [max(0, N) for N in self.X]
        Nh = self.X[4]
        Fg_in, Cg_in, Fm_in = inputs
        F_out = Fg_in + Fm_in

        V = self.V

        # Concentrations
        Cg, Cx, Cfa, Ce, Ch = [N/V for N in [Ng, Nx, Nfa, Ne, Nh]]

        if self.high_N:
            ks = 1/230, 1/12, 1/21
            rFAf, rEf, rX = [k * (Cg / (1e-3 + Cg)) for k in ks]
            theta_calc = 1.1 * (Cg / (1e-3 + Cg))

            RHS = [rFAf, rEf, rX, theta_calc, 0]

            rFAf, rTCA, rResp, rEf, rX = self.rate_matrix_inv @ RHS

            rG = (-rFAf - rTCA - rEf - rX) * Cx * V
            rX = 6 * rX * Cx * V
            rFA = 2 * rFAf * Cx * V
            rE = 2 * rEf * Cx * V
            rH = 0
        else:
            ks = 300 / 230, 1 / 120, 0
            rFAf, rEf, rX = [k * (Cg / (1e-3 + Cg)) for k in ks]
            theta_calc = 1.1 * (Cg / (1e-3 + Cg))

            RHS = [rFAf, rEf, rX, theta_calc, 0]

            rFAf, rTCA, rResp, rEf, rX = self.rate_matrix_inv @ RHS

            rG = (-rFAf - rTCA - rEf - rX) * Cx * V
            rX = 6 * rX * Cx * V
            rFA = 2 * rFAf * Cx * V
            rE = 2 * rEf * Cx * V
            rH = 0

        # DE's
        dNg = Fg_in * Cg_in - F_out * Cg + rG
        dNx = rX
        dNfa = -F_out * Cfa + rFA
        dNe = -F_out * Ce + rE
        dNh = rH

        return numpy.array([dNg, dNx, dNfa, dNe, dNh])

    def step(self, dt, inputs):
        """Updates the model with inputs

        Parameters
        ----------
        dt : float
            Time since previous step

        inputs : ndarray
            The inputs to the system at the current time
        """
        self.t += dt
        dX = self.DEs(inputs)
        self.X += numpy.array(dX)*dt

    def outputs(self, inputs):
        """Returns all the outputs (state and calculated)

        Returns
        -------
        outputs : array_like
            List of all the outputs from the model
        """
        outs = self.X.copy()
        molar_mass = numpy.array([180, 24.6, 116, 46, 1])
        outs[:5] = outs[:5] * molar_mass / self.V
        return outs

    def raw_outputs(self, inputs):
        """Returns all the outputs (state and calculated)

        Returns
        -------
        outputs : array_like
            List of all the outputs from the model
        """
        _ = inputs
        outs = self.X
        return outs
