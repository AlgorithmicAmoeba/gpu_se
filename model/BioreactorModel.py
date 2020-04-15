# Contains code for the system model
import numpy


class Bioreactor:
    """A nonlinear model of the system

    Parameters
    ----------
    X0 : array_like
        Initial states

    inputs : callable
        Must take in a parameter t (the current time) and return an array_like of the current inputs

    t : float, optional
        Initial time.
        Defaults to zero

    pH_calculations : bool, optional
        If `True` then pH calculations are made.
        Defaults to `False`

    Attributes
    -----------
    X : array_like
        Array of current state

    inputs : callable
        Must take in a parameter t (the current time) and return an array_like of the current inputs

    t : float
        Initial time

    pH_calculations : bool
        If `True` then pH calculations are made

    rate_matrix_inv : 2d array_like
        The inverse of the rate matrix.
        Placed here so that it is only calculated once
    """
    def __init__(self, X0, inputs, t=0, pH_calculations=False):
        self.X = numpy.array(X0)
        self.inputs = inputs
        self.t = t
        self.pH_calculations = pH_calculations

        alpha, PO, gamma, theta, beta = 0.1, 0.1, 1.8, 0.1, 0.1
        rate_matrix = numpy.array([[1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 1],
                                   [-6, 4, 7/3, 2, -gamma],
                                   [0, 12, -1, 0, beta]])
        self.rate_matrix_inv = numpy.linalg.inv(rate_matrix)

    def DEs(self, t):
        """Contains the differential and algebraic equations for the system model.
        The rate equations defined in the matrix `rate_matrix` are described by: \n
        1) glucose + 2*CO2 + 6*ATP --> 2*FA + 2*water
        2) glucose --> 6*CO2 + 12*NADH + 4*ATP (TCA)
        3) NADH + 0.5*O2 -> 7/3 ATP (Respiration)
        4) glucose -> 2*ethanol + 2*CO2 + 2*ATP
        5) glucose + gamma*ATP --> 6*biomass + beta*NADH

        where the unknowns are: rFAp, rTCA, rResp, rEp, rXp

        Parameters
        ----------
        t : float
            The current time

        Returns
        -------
        dX : array_like
            The differential changes to the state variables
        """
        Ng, Nx, Nfa, Ne, Na, Nb, V, T = [max(0, N) for N in self.X]
        Fg_in, Cg_in, Fa_in, Ca_in, Fb_in, Cb_in, Fm_in, Fout, Tamb, Q = self.inputs(t)

        alpha, gamma, theta, beta = 0.1, 1.8, 0.2, 0.1

        # Concentrations
        Cg, Cx, Cfa, Ce, Ca, Cb = [N/V for N in [Ng, Nx, Nfa, Ne, Na, Nb]]

        rFAf = (1/3000) * (Cg / (1e-3 + Cg))
        rEf = (1/120) * (Cg / (1e-3 + Cg))
        rbio = (1/15) * (Cg / (1e-3 + Cg))
        theta_calc = theta * (Cg / (1e-3 + Cg))
        RHS = [rFAf, rEf, rbio, theta_calc, 0]

        rFAf, rTCA, rResp, rEf, rbio = self.rate_matrix_inv @ RHS

        rG = -rFAf - rTCA - rEf - rbio
        rX = 6 * rbio
        rFA = 2*rFAf
        rE = 2*rEf

        # DE's
        dNg = Fg_in*Cg_in - Fout*Cg + rG*Cx*V
        dNx = rX*Cx*V
        dNfa = -Fout*Cfa + rFA*Cx*V
        dNe = -Fout*Ce + rE*Cx*V
        dNa = - Fout * Ca
        dNb = Fb_in*Cb_in - Fout*Cb
        dV = Fg_in + Fb_in + Fm_in - Fout
        dT = 4.5*Q - 0.25*(T - Tamb)

        return dNg, dNx, dNfa, dNe, dNa, dNb, dV, dT

    def step(self, dt):
        """Updates the model with inputs

        Parameters
        ----------
        dt : float
            Time since previous step

        """
        self.t += dt
        dX = self.DEs(self.t)
        self.X += numpy.array(dX)*dt

    def calculate_pH(self):
        """Calculates the pH in the vessel.

        Returns
        -------
        pH : float
            The pH of the tank
        """
        K_fa1, K_fa2,  K_a, K_b, K_w = 10 ** (-3.03), 10 ** 4.44, 10 ** 8.08, 10 ** 0.56, 10 ** (-14)
        _, _, Nfa, _, Na, Nb, V, _ = self.X
        C_fa = Nfa/V
        C_a = Na/V
        C_b = Nb/V

        def charge_balance(pH_guess):
            Ch = 10 ** (-pH_guess)
            C_fa_minus = K_fa1 * C_fa / (K_fa1 + Ch)
            C_fa_minus2 = K_fa2 * C_fa_minus / (K_fa2 + Ch)
            C_cl_minus = K_a * C_a / (K_a + Ch)
            C_oh_minus = K_w / Ch
            C_na_plus = K_b * C_b / (K_b + C_oh_minus)

            balance = Ch + C_na_plus - C_fa_minus - C_fa_minus2 - C_cl_minus - C_oh_minus
            return balance

        pHs = numpy.linspace(0, 14, 100)
        CBs = charge_balance(pHs)
        index = numpy.argmin(abs(CBs))
        pH = pHs[index]
        if abs(CBs[index]) > 1e-1:
            print('ph CB:', CBs[index])

        return pH

    def outputs(self):
        """Returns all the outputs (state and calculated)

        Returns
        -------
        outputs : array_like
            List of all the outputs from the model
        """
        if self.pH_calculations:
            pH = self.calculate_pH()
            outs = numpy.append(self.X, pH)
        else:
            outs = self.X
        return outs
