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

    pH_calculations : bool, optional
        If `True` then pH calculations are made.
        Defaults to `False`

    Attributes
    -----------
    X : array_like
        Array of current state

    t : float
        Initial time

    pH_calculations : bool
        If `True` then pH calculations are made

    rate_matrix_inv : 2d array_like
        The inverse of the rate matrix.
        Placed here so that it is only calculated once
    """
    def __init__(self, X0, t=0, pH_calculations=False):
        self.X = numpy.array(X0)
        self.t = t
        self.pH_calculations = pH_calculations

        gamma, beta = 1.8, 0.1
        rate_matrix = numpy.array([[1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 1],
                                   [-6, 4, 7/3, 2, -6*gamma],
                                   [0, 12, -1, 0, 6*beta]])
        self.rate_matrix_inv = numpy.linalg.inv(rate_matrix)
        self.high_N = True

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
        Ng, Nx, Nfa, Ne, Na, Nb, _, V, T = [max(0, N) for N in self.X]
        Nh = self.X[6]
        Fg_in, Cg_in, Fa_in, Ca_in, Fb_in, Cb_in, Fm_in, Fout, Tamb, Q = inputs

        # Concentrations
        Cg, Cx, Cfa, Ce, Ca, Cb, Ch = [N/V for N in [Ng, Nx, Nfa, Ne, Na, Nb, Nh]]

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
            rX = 0
            rH = (0.28 / 180 - Cg)

            rFA_max = 0.15/116
            rFA = rFA_max * (Cg / (1e-5 + Cg))

            r_theta1_max = 0.24/180 - 0.15/180
            r_theta1_req = r_theta1_max - (r_theta1_max/2/(0.28/180)*rH + 0.01 * Ch)
            r_theta1 = min(r_theta1_max, max(0, r_theta1_req)) * (Cg / (1e-5 + Cg))

            rE_req = r_theta1_req - r_theta1_max
            rE = min(3.804e-4, max(0, rE_req))

            r_theta2_max = 0.06 / 180 - 0.0175 / 180
            r_theta2_req = r_theta1_req - r_theta1_max - rE
            r_theta2 = min(r_theta2_max, max(0, r_theta2_req))

            rG = -rFA * (116/180) - r_theta1 - rE * (46/180) - r_theta2

        # DE's
        dNg = Fg_in*Cg_in - Fout*Cg + rG
        dNx = rX
        dNfa = -Fout*Cfa + rFA
        dNe = -Fout*Ce + rE
        dNa = Fa_in*Ca_in - Fout * Ca
        dNb = Fb_in*Cb_in - Fout*Cb
        dV = Fg_in + Fa_in + Fb_in + Fm_in - Fout
        dT = 4.5*Q - 0.25*(T - Tamb)
        dNh = rH

        return numpy.array([dNg, dNx, dNfa, dNe, dNa, dNb, dNh, dV, dT])

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

    def calculate_pH(self):
        """Calculates the pH in the vessel.

        Returns
        -------
        pH : float
            The pH of the tank
        """
        K_fa1, K_fa2,  K_a, K_b, K_w = 10 ** (-3.03), 10 ** 4.44, 10 ** 8.08, 10 ** 0.56, 10 ** (-14)
        _, _, Nfa, _, Na, Nb, _, V, _ = self.X
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
        # if abs(CBs[index]) > 1e-1:
        #     print('ph CB:', CBs[index])

        return pH

    def outputs(self, inputs):
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
        outs[:7] /= outs[7]
        return outs

    def raw_outputs(self, inputs):
        """Returns all the outputs (state and calculated)

        Returns
        -------
        outputs : array_like
            List of all the outputs from the model
        """
        _ = inputs
        if self.pH_calculations:
            pH = self.calculate_pH()
            outs = numpy.append(self.X, pH)
        else:
            outs = self.X
        return outs
