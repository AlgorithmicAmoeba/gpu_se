import numpy
import noise


class LinearModel:
    """Structured data for a discrete state space linear model with noise

    .. math:
        x_{k+1} &= A x_k + B u_k + w_k\\
        y_k &= C x_k + D u_k + v_k

    where :math:`w_k` and :math:`v_k` are additive noise sampled from some distribution

    Parameters
    ----------
    A, B, C, D : ndarray
        2D array containing the relevant state space matrices

    state_noise, measurement_noise : noise.Noise
        Objects containing state and measurement noise information

    Attributes
    -----------
    A, B, C, D : ndarray
        2D array containing the relevant state space matrices

    state_noise, measurement_noise : noise.Noise
        Objects containing state and measurement noise information

    Nx, Ni, No : int
        Number of states, inputs and outputs

    """
    def __init__(self, A, B, C, D,
                 state_noise: noise.Noise,
                 measurement_noise: noise.Noise):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.w = self.state_noise = state_noise
        self.v = self.measurement_noise = measurement_noise

        self.Nx = self.A.shape[0]
        self.Ni = self.B.shape[1]
        self.No = self.C.shape[0]


class BioreactorModel:
    """A nonlinear model of a low dilution rate fed batch bioreactor
    with *Rhizopus oryzae* producing fumaric acid and ethanol from
    glucose feed. The following reactions take place in the reactor: \n
    1) glucose + 2 :math:`CO_2` + 6 ATP --> 2 FA + 2 water
    2) glucose --> 6 :math:`CO_2` + 12 NADH + 4 ATP (TCA)
    3) NADH + 0.5 :math:`O_2` -> 7/3 ATP (Respiration)
    4) glucose -> 2 ethanol + 2 :math:`CO_2` + 2 ATP
    5) glucose + :math:`\gamma` ATP --> 6 biomass + :math:`\\beta` NADH

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

        alpha, PO, gamma, theta, beta = 0.1, 0.1, 1.8, 0.1, 0.1
        rate_matrix = numpy.array([[1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 1],
                                   [-6, 4, 7/3, 2, -gamma],
                                   [0, 12, -1, 0, beta]])
        self._rate_matrix_inv = numpy.linalg.inv(rate_matrix)

    def DEs(self, Xs, inputs):
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
        Xs : ndarray
            The states of the system at the current time

        inputs : ndarray
            The inputs to the system at the current time

        Returns
        -------
        dX : array_like
            The differential changes to the state variables
        """
        Ng, Nx, Nfa, Ne, Nco, No, Nn, Na, Nb, Nz, Ny, V, Vg, T = [max(0, N) for N in Xs]
        Fg_in, Cg_in, Fco_in, Cco_in, Fo_in, Co_in, \
            Fg_out, Cn_in, Fn_in, Fb_in, Cb_in, Fm_in, Fout, Tamb, Q = inputs

        alpha, PO, gamma, theta, beta = 0.1, 0.1, 1.8, 0.1, 0.1
        delta = 0.2

        # Concentrations
        Cg, Cx, Cfa, Ce, Cn, Ca, Cb, Cz, Cy = [N/V for N in [Ng, Nx, Nfa, Ne, Nn, Na, Nb, Nz, Ny]]
        Cco, Co = [N/Vg for N in [Nco, No]]

        first_increase = (0.6 / 46 / 25 * 4) * Cy * 1.8
        second_increase = 2/46/120*3.2
        decrease = (0.6 / 46 / 40*3) * Cz/3
        rZ = decrease + second_increase if Cz > 0 else 0  # decrease
        rY = first_increase + decrease if Cy > 0 else 0  # increase

        rFAf = 15e-3 * (Cg / (1e-2 + Cg)) - 0.5 * rZ
        rEf = (second_increase + rY) * (Cg / (1e-5 + Cg))
        theta_calc = theta * (Cg / (1e-3 + Cg))
        RHS = [rFAf, rEf, 8e-5, theta_calc, 0]

        rFAf, rTCA, rResp, rEf, rbio = self._rate_matrix_inv @ RHS

        rG = -rFAf - rTCA - rEf - rbio
        rX = 6 * rbio
        rFA = 2*(rFAf + 0.5 * rZ)
        rE = 2 * (rEf - rZ) * (Cg / (1e-5 + Cg))
        rCO = -2 * rFAf + 6 * rTCA + 2 * rEf + alpha * rbio
        rO = -0.5*rResp

        # DE's
        dNg = Fg_in*Cg_in - Fout*Cg + rG*Cx*V
        dNx = rX*Cx*V
        dNfa = -Fout*Cfa + rFA*Cx*V
        dNe = -Fout*Ce + rE*Cx*V
        dNco = Fco_in*Cco_in - Fg_out*Cco + rCO*Cx*V
        dNo = Fo_in*Co_in - Fg_out*Co - rO*Cx*V
        dNn = Fn_in*Cn_in - Fout*Cn - delta*rX*Cx*V
        dNa = - Fout * Ca
        dNb = Fb_in*Cb_in - Fout*Cb
        dNz = -190*rZ*Cx*V
        dNy = -95*rY*Cx*V
        dV = Fg_in + Fn_in + Fb_in + Fm_in - Fout
        dVg = Fco_in + Fo_in - Fg_out
        dT = 4.5*Q - 0.25*(T - Tamb)

        return dNg, dNx, dNfa, dNe, dNco, dNo, dNn, dNa, dNb, dNz, dNy, dV, dVg, dT

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
        self.X += numpy.array(dX)*dt
        return self.outputs(self.X, inputs)

    @staticmethod
    def calculate_pH(Xs):
        """Calculates the pH in the vessel.

        Returns
        -------
        pH : float
            The pH of the tank
        """
        K_fa1, K_fa2,  K_a, K_b, K_w = 10 ** (-3.03), 10 ** 4.44, 10 ** 8.08, 10 ** 0.56, 10 ** (-14)
        _, _, Nfa, _, _, _, _, Na, Nb, _, _, V, _, _ = Xs
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

    def outputs(self, Xs, inputs):
        """Returns all the outputs (state and calculated)

        Returns
        -------
        outputs : array_like
            List of all the outputs from the model
        """

        pH = self.calculate_pH(Xs)
        Ng, Nx, Nfa, Ne, Nco, No, Nn, V, Vg, T = Xs
        _ = inputs
        outs = Ng/V, Nx/V, Nfa/V, Ne/V, Nco/Vg, No/Vg, Nn/V, T, pH
        return outs


class CSTRModel:
    """A nonlinear model of a CSTR with an exothermic, irreversible reaction
    :math:`A \rightarrow B`. The only manipulated variable is the heat added
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

    @staticmethod
    def DEs(Xs, inputs):
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
        Q, = inputs

        V, Ca0, dH, E, rho, R, Ta0, k0, Cp, F = 5, 1, -4.78e4, 8.314e4, 1e3, 3.314, 310, 72e7, 0.239, 0.1

        D = F/V
        rate = k0*numpy.exp(-E/R/T)*Ca

        dCa = D*(Ca0 - Ca) - rate
        dT = D*(Ta0 - T) - dH/rho/Cp*rate + Q/rho/Cp/V

        return dCa, dT

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
        self.X += numpy.array(dX)*dt
        return self.outputs(self.X, inputs)

    @staticmethod
    def outputs(Xs, inputs):
        """Returns all the outputs (state and calculated)

        Returns
        -------
        outputs : array_like
            List of all the outputs from the model
        """

        outs = Xs
        _ = inputs
        return outs
