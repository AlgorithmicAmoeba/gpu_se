# Contains code for the system model
import numpy


def smooth(x, x_mid, x_range, y1, y2):
    dist = x_range / 2
    x1 = x_mid - dist
    x2 = x_mid + dist

    if x <= x1:
        return y1

    if x >= x2:
        return y2

    return numpy.interp(x, [x1, x2], [y1, y2])


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

        self._Xs = [self.outputs()]

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
        Ng, Nx, Nfa, Ne, Nco, No, Nn, Na, Nb, Nez, Nfaz, Nezfa, V, Vg, T = [max(0, N) for N in self.X]
        Fg_in, Cg_in, Fco_in, Cco_in, Fo_in, Co_in, \
            Fg_out, Cn_in, Fn_in, Fb_in, Cb_in, Fm_in, Fout, Tamb, Q = self.inputs(t)

        alpha, gamma, theta, beta = 0.1, 1.8, 0.2, 0.1
        delta = 0.2

        # Concentrations
        Cg, Cx, Cfa, Ce, Cn, Ca, Cb, Cez, Cfaz, Cezfa = [N/V for N in [Ng, Nx, Nfa, Ne, Nn, Na, Nb, Nez, Nfaz, Nezfa]]
        Cco, Co = [N/Vg for N in [Nco, No]]

        ln2 = numpy.log(2)
        kI_faz, kD_faz, r_max_faz = [1.6, ln2/0.25, 400] if Cn < 0.01 else [7e-3, ln2/0.25, 400]
        kI_ez, kI_ezfa = 1.6, 0.48
        kD_ez, kD_ezfa = ln2 / (1/15), ln2 / 0.75
        r_max_ez = smooth(Cg, 0.5/180, 0.6/180, 0, 1500)
        # r_max_ez =  1500 if Cg > 0.3/180 else 0
        r_max_ezfa = smooth(Cg, 0.3/180, 0.2/180, 200, 0) if Cn < 0.01 else 0
        # r_max_ezfa = 200 if Cg < 0.3/180 and Cn < 0.01 else 0

        Cfaz_eqi = (-kI_faz + numpy.sqrt(kI_faz ** 2 + 4 * r_max_faz * kI_faz / kD_faz)) / 2
        Cez_eqi = (-kI_ez + numpy.sqrt(kI_ez ** 2 + 4 * r_max_ez * kI_ez / kD_ez)) / 2
        Cezfa_eqi = (-kI_ezfa + numpy.sqrt(kI_ezfa ** 2 + 4 * r_max_ezfa * kI_ezfa / kD_ezfa)) / 2

        x = 0.3
        kBio = 0.02/x
        kFA_max = 0.0001/x  # (kBio * 0.4) / Cfaz_eqi
        kE_max = 0.0025/x  # (kBio * 14) / Cez_eqi
        kEzFA_max = 0.0022/x/1.5  # 1e-2 / Cezfa_eqi

        rE2FA = kEzFA_max * Cezfa * (Ce / (1e-3 + Ce))
        rFAf = kFA_max * Cfaz * (Cg / (1e-3 + Cg))
        rEf = kE_max * Cez * (Cg / (1e-3 + Cg))
        rbio = kBio * (Cg / (1e-3 + Cg)) * (Cn / (1e-3 + Cn))
        theta_calc = theta * (Cg / (1e-3 + Cg))
        RHS = [rFAf, rEf, rbio, theta_calc, 0]

        rFAf, rTCA, rResp, rEf, rbio = self.rate_matrix_inv @ RHS

        rG = -rFAf - rTCA - rEf - rbio
        rX = 6 * rbio
        rCO = -2 * rFAf + 6 * rTCA + 2 * rEf + alpha * rbio
        rO = -0.5*rResp
        rFA = 2*rFAf + + 0.5 * rE2FA
        rE = 2*rEf - rE2FA

        # Enzymatic rates
        # kD_faz = kD_faz if Cg > 0.01/180 else ln2 / 10
        kD_ez = smooth(Cg, 0.3/180, 0.1/180, ln2 / 0.15, kD_ez)
        # kD_ez = kD_ez if Cg > 0.3/180 else ln2 / 0.15
        # kD_ezfa =  kD_ezfa if (Ce > 0 and Cg < 0.3/180) else ln2 / 0.1
        kD_ezfa = smooth(Cg, 0.3/180, 0.2/180, kD_ezfa, ln2 / 0.1) if Ce > 0 else ln2 / 0.1

        # rFAz = r_max_faz * (kI_faz / (kI_faz + Cfaz)) if Cg > 0 else 0
        rFAz = r_max_faz * (kI_faz / (kI_faz + Cfaz))
        rFAz = smooth(Cg, 0.01/180, 0.02/180, 0, rFAz)
        rFAz -= kD_faz * Cfaz
        rEz = r_max_ez * (kI_ez / (kI_ez + Cez))  # if Cg > 0.5/180 else 0
        rEz -= kD_ez * Cez
        # rEzFA = r_max_ezfa * (kI_ezfa / (kI_ezfa + Cezfa)) if Ce > 0 else 0
        rEzFA = r_max_ezfa * (kI_ezfa / (kI_ezfa + Cezfa))
        rEzFA = smooth(Ce, 0.01/40, 0.02/40, 0, rEzFA)
        rEzFA -= kD_ezfa * Cezfa

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
        dNez = rEz*Cx*V
        dNfaz = rFAz*Cx*V
        dNezfa = rEzFA*Cx*V
        dV = Fg_in + Fn_in + Fb_in + Fm_in - Fout
        dVg = Fco_in + Fo_in - Fg_out
        dT = 4.5*Q - 0.25*(T - Tamb)

        return dNg, dNx, dNfa, dNe, dNco, dNo, dNn, dNa, dNb, dNez, dNfaz, dNezfa,  dV, dVg, dT

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
        self._Xs.append(self.outputs())

    def calculate_pH(self):
        """Calculates the pH in the vessel.

        Returns
        -------
        pH : float
            The pH of the tank
        """
        K_fa1, K_fa2,  K_a, K_b, K_w = 10 ** (-3.03), 10 ** 4.44, 10 ** 8.08, 10 ** 0.56, 10 ** (-14)
        _, _, Nfa, _, _, _, _, Na, Nb, _, _, _, V, _, _ = self.X
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

    def get_Xs(self):
        """Gets all the states that are stored"""
        return numpy.array(self._Xs)

    def get_data(self):
        """Gets all relevant information from the object """
        return self.get_Xs()
