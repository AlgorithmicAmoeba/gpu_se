# Contains code for the system model
import numpy
import model
import scipy.optimize


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
    def __init__(self, X0, t=0, high_N=True):
        self.X = numpy.array(X0)
        self.t = t

        gamma, beta = 1.8, 0.1
        rate_matrix = numpy.array([[1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 1],
                                   [-6, 4, 7/3, 2, -6*gamma],
                                   [0, 12, -1, 0, 6*beta]])
        self.rate_matrix_inv = numpy.linalg.inv(rate_matrix)
        self.high_N = high_N

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
        Cg, Cx, Cfa, Ce, _ = [max(0, N) for N in self.X]
        Fg_in, Cg_in, Fm_in = inputs
        F_out = Fg_in + Fm_in

        V = 1  # L

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

            dCg = (Fg_in * Cg_in - F_out * Cg + rG) / V
            dCx = rX / V
            dCfa = (-F_out * Cfa + rFA) / V
            dCe = (-F_out * Ce + rE) / V
            dCh = rH / V
        else:
            dCg, dCx, dCfa, dCe, dCh = Bioreactor.homeostatic_DEs(self.X, inputs)

        return numpy.array([dCg, dCx, dCfa, dCe, dCh])

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
        self.X[:4] = numpy.maximum(self.X[:4], 0)

    def outputs(self, inputs):
        """Returns all the outputs (state and calculated)

        Returns
        -------
        outputs : array_like
            List of all the outputs from the model
        """
        outs = self.X.copy()
        molar_mass = numpy.array([180, 24.6, 116, 46, 1])
        outs[:5] = outs[:5] * molar_mass
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

    @staticmethod
    def find_SS(U_op, X0):
        bioreactor_SS = model.Bioreactor(X0=[], high_N=False)

        def fun(x_ss):
            temp = bioreactor_SS.X
            bioreactor_SS.X = x_ss
            ans = bioreactor_SS.DEs(U_op)
            bioreactor_SS.X = temp
            return ans

        return scipy.optimize.fsolve(fun, X0)

    @staticmethod
    def homeostatic_DEs(x, u, dt=1):
        Cg, Cx, Cfa, Ce, Ch = x
        Cg, Cx, Cfa, Ce = max(Cg, 0), max(Cx, 0), max(Cfa, 0), max(Ce, 0)

        Fg_in, Cg_in, Fm_in = u
        F_out = Fg_in + Fm_in

        V = 1  # L

        rX = 0
        rH = (0.28 / 180 - Cg)

        rFA_max = 0.15 / 116
        rFA = rFA_max * (Cg / (1e-5 + Cg))

        r_theta1_max = 0.24 / 180 - 0.15 / 180
        r_theta1_req = r_theta1_max - (r_theta1_max / 2 / (0.28 / 180) * rH + 0.01 * Ch)
        r_theta1 = min(r_theta1_max, max(0, r_theta1_req)) * (Cg / (1e-5 + Cg))

        rE_req = r_theta1_req - r_theta1_max
        rE = min(3.804e-4, max(0, rE_req))

        r_theta2_max = 0.06 / 180 - 0.0175 / 180
        r_theta2_req = r_theta1_req - r_theta1_max - rE
        r_theta2 = min(r_theta2_max, max(0, r_theta2_req))

        rG = -rFA * (116 / 180) - r_theta1 - rE * (46 / 180) - r_theta2

        dCg = (Fg_in * Cg_in - F_out * Cg + rG) / V * dt
        dCx = rX / V * dt
        dCfa = (-F_out * Cfa + rFA) / V * dt
        dCe = (-F_out * Ce + rE) / V * dt
        dCh = rH / V * dt

        return dCg, dCx, dCfa, dCe, dCh

    @staticmethod
    def static_outputs(x, u):
        Cg, _, Cfa, _, _ = x
        _ = u
        return Cg*180, Cfa*116
