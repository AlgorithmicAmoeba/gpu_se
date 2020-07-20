# Contains code for the system model
import numpy
import model
import scipy.optimize


class Bioreactor(model.NonlinearModel):
    """A nonlinear model of a bioreactor based off of
    conclusions by Swart and Iplik.

    Parameters
    ----------
    X0 : numpy.array
        Initial states

    t : float, optional
        Current time.
        Defaults to zero

    high_N : bool, optional
        A flag setting whether the reactor is in the high
        or low nitrogen regime

    Attributes
    -----------
    X : numpy.array
        Array of current state
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
        self._rate_matrix_inv = numpy.linalg.inv(rate_matrix)
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
        inputs : numpy.array
            The inputs to the system at the current time

        Returns
        -------
        dX : numpy.array
            The differential changes to the state variables
        """
        Cg, Cx, Cfa, Ce, _ = [max(0, N) for N in self.X]
        Fg_in, Fm_in = inputs
        Cg_in = 5000/180
        F_out = Fg_in + Fm_in

        V = 1  # L

        if self.high_N:
            ks = 1/230, 1/12, 1/21
            rFAf, rEf, rX = [k * (Cg / (1 + Cg)) for k in ks]
            theta_calc = 1.1 * (Cg / (1 + Cg))

            RHS = [rFAf, rEf, rX, theta_calc, 0]

            rFAf, rTCA, rResp, rEf, rX = self._rate_matrix_inv @ RHS

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

        inputs : numpy.array
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
        outputs : numpy.array
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
        outputs : numpy.array
            List of all the outputs from the model
        """
        _ = inputs
        outs = self.X
        return outs

    # noinspection PyTupleItemAssignment
    @staticmethod
    def find_SS(U_op, X0):
        """Determines a steady state of the system,
        given an inputs, and a nearby state.

        Parameters
        ----------
        U_op : numpy.array
            Input array

        X0 : numpy.array
            A nearby state

        Returns
        -------
        res : numpy.array
            Steady state values
        """
        bioreactor_SS = model.Bioreactor(X0=[], high_N=False)

        def fun(x_ss):
            temp = bioreactor_SS.X
            bioreactor_SS.X = x_ss
            bioreactor_SS.X[1] = X0[1]
            ans = bioreactor_SS.DEs(U_op)
            bioreactor_SS.X = temp
            return ans

        res = scipy.optimize.fsolve(fun, X0)
        # noinspection PyUnresolvedReferences
        res[1] = X0[1]
        return res

    @staticmethod
    def homeostatic_DEs(x, u, dt=1):
        """The differential equations for the low nitrogen production
        phase of the reactor operation.
        Separated so that it can be passed to numba and parallelized

        Parameters
        ----------
        x : array
            Current state

        u : array
            Input to the system

        dt : array
            Time since previous euler update

        Returns
        -------
        dCg, dCx, dCfa, dCe, dCh : float
            Changes in the states
        """
        Cg, Cx, Cfa, Ce, Ch = x
        Cg, Cx, Cfa, Ce = max(Cg, 0), max(Cx, 0), max(Cfa, 0), max(Ce, 0)

        Fg_in, Fm_in = u
        Cg_in = 5000/180
        F_out = Fg_in + Fm_in

        V = 1  # L

        rX = 0. * Cx
        rH = (280 / 180 - Cg)

        # (molFA / min) = (gFA/gX/min) (molFA/gFA) (molX/Lv) (gX/molX) (Lv)
        rFA_max = 0.25 / 116 * Cx * 24.6 * V
        rFA = rFA_max * (Cg / (1e-2 + Cg))

        # (molG / min) = (gG/gX/min) (molG/gG) (molX/Lv) (gX/molX) (Lv)
        r_theta1_max = (0.4 - 0.25) / 180 * Cx * 24.6 * V
        r_theta1_req = r_theta1_max - (r_theta1_max / 2000 / (0.28 / 180) * rH + 0.01 * Ch)
        r_theta1 = min(r_theta1_max, max(0, r_theta1_req)) * (Cg / (1e-2 + Cg))

        # (molE / min) = (gE/gX/min) (molE/gE) (molX/Lv) (gX/molX) (Lv)
        r_E_max = 0.025 / 46 * Cx * 24.6 * V
        rE_req = r_theta1_req - r_theta1_max
        rE = min(r_E_max, max(0, rE_req))

        # (molG / min) = (gG/gX/min) (molG/gG) (molX/Lv) (gX/molX) (Lv)
        r_theta2_max = (0.1 - 0.025) / 180 * Cx * 24.6 * V
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
        """Returns the outputs.
        Made static so that it can be parallelized by numba

        Parameters
        ----------
        x : array
            Current state

        u : array
            Input to the system

        Returns
        -------
        Mg, Mfa : float
            Mass of glucose and fumaric acid
        """
        Cg, _, Cfa, _, _ = x
        _ = u
        return Cg*180, Cfa*116
