import numpy
import scipy.signal

import model


class LinearModel:
    """Structured data for a discrete state space linear model with noise

    .. math:
        x_{k+1} &= A x_k + B u_k + w_k\\
        y_k &= C x_k + D u_k + v_k

    where :math:`w_k` and :math:`v_k` are additive noise sampled from some distribution

    Parameters
    ----------
    A, B, C, D : array-like
        2D array containing the relevant state space matrices

    dt : float
        The sampling interval

    x_bar, u_bar : array-like
        The state and input linearisation points

    f_bar, y_bar : array-lke
        The constants at linearisation

    Attributes
    -----------
    A, B, C, D : array-like
        2D array containing the relevant state space matrices

    dt : float
        The sampling interval

    x_bar, u_bar : array-like
        The state and input linearisation points

    f_bar, y_bar : array-lke
        The constants at linearisation

    Nx, Ni, No : int
        Number of states, inputs and outputs

    """
    def __init__(self, A, B, C, D, dt,
                 x_bar, u_bar, f_bar, y_bar):
        A, B, C, D = [numpy.atleast_2d(m) for m in [A, B, C, D]]
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.T = dt
        self.x_bar = x_bar
        self.u_bar = u_bar
        self.f_bar = f_bar
        self.y_bar = y_bar

        self.Nx = self.A.shape[0]
        self.Ni = self.B.shape[1]
        self.No = self.C.shape[0]

        self.states = range(self.Nx)
        self.inputs = range(self.Ni)
        self.outputs = range(self.No)

    @staticmethod
    def create_LinearModel(nonlinear_model: model.NonlinearModel,
                           x_bar, u_bar, T):
        """Linearise a non-linear model about an operating point to give
        a linear state space model

        .. math::
            x_{k+1} = A (x_k - x_bar) + B (u_k - u_bar) + f_bar
            y_k = C (x_k - x_bar) + D (u_k - u_bar) + g_bar

        Parameters
        ----------
        nonlinear_model : model.NonlinearModel
            The nonlinear model to be linearised

        x_bar, u_bar : array-like
            The state and input around which the model should be linearised

        T : float
            The sampling interval

        Returns
        -------
        linear_model : model.LinearModel
            The linear model of the system
        """

        def max_norm_error_close(g, tol=1e-8, x=0.1):
            """Takes in a function :math:`g` that takes in a
            parameter :math:`x` and evaluates
            :math:`\gamma = \frac{g(x) - g(-x)}{2x}` as an approximation
            of the gradient.
            It halves :math:`x` until :math:`e_k = || \gamma_k - \gamma_{k-1}||_\infty < tol`
            """
            gamma = (g(x) - g(-x)) / 2 / x
            e = tol + 1
            while e > tol:
                x /= 2
                new_gamma = (g(x) - g(-x)) / 2 / x
                e = numpy.max(numpy.abs(new_gamma - gamma))
                gamma = new_gamma

            return gamma

        def f(x):
            new_vec = vec.copy()
            new_vec[k] += x

            if j == 0:
                nonlinear_model.X = new_vec
                ans = fun(u_bar)
                nonlinear_model.X = x_bar
                return ans

            return fun(new_vec)

        matrices = [[[], []], [[], []]]
        old_X = nonlinear_model.X
        for i, fun in enumerate([lambda x: nonlinear_model.DEs(x), lambda x: nonlinear_model.outputs(x)]):
            for j, vec in enumerate([x_bar, u_bar]):
                nonlinear_model.X = x_bar
                matrix = []
                for k in range(len(vec)):
                    gradient = max_norm_error_close(f)
                    matrix.append(gradient)
                matrices[i][j] = numpy.array(matrix).T

        (A, B), (C, D) = matrices
        Ad, Bd, Cd, Dd, _ = scipy.signal.cont2discrete((A, B, C, D), T)
        f_bar = nonlinear_model.DEs(u_bar)
        y_bar = nonlinear_model.outputs(u_bar)
        linear_model = model.LinearModel(Ad, Bd, Cd, Dd, T, x_bar, u_bar, f_bar, y_bar)
        nonlinear_model.X = old_X

        return linear_model

    def select_subset(self, states, inputs, outputs):
        self.states = states
        self.inputs = inputs
        self.outputs = outputs
        self.A = self.A[states][:, states]
        self.B = self.B[states][:, inputs]
        self.C = self.C[outputs][:, states]
        self.D = self.D[outputs][:, inputs]
        self.x_bar = self.x_bar[states]
        self.u_bar = self.u_bar[inputs]
        self.f_bar = self.f_bar[states]
        self.y_bar = self.y_bar[outputs]
        self.Nx = len(states)
        self.Ni = len(inputs)
        self.No = len(outputs)

    def xd2n(self, x):
        return x + self.x_bar

    def xn2d(self, x, subselect=True):
        if subselect:
            return x[self.states] - self.x_bar
        return x - self.x_bar

    def yd2n(self, y):
        return y + self.y_bar

    def yn2d(self, y, subselect=True):
        if subselect:
            return y[self.outputs] - self.y_bar
        return y - self.y_bar

    def ud2n(self, u):
        return u + self.u_bar

    def un2d(self, u, subselect=True):
        if subselect:
            return u[self.inputs] - self.u_bar
        return u - self.u_bar


