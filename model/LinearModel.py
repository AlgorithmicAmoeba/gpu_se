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

    f_bar, g_bar : array-lke
        The constants at linearisation

    Attributes
    -----------
    A, B, C, D : array-like
        2D array containing the relevant state space matrices

    dt : float
        The sampling interval

    x_bar, u_bar : array-like
        The state and input linearisation points

    f_bar, g_bar : array-lke
        The constants at linearisation

    Nx, Ni, No : int
        Number of states, inputs and outputs

    """
    def __init__(self, A, B, C, D, dt,
                 x_bar, u_bar, f_bar, g_bar):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.T = dt
        self.x_bar = x_bar
        self.u_bar = u_bar
        self.f_bar = f_bar
        self.g_bar = g_bar

        self.Nx = self.A.shape[0]
        self.Ni = self.B.shape[1]
        self.No = self.C.shape[0]

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

        nonlinear_model.X = old_X
        (A, B), (C, D) = matrices
        Ad, Bd, Cd, Dd, _ = scipy.signal.cont2discrete((A, B, C, D), T)
        f_bar = nonlinear_model.DEs(u_bar)
        g_bar = nonlinear_model.outputs(u_bar)
        linear_model = model.LinearModel(Ad, Bd, Cd, Dd, T, x_bar, u_bar, f_bar, g_bar)

        return linear_model
