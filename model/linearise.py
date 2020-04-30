import numpy
import model
import scipy.signal


def create_LinearModel(nonlinear_model: model.NonlinearModel,
                       X_op, input_op, T):
    """Linearise a non-linear model about an operating point to give
    a linear state space model

    Parameters
    ----------
    nonlinear_model : model.NonlinearModel
        The nonlinear model to be linearised
        
    X_op, input_op : array-like
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
            ans = fun(input_op)
            nonlinear_model.X = X_op
            return ans

        return fun(new_vec)

    matrices = [[[], []], [[], []]]
    for i, fun in enumerate([nonlinear_model.DEs, nonlinear_model.outputs]):
        for j, vec in enumerate([X_op, input_op]):
            nonlinear_model.X = X_op
            matrix = []
            for k in range(len(vec)):
                gradient = max_norm_error_close(f)
                matrix.append(gradient)
            matrices[i][j] = numpy.array(matrix).T

    (A, B), (C, D) = matrices
    Ad, Bd, Cd, Dd, _ = scipy.signal.cont2discrete((A, B, C, D), T)
    linear_model = model.LinearModel(Ad, Bd, Cd, Dd, T)

    return linear_model
