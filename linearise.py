import numpy
import model


def create_LinearModel(nonlinear_model: model.NonlinearModel,
                       X_op, input_op):
    """Linearise a non-linear model about an operating point to give
    a linear state space model

    Parameters
    ----------
    nonlinear_model : model.NonlinearModel
        The nonlinear model to be linearised
        
    X_op, input_op : ndarray
        The state and input around which the model should be linearised

    Returns
    -------
    linear_model : model.LinearModel
        The linear model of the system
    """

    def max_norm_error_close(f, tol=1e-8, x=0.1):
        """Takes in a function :math:`f` that takes in a
        parameter :math:`x` and evaluates
        :math:`\gamma = \frac{f(x) - f(-x)}{2x}` as an approximation
        of the gradient.
        It halves :math:`x` until :math:`e_k = || \gamma_k - \gamma_{k-1}||_\infty < tol`
        """
        gamma = (f(x) - f(-x))/2/x
        e = tol + 1
        while e > tol:
            x /= 2
            new_gamma = (f(x) - f(-x))/2/x
            e = numpy.max(numpy.abs(new_gamma - gamma))

        return gamma

    def f(x):
        new_vec = vec.copy()
        new_vec[k] += x

        if j == 0:
            return fun(new_vec, input_op)

        return fun(X_op, new_vec)

    matrices = [[[], []], [[], []]]
    for i, fun in enumerate([nonlinear_model.DEs, nonlinear_model.outputs]):
        for j, vec in enumerate([X_op, input_op]):
            matrix = []
            for k in range(len(vec)):
                gradient = max_norm_error_close(f)
                matrix.append(gradient)
            matrices[i][j] = numpy.array(matrix)

    (A, B), (C, D) = matrices
    linear_model = model.LinearModel(A.T, B.T, C.T, D.T)

    return linear_model
