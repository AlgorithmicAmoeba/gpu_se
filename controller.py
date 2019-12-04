import cvxpy
import model


class SMPC:
    r"""A deterministic reformulation of a linear chance constrained MPC
    that uses a discrete linear state space model of the system.
    The original stochastic problem is given by:

    .. math::
        \displaystyle \underset{u}{\min}
        \quad
        & \mathbb{E}
        \left[
            \sum_{i=0}^{P-1}
                \left(
                e_i^T Q e_i + u_i^T R u_i
                \right)
        \right] \\
        x_{k+1} &= A x_k + B u_k \\
        e_k &= r - x_k \\
        P
        \left[
            d x_k + e \ge 0
        \right]
        &\ge p
        \quad \forall \; 0 \le k < N

    where :math:`Q` and :math:`R` are tuning parameters,
    :math:`A` and :math:`B`
    are state space matrices,
    :math:`r` is the set point,
    and :math:`P[c] \ge p`
    ensures that the probability of :math:`c` is larger than :math:`p`.

    The reformulation is done by Wilken (2015) and the deterministic problem
    is given by:

    .. math::
        \displaystyle \underset{u}{\min}
        \quad
        & \sum_{i=0}^{P-1}
        \left(
        e_i^T Q e_i + u_i^T R u_i
        \right)	\\
        \mu_{k+1} &= A \mu_k + B u_k \\
        e_k &= r - \mu_k \\
        \Sigma_{k+1} &= A \Sigma_k A^T + W
        \quad \forall \; 0 \le k < N \\
        d \mu_k + e &\ge k \sqrt{d \Sigma_k d^T}
        \quad \forall \; 0 \le k < N

    where :math:`\mu` is the state estimated mean,
    :math:`k` is a constant that depends on :math:`p`,
    :math:`\Sigma_k` is the covariance prediction,
    :math:`\Sigma_0` is the state estimated covariance,
    and :math:`W` is the covariance for the state noise.

    For convenience we let :math:`c =  k \sqrt{d \Sigma_k d^T} - e `
    in the code.

    Parameters
    ----------
    P, M : int
        The prediction and control horizon.
        :math:`P \ge M`

    Q, R : ndarray
        2D arrays of the diagonal tuning matrices

    d, e : ndarray
        1D arrays defining the linear constraints

    lin_model : model.LinearModel
        Internal model for the controller

    Attributes
    -----------
    P, M : int
        The prediction and control horizon.
        :math:`P \ge M`

    Q, R : ndarray
        2D arrays of the diagonal tuning matrices

    d, e : ndarray
        1D arrays defining the linear constraints

    lin_model : model.LinearModel
        Internal model for the controller
    """
    def __init__(self, P, M, Q, R, d, e,
                 lin_model: model.LinearModel):
        assert P >= M

        self.P = P
        self.M = M
        self.Q = Q
        self.R = R
        self.d = d
        self.e = e
        self.model = lin_model

        self._mu0 = cvxpy.Parameter(self.model.Nx)
        self._u0 = cvxpy.Parameter(self.model.Ni)
        self._cs = cvxpy.Parameter(self.P)

        us = cvxpy.Variable(self.M, self.model.Ni)
        mus = cvxpy.Variable(self.P, self.model.Nx)

        # Objective function
        obj = cvxpy.sum([cvxpy.quad_form(mu, self.Q) for mu in mus])
        obj += cvxpy.sum([cvxpy.quad_form(u, self.R) for u in us])
        obj += cvxpy.quad_form(us[-1], R) * (self.P - self.M)
        min_obj = cvxpy.Minimize(obj)

        # State constraints
        state_constraints = [False] * self.P
        state_constraints[0] = mus[0] - (self.model.A @ self._mu0 + self.model.B @ self._u0)
        for i in range(1, self.P):
            state_constraints[i] = mus[i] == self.model.A @ mus[i - 1] + self.model.B @ us[i - 1]

        # Linear constraints
        lin_constraints = [self.d @ mu >= c for mu, c in zip(mus, self._cs)]

        # All constraints
        constraints = state_constraints + lin_constraints

        self._problem = cvxpy.Problem(min_obj, constraints)
