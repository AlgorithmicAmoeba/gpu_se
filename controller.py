import numpy
import scipy.sparse
import cvxpy
import cvxpy.expressions.expression
import model
import osqp


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
        y_k &= C x_k + D u_k \\
        e_k &= r - y_k \\
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
        y_k &= C \mu_k + D u_k \\
        e_k &= r - y_k \\
        \Sigma_{k+1} &= A \Sigma_k A^T + W
        \quad \forall \; 0 \le k < N \\
        d \mu_k + e &\ge k \sqrt{d \Sigma_k d^T}
        \quad \forall \; 0 \le k < N

    where :math:`\mu` is the state estimated mean,
    :math:`k` is a constant that depends on :math:`p`,
    :math:`\Sigma_k` is the covariance prediction,
    :math:`\Sigma_0` is the state estimated covariance,
    and :math:`W` is the covariance for the state noise.

    For convenience we let :math:`c =  k \sqrt{d \Sigma_k d^T} - e`
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

    d : ndarray
        1D array defining the linear constraints

    e : int
        Value defining the linear constraints

    lin_model : model.LinearModel
        Internal model for the controller

    k : float
        Constant that depends on :math:`p`
    """
    def __init__(self, P, M, Q, R, d,
                 e: float,
                 lin_model: model.LinearModel,
                 k):
        assert P >= M

        self.P = P
        self.M = M
        self.Q = Q
        self.R = R
        self.d = d
        self.e = e
        self.model = lin_model
        self.k = k

        self._mu0 = cvxpy.Parameter(self.model.Nx)
        self._u0 = cvxpy.Parameter(self.model.Ni)
        self._sigma0 = cvxpy.Parameter((self.model.Nx, self.model.Nx))
        self._r = cvxpy.Parameter(self.model.No)

        self._us = cvxpy.Variable((self.M, self.model.Ni))
        mus = []

        # Objective function
        obj = self._r @ numpy.zeros(self.model.No)
        mu = self._mu0
        for i in range(1, self.P):
            us_indx = i-1 if i-1 < M else -1
            u = self._us[us_indx]
            mu = self.model.A @ mu + self.model.B @ u
            mus.append(mu)

            # y = self.model.C @ mu + self.model.D @ u
            e = self._r - mu
            obj += cvxpy.quad_form(e, self.Q)
            obj += cvxpy.quad_form(u, self.R)
        min_obj = cvxpy.Minimize(obj)

        # Linear constraints
        # First let us calculate future sigma values
        sigmas = [numpy.array([])] * self.P
        sigmas[0] = self.model.state_noise.cov() + self.model.A @ self._sigma0 @ self.model.A.T
        for i in range(1, self.P):
            sigmas[i] = self.model.state_noise.cov() + self.model.A @ sigmas[i - 1] @ self.model.A.T

        # Calculate c values
        cs = [None]*self.P
        for i in range(self.P):
            cs[i] = self.k * cvxpy.sqrt(cvxpy.quad_form(self.d, sigmas[i])) - self.e

        lin_constraints = [self.d @ mu >= c for mu, c in zip(mus, cs)]

        # All constraints
        constraints = lin_constraints

        self._problem = cvxpy.Problem(min_obj, constraints)
        assert self._problem.is_qp()

    def step(self, mu0, u0, sigma0, r):
        self._mu0.value = mu0.copy()
        self._u0.value = u0.copy()
        self._sigma0.value = sigma0.copy()
        self._r.value = r.copy()

        self._problem.solve()

        u_now = self._us[0].value.copy()

        return u_now


class SMPC2:
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
        y_k &= C x_k + D u_k \\
        e_k &= r - y_k \\
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
        y_k &= C \mu_k + D u_k \\
        e_k &= r - y_k \\
        \Sigma_{k+1} &= A \Sigma_k A^T + W
        \quad \forall \; 0 \le k < N \\
        d \mu_k + e &\ge k \sqrt{d \Sigma_k d^T}
        \quad \forall \; 0 \le k < N

    where :math:`\mu` is the state estimated mean,
    :math:`k` is a constant that depends on :math:`p`,
    :math:`\Sigma_k` is the covariance prediction,
    :math:`\Sigma_0` is the state estimated covariance,
    and :math:`W` is the covariance for the state noise.

    For convenience we let :math:`c =  k \sqrt{d \Sigma_k d^T} - e`
    in the code.

    Parameters
    ----------
    P_matrix, M : int
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
    P_matrix, M : int
        The prediction and control horizon.
        :math:`P \ge M`

    Q, R : ndarray
        2D arrays of the diagonal tuning matrices

    d : ndarray
        1D array defining the linear constraints

    e : int
        Value defining the linear constraints

    lin_model : model.LinearModel
        Internal model for the controller

    k : float
        Constant that depends on :math:`p`
    """
    def __init__(self, P, M, Q, R, d, e,
                 lin_model: model.LinearModel,
                 k, r):
        assert P >= M

        self.P = P
        self.M = M
        self.Q = Q
        self.R = R
        self.d = d
        self.e = e
        self.model = lin_model
        self.k = k

        nx = self.model.Nx
        nu = self.model.Ni
        Ad = self.model.A
        Bd = self.model.B

        # Constraints
        umin = numpy.array([-numpy.inf]*nu)
        umax = numpy.array([numpy.inf]*nu)
        xmin = numpy.array([-numpy.inf]*nx)
        xmax = numpy.array([numpy.inf]*nx)

        # Objective function
        QN = Q

        # Initial and reference states
        x0 = numpy.zeros(nx)

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(P),u(0),...,u(P-1))
        # - quadratic objective
        P_matrix = scipy.sparse.block_diag([scipy.sparse.kron(scipy.sparse.eye(P), Q), QN,
                                            scipy.sparse.kron(scipy.sparse.eye(P), R)], format='csc')
        # - linear objective
        q = numpy.hstack([numpy.kron(numpy.ones(P), -Q.dot(r)), -QN.dot(r),
                          numpy.zeros(P * nu)])
        # - linear dynamics
        Ax = scipy.sparse.kron(scipy.sparse.eye(P + 1),
                               -scipy.sparse.eye(nx)) + scipy.sparse.kron(scipy.sparse.eye(P + 1, k=-1), Ad)
        Bu = scipy.sparse.kron(scipy.sparse.vstack([scipy.sparse.csc_matrix((1, P)), scipy.sparse.eye(P)]), Bd)
        Aeq = scipy.sparse.hstack([Ax, Bu])
        leq = numpy.hstack([-x0, numpy.zeros(P * nx)])
        ueq = leq

        # - input and state constraints
        Aineq = scipy.sparse.eye((P + 1) * nx + P * nu)
        lineq = numpy.hstack([numpy.kron(numpy.ones(P + 1), xmin), numpy.kron(numpy.ones(P), umin)])
        uineq = numpy.hstack([numpy.kron(numpy.ones(P + 1), xmax), numpy.kron(numpy.ones(P), umax)])

        # - OSQP constraints
        A = scipy.sparse.vstack([Aeq, Aineq], format='csc')
        self.lower = numpy.hstack([leq, lineq])
        self.u = numpy.hstack([ueq, uineq])

        # Create an OSQP object
        self.prob = osqp.OSQP()

        # Setup workspace
        self.prob.setup(P_matrix, q, A, self.lower, self.u, warm_start=True, verbose=False)

    def step(self, x0):
        # Update initial state
        self.lower[:self.model.Nx] = -x0
        self.u[:self.model.Nx] = -x0
        self.prob.update(l=self.lower, u=self.u)

        # Solve
        res = self.prob.solve()

        # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        # Apply first control input to the plant
        ctrl = res.x[-self.P * self.model.Ni:-(self.P - 1) * self.model.Ni]

        return ctrl
