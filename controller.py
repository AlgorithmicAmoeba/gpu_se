import numpy
import scipy.sparse
import scipy.stats
import scipy.optimize
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
            D x_k + e \ge 0
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
        d_i \mu_k + e_i &\ge k \sqrt{d_i \Sigma_k d_i^T}
        \quad \forall \; 0 \le k < N \; i

    where :math:`\mu` is the state estimated mean,
    :math:`k` is a constant that depends on :math:`p`,
    :math:`\Sigma_k` is the covariance prediction,
    :math:`\Sigma_0` is the state estimated covariance,
    and :math:`W` is the covariance for the state noise.

    Parameters
    ----------
    P, M : int
        The prediction and control horizon.
        :math:`P \ge M`

    Q, R : ndarray
        2D arrays of the diagonal tuning matrices

    D : ndarray
        2D array defining the linear constraints

    e : ndarray
        1D array defining the linear constraints

    lin_model : model.LinearModel
        Internal model for the controller

    Attributes
    -----------
    P, M : int
        The prediction and control horizon.
        :math:`P \ge M`

    Q, R : ndarray
        2D arrays of the diagonal tuning matrices

    D : ndarray
        1D array defining the linear constraints

    e : int
        Value defining the linear constraints

    lin_model : model.LinearModel
        Internal model for the controller

    p : float
        Constant that depends on :math:`p`
    """
    def __init__(self, P, M, Q, R, D, e,
                 lin_model: model.LinearModel,
                 r, x_bounds=None,
                 u_bounds=None, u_step_bounds=None,
                 p=0):
        assert P >= M

        self.P = P
        self.M = M
        self.Q = Q
        self.R = R
        self.D = D
        self.e = e
        self.model = lin_model
        self.p = p
        self.k = 0

        if self.p:
            self.k = self._inv_chi2()

        Nx = self.model.Nx
        Ni = self.model.Ni
        Ad = self.model.A
        Bd = self.model.B

        # Limit constraints
        if x_bounds is None:
            x_min = numpy.array([-numpy.inf] * Nx)
            x_max = numpy.array([numpy.inf] * Nx)
        else:
            x_min, x_max = [numpy.array(x) for x in zip(*x_bounds)]

        if u_bounds is None:
            u_min = numpy.array([-numpy.inf] * Ni)
            u_max = numpy.array([numpy.inf] * Ni)
        else:
            u_min, u_max = [numpy.array(x) for x in zip(*u_bounds)]

        if u_step_bounds is None:
            u_step_min = numpy.array([-numpy.inf] * Ni)
            u_step_max = numpy.array([numpy.inf] * Ni)
        else:
            u_step_min, u_step_max = [numpy.array(x) for x in zip(*u_step_bounds)]

        # Initial and reference states
        x0 = numpy.zeros(Nx)
        sigma0 = numpy.zeros((Nx, Nx))

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(P),u(0),...,u(M-1))
        # min x^T P x + q^T x
        # such that l <= Ax <= u

        P_matrix = scipy.sparse.block_diag([scipy.sparse.kron(scipy.sparse.eye(P + 1), Q),
                                            scipy.sparse.kron(scipy.sparse.eye(M), R)], format='csc')

        q_matrix = numpy.hstack([numpy.kron(numpy.ones(P + 1), -r.T @ Q),
                                 numpy.zeros(M * Ni)])

        # x_{k+1} = A x_k + B u_k
        Ax = scipy.sparse.kron(scipy.sparse.eye(P + 1), -scipy.sparse.eye(Nx))
        Ax += scipy.sparse.kron(scipy.sparse.eye(P + 1, k=-1), Ad)

        Bu = scipy.sparse.kron(
                scipy.sparse.vstack([
                    scipy.sparse.csc_matrix((1, M)),
                    scipy.sparse.eye(M),
                    scipy.sparse.hstack([scipy.sparse.csc_matrix((P-M, M-1)), numpy.ones((P-M, 1))])
                     ]),
                Bd)
        Aeq = scipy.sparse.hstack([Ax, Bu])
        leq = numpy.hstack([-x0, numpy.zeros(P * Nx)])
        ueq = leq

        # x_min <= x_k <= x_max and u_min <= u_k <= u_max for all k
        A_bound = scipy.sparse.eye((P + 1) * Nx + M * Ni)
        lower_bound = numpy.hstack([numpy.kron(numpy.ones(P + 1), x_min), numpy.kron(numpy.ones(M), u_min)])
        upper_bound = numpy.hstack([numpy.kron(numpy.ones(P + 1), x_max), numpy.kron(numpy.ones(M), u_max)])

        # u_step_min <= u_k - u_{k-1} <= u_step_max for all k >= 1
        if M >= 2:
            Ax_step_bound = scipy.sparse.csc_matrix(((M - 1)*Ni, (P+1)*Nx))
            Bu_step_bound = scipy.sparse.kron(scipy.sparse.hstack([
                -scipy.sparse.eye(M-1) + scipy.sparse.eye(M-1, k=1),
                scipy.sparse.vstack([
                    scipy.sparse.csc_matrix((M-2, 1)),
                    1
                ])
            ]), scipy.sparse.eye(Ni))

            A_step = scipy.sparse.hstack([Ax_step_bound, Bu_step_bound])
            lower_step_bound = numpy.kron(numpy.ones(M-1), u_step_min)
            upper_step_bound = numpy.kron(numpy.ones(M-1), u_step_max)
        else:
            A_step = scipy.sparse.csc_matrix((0, 0))
            lower_step_bound = numpy.array(0)
            upper_step_bound = numpy.array(0)

        # Stochastic constraints
        if D.ndim != 2:
            raise ValueError("D must be 2D")
        Nd = D.shape[0]
        Ax_stochastic = scipy.sparse.kron(scipy.sparse.eye(P + 1), D)
        Bu_stochastic = scipy.sparse.csc_matrix(((P + 1)*Nd, M*Ni))

        A_stochastic = scipy.sparse.hstack([Ax_stochastic, Bu_stochastic])
        lower_stochastic = numpy.full((P + 1) * Nd, 0)
        upper_stochastic = numpy.full((P + 1)*Nd, numpy.inf)
        self._stochastic_bounds(sigma0, lower_stochastic)

        # OSQP constraints
        A_matrix = scipy.sparse.vstack([Aeq, A_bound, A_step, A_stochastic], format='csc')
        self.lower = numpy.hstack([leq, lower_bound, lower_step_bound, lower_stochastic])
        self.upper = numpy.hstack([ueq, upper_bound, upper_step_bound, upper_stochastic])

        # Create an OSQP object
        self.prob = osqp.OSQP()

        # Setup workspace
        self.prob.setup(P_matrix, q_matrix, A_matrix, self.lower, self.upper, warm_start=True, verbose=False)

    def _inv_chi2(self):
        Nx = self.model.Nx

        def fun(k2):
            return abs(scipy.stats.chi2.cdf(k2, df=Nx) - self.p)

        ans = scipy.optimize.minimize_scalar(fun)
        k = numpy.sqrt(ans.x)
        return numpy.sqrt(k)

    def _stochastic_bounds(self, sigma0, array):
        if self.k == 0:
            return
        sigma = sigma0
        Nd = self.D.shape[0]
        for k in range(self.P+1):
            sigma = self.model.state_noise.cov() + self.model.A @ sigma @ self.model.A.T
            for i in range(Nd):
                d_i = self.D[i]
                e_i = self.e[i]
                bound = self.k * numpy.sqrt(d_i @ sigma @ d_i.T) - e_i
                array[k*Nd + i] = bound

    def step(self, x0, sigma0):
        # Added due to OSQP bug
        x0 = numpy.maximum(numpy.minimum(x0, 1e30), -1e30)
        sigma0 = numpy.maximum(numpy.minimum(sigma0, 1e30), -1e30)

        # Update initial state
        self.lower[:self.model.Nx] = -x0
        self.upper[:self.model.Nx] = -x0
        self.prob.update(l=self.lower, u=self.upper)

        # Update stochastic constraints
        Nd = self.D.shape[0]
        self._stochastic_bounds(sigma0, self.lower[-(self.P + 1) * Nd:])

        # Solve
        res = self.prob.solve()

        # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        # Apply first control input to the plant
        ctrl = res.x[-self.M * self.model.Ni: -(self.M - 1) * self.model.Ni]

        return ctrl
