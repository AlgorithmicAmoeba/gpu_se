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

    lin_model : model.LinearModel
        Internal model for the controller

    Attributes
    -----------
    P, M : int
        The prediction and control horizon.
        :math:`P \ge M`

    Q, R : ndarray
        2D arrays of the diagonal tuning matrices

    lin_model : model.LinearModel
        Internal model for the controller
    """
    def __init__(self, P, M, Q, R,
                 lin_model: model.LinearModel,
                 r, y_bounds=None,
                 u_bounds=None, u_step_bounds=None):
        assert P >= M

        self.P = P
        self.M = M
        self.Q = Q
        self.R = R
        self.model = lin_model

        Nx = self.model.Nx
        Ni = self.model.Ni
        No = self.model.No
        Ad = self.model.A
        Bd = self.model.B
        Cd = self.model.C
        Dd = self.model.D

        self.x_predicted = None

        # Limit constraints
        if y_bounds is None:
            y_min = numpy.full(No, -numpy.inf)
            y_max = numpy.full(No, numpy.inf)
        else:
            y_min, y_max = [numpy.asarray(y) for y in zip(*y_bounds)]

        if u_bounds is None:
            u_min = numpy.full(Ni, -numpy.inf)
            u_max = numpy.full(Ni, numpy.inf)
        else:
            u_min, u_max = [numpy.asarray(u) for u in zip(*u_bounds)]

        if u_step_bounds is None:
            u_step_min = numpy.full(Ni, -numpy.inf)
            u_step_max = numpy.full(Ni, numpy.inf)
        else:
            u_step_min, u_step_max = [numpy.array(u) for u in zip(*u_step_bounds)]

        # Initial state, initial input, and bias
        mu0 = numpy.zeros(Nx)
        um1 = numpy.zeros(Ni)
        b = numpy.zeros(No)

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(P),y(1), ..., y(P), u(0),...,u(M-1))
        # min x^T P x + q^T x
        # such that l <= Ax <= u

        H = scipy.sparse.block_diag([scipy.sparse.csc_matrix(((P + 1) * Nx, (P + 1) * Nx)),
                                     scipy.sparse.kron(scipy.sparse.eye(P), Q),
                                     scipy.sparse.csc_matrix((Ni, Ni)),
                                     scipy.sparse.kron(scipy.sparse.eye(M), R)], format='csc')

        qT = numpy.hstack([numpy.zeros((P + 1) * Nx),
                           numpy.kron(numpy.ones(P), -r.T @ Q),
                           numpy.zeros((M + 1) * Ni)])

        # Init
        A_init = scipy.sparse.vstack([
            scipy.sparse.hstack([
                scipy.sparse.eye(Nx),
                scipy.sparse.csc_matrix((Nx, P*Nx + P*No + (M+1)*Ni))
            ]),
            scipy.sparse.hstack([
                scipy.sparse.csc_matrix((Ni, (P+1)*Nx + P*No)),
                scipy.sparse.eye(Ni),
                scipy.sparse.csc_matrix((Ni, M*Ni))
            ])
        ])
        l_init = numpy.hstack([mu0, um1])
        u_init = l_init

        # d\mu_{k+1} = A d\mu_k + B du_k
        A_state_x = scipy.sparse.hstack([
            scipy.sparse.vstack([
                Ad - scipy.sparse.eye(Nx),
                scipy.sparse.csc_matrix(((P-1)*Nx, Nx))
            ]),
            -scipy.sparse.eye(P*Nx) + scipy.sparse.kron(
                scipy.sparse.eye(P, k=-1),
                Ad
            )
        ])

        A_state_u = scipy.sparse.vstack([
            scipy.sparse.hstack([
                scipy.sparse.vstack([
                    Bd,
                    scipy.sparse.csc_matrix(((M-1)*Nx, Ni))
                ]),
                scipy.sparse.kron(
                    scipy.sparse.eye(M),
                    Bd
                )
            ]),
            scipy.sparse.csc_matrix(((P-M)*Nx, (M+1)*Ni))
        ])

        A_state = scipy.sparse.hstack([
            A_state_x,
            scipy.sparse.csc_matrix((P*Nx, P*No)),
            A_state_u
        ])
        l_state = numpy.zeros(P*Nx)
        u_state = l_state

        # y_k = y_{k-1} + C d\mu_k + D du_k + b
        A_output_x = scipy.sparse.hstack([
            scipy.sparse.vstack([
                Cd,
                scipy.sparse.csc_matrix(((P - 1) * No, Nx))
            ]),
            scipy.sparse.kron(
                scipy.sparse.eye(P),
                Cd
            )
        ])

        A_output_y = scipy.sparse.kron(
            -scipy.sparse.eye(P) + scipy.sparse.eye(P, k=-1),
            scipy.sparse.eye(No)
        )

        A_output_u = scipy.sparse.vstack([
            scipy.sparse.hstack([
                scipy.sparse.vstack([
                    scipy.sparse.kron(
                        numpy.ones(2),
                        Dd
                    ),
                    scipy.sparse.csc_matrix(((M - 2) * No, 2*Ni))
                ]),
                scipy.sparse.kron(
                    scipy.sparse.eye(M-1),
                    Dd
                )
            ]),
            scipy.sparse.csc_matrix(((P - M + 1) * No, (M + 1) * Ni))
        ])

        A_output = scipy.sparse.hstack([
            A_output_x,
            A_output_y,
            A_output_u
        ])
        l_output = numpy.kron(
            -numpy.hstack([
                2,
                numpy.ones(P-1)
            ]),
            -b
        )
        u_output = l_output

        # y_min <= y_k <= y_max
        A_output_ineq = scipy.sparse.hstack([
            scipy.sparse.csc_matrix((P*No, (P+1)*Nx)),
            scipy.sparse.eye(P*No),
            scipy.sparse.csc_matrix((P*No, (M+1)*Ni))
        ])
        l_output_ineq = numpy.kron(numpy.ones(P), y_min)
        u_output_ineq = numpy.kron(numpy.ones(P), y_max)

        # du_min <= du_k <= du_max
        A_input_steps = scipy.sparse.hstack([
            scipy.sparse.csc_matrix((M*Ni, (P+1)*Nx + P*No + Ni)),
            scipy.sparse.eye(M*Ni)
        ])
        l_input_steps = numpy.kron(numpy.ones(M), u_step_min)
        u_input_steps = numpy.kron(numpy.ones(M), u_step_max)

        # u_min <= u_k <= u_max
        A_input_ineq = scipy.sparse.hstack([
            scipy.sparse.csc_matrix((M*Ni, (P+1)*Nx + P*No)),
            scipy.sparse.kron(
                numpy.ones((M, 1)),
                scipy.sparse.eye(Ni)
            ),
            scipy.sparse.kron(
                scipy.sparse.csc_matrix(numpy.tril(numpy.ones((M, M)))),
                numpy.eye(Ni)
            )
        ])
        l_input_ineq = numpy.kron(numpy.ones(M), u_min)
        u_input_ineq = numpy.kron(numpy.ones(M), u_max)

        # OSQP constraints
        self.A_matrix = scipy.sparse.vstack([A_init, A_state, A_output, A_output_ineq, A_input_steps, A_input_ineq],
                                            format='csc')
        self.lower = numpy.hstack([l_init, l_state, l_output, l_output_ineq, l_input_steps, l_input_ineq])
        self.upper = numpy.hstack([u_init, u_state, u_output, u_output_ineq, u_input_steps, u_input_ineq])

        # Create an OSQP object
        self.prob = osqp.OSQP()

        # Setup workspace
        self.prob.setup(H, qT, self.A_matrix, self.lower, self.upper, warm_start=True,
                        verbose=False, eps_abs=1e-10, eps_rel=1e-5, eps_prim_inf=1e-10,
                        max_iter=100000)

    def step(self, mu0, um1, y0):
        # Added due to OSQP bug
        mu0 = numpy.maximum(numpy.minimum(mu0, 1e30), -1e30)

        Nx = self.model.Nx
        Ni = self.model.Ni
        No = self.model.No

        # Update initial state
        self.lower[:Nx+Ni] = numpy.hstack([mu0, um1])
        self.upper[:Nx+Ni] = numpy.hstack([mu0, um1])

        # Update bias
        if self.x_predicted is not None:
            b = y0 - self.model.C @ self.x_predicted
        else:
            b = numpy.zeros_like(y0)

        n = Nx + Ni + self.P*Nx
        self.lower[n: n + self.P*No] = numpy.tile(-b, self.P)
        self.upper[n: n + self.P*No] = numpy.tile(-b, self.P)

        self.prob.update(l=self.lower, u=self.upper)

        # Solve
        res = self.prob.solve()

        # Check solver status
        if res.info.status_val not in [1]:
            raise ValueError(f'OSQP did not solve the problem! Status: {res.info.status}')

        # Apply first control input to the plant
        m = (self.P+1)*Nx + self.P*No + Ni
        ctrl = res.x[m: m + Ni]

        self.x_predicted = mu0 + res.x[Nx:2*Nx]

        return ctrl
