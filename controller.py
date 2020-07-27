import numpy
import scipy.sparse
import scipy.stats
import scipy.optimize
import model
import osqp


class MPC:
    r"""A deterministic reformulation of a linear chance constrained MPC
        that uses a discrete linear state space model of the system.
        The reformulation is given as:

        .. math::
            \underset{\mathbf{\Delta u}}{\min}
            \quad
            & \frac{1}{2}
            \sum_{k=1}^{P}
            \left(
            e_k^T Q e_k
            \right)
            +
            \frac{1}{2}
            \sum_{k=0}^{M-1}
            \left(
            \Delta u_k^T R \Delta u_k
            \right)	\\
            \Delta \mu_1 &= A x_0 + B (u_{-1} + \Delta u_0) - x_0 \\
            \Delta \mu_{k+1} &= A \Delta \mu_k + B \Delta u_k \\
            y_0 &= C \mu_0 + D (u_{-1} + \Delta u_0) + b \\
            y_k &= y_{k-1} + C \Delta x_k + D \Delta u_k + b \\
            e_k &= r - y_k \\
            y_{\min} \le &y_k \le y_{\max} \\
            \Delta u_{\min} \le &\Delta u_k \le \Delta u_{\max} \\
            u_{\min} \le &u_k \le u_{\max} \\

        where :math:`\mu_0` is the state estimated mean.

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
    def __init__(self, P, M, Q, R, lin_model, ysp,
                 y_bounds=None, u_bounds=None, u_step_bounds=None):
        self.P = P
        self.M = M
        self.Q = Q
        self.R = R
        self.model = lin_model
        self.ysp = ysp

        Nx, Ni = self.model.B.shape
        No, _ = self.model.C.shape

        x0 = numpy.zeros(Nx)
        um1 = numpy.zeros(Ni)

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

        self.H = scipy.sparse.block_diag([
            scipy.sparse.csc_matrix(((P + 1) * Nx, (P + 1) * Nx)),
            scipy.sparse.kron(scipy.sparse.eye(P), self.Q),
            scipy.sparse.csc_matrix((Ni, Ni)),
            scipy.sparse.kron(scipy.sparse.eye(M + 1), self.R)
        ], format='csc')

        self.q = numpy.hstack([
            numpy.zeros((P + 1) * Nx),
            numpy.kron(numpy.ones(P), -self.Q @ ysp),
            numpy.zeros((M + 2)*Ni)
        ])

        # Handling of initial condition um1
        A_um1_init = scipy.sparse.hstack([
            scipy.sparse.csc_matrix((Ni, (P + 1) * Nx + P * No)),
            scipy.sparse.eye(Ni),
            scipy.sparse.csc_matrix((Ni, (M + 1) * Ni))
        ])

        l_um1_init = um1

        # Handling of mu_(k+1) = A @ mu_k + B @ u_k
        A_state_x = scipy.sparse.hstack([
            scipy.sparse.vstack([
                -scipy.sparse.eye(Nx),
                self.model.A - scipy.sparse.eye(Nx),
                scipy.sparse.csc_matrix(((P-1)*Nx, Nx))
            ]),
            scipy.sparse.vstack([
                scipy.sparse.csc_matrix((Nx, P*Nx)),
                scipy.sparse.kron(
                    scipy.sparse.eye(P, k=-1),
                    self.model.A
                ) - scipy.sparse.eye(P*Nx)
            ])
        ])

        A_state_u = scipy.sparse.vstack([
            scipy.sparse.csc_matrix((Nx, (M + 2)*Ni)),
            scipy.sparse.kron(
                scipy.sparse.hstack([
                    scipy.sparse.csc_matrix(([1], ([0], [0])), shape=(M, 1)),
                    scipy.sparse.eye(M),
                    scipy.sparse.csc_matrix((M, 1))
                ]),
                self.model.B
            ),
            scipy.sparse.csc_matrix(((P - M) * Nx, (M + 2) * Ni))
        ])

        A_state = scipy.sparse.hstack([
            A_state_x,
            scipy.sparse.csc_matrix(((P + 1) * Nx, P * No)),
            A_state_u
        ])

        b_state = numpy.hstack([-x0, numpy.zeros(P * Nx)])

        # Handling of y_k = C @ mu_k + D u_k
        A_output_x = scipy.sparse.kron(
            scipy.sparse.hstack([
                scipy.sparse.csc_matrix(([1], ([0], [0])), shape=(P, 1)),
                scipy.sparse.eye(P)
            ]),
            self.model.C
        )

        A_output_y = -scipy.sparse.eye(P * No) + scipy.sparse.eye(P * No, k=-No)

        A_output_u = scipy.sparse.vstack([
            scipy.sparse.kron(
                scipy.sparse.hstack([
                    scipy.sparse.csc_matrix(([1, 1], ([0, 0], [0, 1])), shape=(M, 2)),
                    scipy.sparse.eye(M)
                ]),
                self.model.D
            ),
            scipy.sparse.csc_matrix(((P-M)*No, (M+2)*Ni))
        ])

        A_output = scipy.sparse.hstack([A_output_x, A_output_y, A_output_u])

        b_output = numpy.zeros(P * No)

        # y_min <= y_k <= y_max
        A_output_ineq = scipy.sparse.hstack([
            scipy.sparse.csc_matrix((P * No, (P + 1) * Nx)),
            scipy.sparse.eye(P * No),
            scipy.sparse.csc_matrix((P * No, (M + 2) * Ni))
        ])
        l_output_ineq = numpy.kron(numpy.ones(P), y_min)
        u_output_ineq = numpy.kron(numpy.ones(P), y_max)

        # du_min <= du_k <= du_max
        A_input_steps = scipy.sparse.hstack([
            scipy.sparse.csc_matrix(((M+1) * Ni, (P + 1) * Nx + P * No + Ni)),
            scipy.sparse.eye((M+1) * Ni)
        ])
        l_input_steps = numpy.kron(numpy.ones(M+1), u_step_min)
        u_input_steps = numpy.kron(numpy.ones(M+1), u_step_max)

        # u_min <= u_k <= u_max
        self.A_input_ineq = scipy.sparse.hstack([
            scipy.sparse.csc_matrix(((M+1) * Ni, (P + 1) * Nx + P * No)),
            scipy.sparse.kron(
                numpy.ones((M+1, 1)),
                scipy.sparse.eye(Ni)
            ),
            scipy.sparse.kron(
                scipy.sparse.csc_matrix(numpy.tril(numpy.ones((M+1, M+1)))),
                numpy.eye(Ni)
            )
        ])
        self.A_input_ineq = scipy.sparse.hstack([
            scipy.sparse.csc_matrix((Ni, (P + 1) * Nx + P * No)),
            scipy.sparse.kron(
                numpy.ones((1, 2)),
                scipy.sparse.eye(Ni)
            ),
            scipy.sparse.csc_matrix((Ni, M*Ni)),
        ])
        self.l_input_ineq = numpy.kron(numpy.ones(1), u_min)
        self.u_input_ineq = numpy.kron(numpy.ones(1), u_max)

        self.A_matrix = scipy.sparse.vstack([A_um1_init, A_state, A_output,
                                             A_output_ineq, A_input_steps, self.A_input_ineq])

        self.l_matrix = numpy.hstack([l_um1_init, b_state, b_output,
                                      l_output_ineq, l_input_steps, self.l_input_ineq])

        self.u_matrix = numpy.hstack([l_um1_init, b_state, b_output,
                                      u_output_ineq, u_input_steps, self.u_input_ineq])

        # Create an OSQP object
        self.prob = osqp.OSQP()

        # Setup workspace
        self.A_matrix = scipy.sparse.csc_matrix(self.A_matrix)
        self.prob.setup(self.H, self.q, self.A_matrix, self.l_matrix, self.u_matrix, verbose=False)

        self.y_predicted = None

    def step(self, x0, um1, y0):
        """return the MPC control input using a linear system"""
        # Added due to OSQP bug
        x0 = numpy.maximum(numpy.minimum(x0, 1e10), -1e10)
        um1 = numpy.maximum(numpy.minimum(um1, 1e10), -1e10)
        y0 = numpy.maximum(numpy.minimum(y0, 1e10), -1e10)

        Nx, Ni = self.model.B.shape
        No, _ = self.model.C.shape

        self.l_matrix[:Ni] = um1
        self.l_matrix[Ni:Ni+Nx] = -x0

        self.u_matrix[:Ni] = um1
        self.u_matrix[Ni:Ni + Nx] = -x0

        if self.y_predicted is not None:
            bias = y0 - self.y_predicted
        else:
            bias = numpy.zeros_like(y0)

        self.l_matrix[Ni + (self.P+1)*Nx: Ni + (self.P+1)*Nx + self.P*No] = numpy.tile(-bias, self.P)
        self.u_matrix[Ni + (self.P+1)*Nx: Ni + (self.P+1)*Nx + self.P*No] = numpy.tile(-bias, self.P)

        self.prob.update(l=self.l_matrix, u=self.u_matrix)

        # Solve
        res = self.prob.solve()

        # Check solver status
        if res.info.status_val not in [1]:
            raise ValueError(f'OSQP did not solve the problem! Status: {res.info.status}')

        # Apply first control input to the plant
        m = (self.P + 1) * Nx + self.P * No + Ni
        ctrl = res.x[m: m + Ni] + um1

        self.y_predicted = res.x[(self.P+1)*Nx:(self.P+1)*Nx + No] - bias

        return ctrl
