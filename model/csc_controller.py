import numpy
import scipy.sparse
import cvxpy


def mpc_var(x0, cov0, N, A, B, b, aline, bline, cline, QQ, RR,
            ysp, usp, lim_u, lim_step_u, Q, k, growvar=True):
    """return the MPC control input using a linear system"""

    nx, nu = B.shape
    QN = QQ
    d_T = numpy.matrix(numpy.hstack([aline, bline]))

    P = scipy.sparse.block_diag([scipy.sparse.kron(scipy.sparse.eye(N), QQ), QN,
                                 scipy.sparse.kron(scipy.sparse.eye(N), RR)])

    q = numpy.hstack([numpy.kron(numpy.ones(N), -2*QQ @ ysp), -2*QN @ ysp,
                      numpy.kron(numpy.ones(N), -RR @ usp)])

    # Handling of mu_(k+1) = A @ mu_k + B @ u_k
    temp1 = scipy.sparse.block_diag([scipy.sparse.kron(scipy.sparse.eye(N+1), -numpy.eye(nx))])
    temp2 = scipy.sparse.block_diag([scipy.sparse.kron(scipy.sparse.eye(N+1, k=-1), A)])
    AA = temp1 + temp2

    temp1 = scipy.sparse.vstack([numpy.zeros([nx, N*nu]), scipy.sparse.kron(scipy.sparse.eye(N), B)])
    AA = scipy.sparse.hstack([AA, temp1])

    # Handling of d.T mu_k > k sqrt(d.T @ Sigma_k @ d) - e
    # temp1 = scipy.sparse.hstack([numpy.zeros([N, nx]), scipy.sparse.kron(scipy.sparse.eye(N), d_T)])
    # temp2 = numpy.zeros([N, N*nu])
    # temp3 = scipy.sparse.hstack([temp1, temp2])
    # GG = temp3

    # Handling of -limstep <= u <= limstepu
    temp1 = numpy.zeros([N-1, (N + 1) * nx])
    temp2 = numpy.zeros([N-1, nu])
    temp2[0][:nu] = 1
    temp3 = scipy.sparse.kron(scipy.sparse.eye(N-1), -numpy.eye(nu))
    temp3 += scipy.sparse.kron(scipy.sparse.eye(N-1, k=-1), numpy.eye(nu))
    temp4 = scipy.sparse.hstack([temp1, temp2, temp3])
    GG = temp4
    # GG = scipy.sparse.vstack([GG, temp4])

    # Handling of -limu <= u <= limu
    temp1 = numpy.zeros([N, (N+1)*nx])
    temp2 = scipy.sparse.kron(scipy.sparse.eye(N), numpy.eye(nu))
    temp3 = scipy.sparse.hstack([temp1, temp2])
    GG = scipy.sparse.vstack([GG, temp3])

    # e = cline + d_T @ b
    #
    # sigmas = Q + A @ cov0 @ A.T
    # limits = numpy.zeros(N)
    # for i in range(N):
    #     rsquared = d_T @ sigmas @ d_T.T
    #     r = - e + numpy.sqrt(k * rsquared)
    #     if growvar:
    #         sigmas = Q + A @ sigmas @ A.T
    #     limits[i] = r

    bb = numpy.hstack([-x0, numpy.zeros(N * nx)])
    L = numpy.hstack([[-lim_step_u] * (N - 1), [-lim_u] * N])
    U = numpy.hstack([[lim_step_u] * (N - 1), [lim_u] * N])

    n = q.shape[0]
    x = cvxpy.Variable(n)
    P = cvxpy.Constant(P)
    objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, P) + q * x)
    constraints = [GG * x <= U, GG * x >= L, AA * x == bb]

    prob = cvxpy.Problem(objective, constraints)

    try:
        prob.solve(solver='OSQP')
    except:
        return None
    if prob.status != "optimal":
        print(prob.status)
        print(prob.is_qp())
        return None
    res = numpy.array(x.value).reshape((n,))

    # check_constraints(GG, L, U, res, b)
    return res[(N+1)*nx: (N+1)*nx+nu]
