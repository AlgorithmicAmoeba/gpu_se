import cvxpy


class ModelPredictiveController:
    """A linear constrained MPC that uses a discrete linear state space model of the system:

    .. math::
        \displaystyle \underset{u}{\min}
        \quad
        \sum_{i=0}^{P-1}
            \left[
            x_i^T Q x_i + u_i^T R u_i
            \right] \\
        x_{k+1} &= A x_k + B u_k
    """
