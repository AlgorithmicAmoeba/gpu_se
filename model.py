import noise


class LinearModel:
    """Structured data for a discrete state space linear model with noise

    .. math:
        x_{k+1} &= A x_k + B u_k + w_k\\
        y_k &= C x_k + D u_k + v_k

    where :math:`w_k` and :math:`v_k` are additive noise sampled from some distribution

    Parameters
    ----------
    A, B, C, D : ndarray
        2D array containing the relevant state space matrices

    state_noise, measurement_noise : noise.Noise
        Objects containing state and measurement noise information

    Attributes
    -----------
    A, B, C, D : ndarray
        2D array containing the relevant state space matrices

    state_noise, measurement_noise : noise.Noise
        Objects containing state and measurement noise information

    """
    def __init__(self, A, B, C, D,
                 state_noise: noise.Noise,
                 measurement_noise: noise.Noise):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.w = self.state_noise = state_noise
        self.v = self.measurement_noise = measurement_noise

