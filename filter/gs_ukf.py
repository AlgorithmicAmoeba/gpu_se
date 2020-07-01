import numpy
import sim_base


class GaussianSumUnscentedKalmanFilter:
    """Implements a Gaussian Sum Unscented Kalman Filter filter algorithm"""

    def __init__(self, f, g, N_gaussians, x0, state_pdf, measurement_pdf):
        self.f = f
        self.g = g
        self.N_gaussians = int(N_gaussians)

        self.means = x0.draw(N_gaussians)

        average_cov = numpy.average(state_pdf.covariances, axis=0)
        self.covariances = numpy.repeat(average_cov[None, :, :], N_gaussians, axis=0)

        self.weights = numpy.full(N_gaussians, 1 / N_gaussians, dtype=numpy.float32)

        self.state_pdf = state_pdf
        self.measurement_pdf = measurement_pdf

        self.Nx = self.means.shape[1]
        self.Ny = measurement_pdf.draw().shape[0]
        self.N_sigmas = 2 * self.Nx + 1

        # weights calculated such that:
        # 1) w_mu + 2*n*w_sigma = 1
        # 2) w_mu / w_sigma = 0.4 / 0.25 ~ Normal_pdf(0) / Normal_pdf(sigma)
        self.w_sigma = numpy.full(self.N_sigmas, 1 / (2 * self.Nx + 8 / 5))
        self.w_sigma[0] = 1 / (1 + 5/4*self.Nx)

    def _get_sigma_points(self):
        stds = numpy.linalg.cholesky(self.covariances).swapaxes(1, 2)
        sigmas = numpy.repeat(self.means[:, None, :], self.N_sigmas, axis=1)
        sigmas[:, 1:self.Nx+1, :] += stds
        sigmas[:, self.Nx+1:, :] -= stds

        return sigmas

    def predict(self, u, dt):
        sigmas = self._get_sigma_points()

        # Move the sigma points through the state transition function
        for gaussian in range(self.N_gaussians):
            for sigma in range(self.N_sigmas):
                sigmas[gaussian, sigma] = self.f(sigmas[gaussian, sigma], u, dt) + self.state_pdf.draw()

        self.means = numpy.average(sigmas, axis=1, weights=self.w_sigma)
        sigmas -= self.means[:, None, :]
        self.covariances = sigmas.swapaxes(1, 2) @ (sigmas * self.w_sigma[:, None])

    def update(self, u, z):
        # Local Update
        sigmas = self._get_sigma_points()
        etas = numpy.zeros((self.N_gaussians, self.N_sigmas, self.Ny))

        # Move the sigma points through the state observation function
        for gaussian in range(self.N_gaussians):
            for sigma in range(self.N_sigmas):
                etas[gaussian, sigma] = self.g(sigmas[gaussian, sigma], u)

        # Compute the Kalman gain
        eta_means = numpy.average(etas, axis=1, weights=self.w_sigma)
        sigmas -= self.means[:, None, :]
        etas -= eta_means[:, None, :]

        P_xys = sigmas.swapaxes(1, 2) @ (etas * self.w_sigma[:, None])
        P_yys = etas.swapaxes(1, 2) @ (etas * self.w_sigma[:, None])
        P_yy_invs = numpy.linalg.inv(P_yys)
        Ks = P_xys @ P_yy_invs

        # Use the gain to update the means and covariances
        es = z - eta_means
        self.means += (Ks @ es[:, :, None]).squeeze()
        # Dimensions from paper do not work, use corrected version
        self.covariances -= Ks @ P_yys @ Ks.swapaxes(1, 2)

        # Global Update
        y_means = numpy.zeros((self.N_gaussians, self.Ny))

        # Move the means through the state observation function
        for gaussian in range(self.N_gaussians):
            y_means[gaussian] = self.g(self.means[gaussian], u)

        glob_es = z - y_means
        consts = (glob_es[:, None, :] @ P_yy_invs @ glob_es[:, :, None]).squeeze()
        self.weights *= numpy.exp(consts)

    def resample(self):
        cumsum = numpy.cumsum(self.weights)
        cumsum /= cumsum[-1]

        sample_index_result = numpy.zeros(self.N_gaussians, dtype=numpy.int64)
        r = numpy.random.rand()
        k = 0

        for i in range(self.N_gaussians):
            u = (i + r) / self.N_gaussians
            while cumsum[k] < u:
                k += 1
            sample_index_result[i] = k

        self.means = self.means[sample_index_result]
        self.weights = numpy.full(self.N_gaussians, 1 / self.N_gaussians)


def my_cov(x, y, w=None):
    avgx = numpy.average(x, weights=w, axis=1)
    avgy = numpy.average(y, weights=w, axis=1)

    X = x - avgx[:, None, :]
    Y = y - avgy[:, None, :]
    if w is None:
        return X.T @ Y
    return X.swapaxes(1, 2) @ (Y*w[:, None])


def paper_cov(x, y, w):
    summed = 0
    avgx = numpy.average(x, weights=w, axis=0)
    avgy = numpy.average(y, weights=w, axis=0)

    for i in range(x.shape[0]):
        X = numpy.atleast_2d(x[i] - avgx)
        Y = numpy.atleast_2d(y[i] - avgy)
        summed += w[i] * X.T @ Y

    return summed


def paper_cov_adapted(x, y, w):
    avgx = numpy.average(x, weights=w, axis=1)
    avgy = numpy.average(y, weights=w, axis=1)

    X = x - avgx[None, :]
    Y = y - avgy[None, :]

    # for i in range(x.shape[0]):
    #     Xi = numpy.atleast_2d(X[i])
    #     Yi = numpy.atleast_2d(Y[i])
    #     summed += w[i] * Xi.T @ Yi
    summed = X.T @ (Y * w[:, None])

    return summed


def main():
    Ng, Ns, Nx = 5, 10, 2

    x, y = numpy.random.randint(1, 100, (Ng, Ns, Nx)), numpy.random.randint(1, 100, (Ng, Ns, Nx))
    w = numpy.random.random(Ns)
    w /= numpy.sum(w)

    mcov = my_cov(x, y, w)

    # ncov = numpy.cov(x, y, aweights=w, bias=True)
    ncov = numpy.array([paper_cov(x[i], y[i], w) for i in range(Ng)])

    print('A', mcov)
    print('B', ncov)
    print('sub', abs(mcov - ncov))
    print('sum', numpy.sum(numpy.abs(ncov - mcov)))


def test_gf():
    state_pdf, measurement_pdf = sim_base.get_noise(lib=numpy)
    bioreactor, _, _, _ = sim_base.get_parts(gpu=False)
    x0, _ = sim_base.get_noise(lib=numpy)
    x0.means = bioreactor.X[numpy.newaxis, :]
    gf = GaussianSumUnscentedKalmanFilter(
        f=bioreactor.homeostatic_DEs,
        g=bioreactor.static_outputs,
        N_gaussians=7,
        x0=x0,
        state_pdf=state_pdf,
        measurement_pdf=measurement_pdf
    )

    u, z = sim_base.get_random_io()
    gf.predict(u, 0.1)
    gf.update(u, z)
