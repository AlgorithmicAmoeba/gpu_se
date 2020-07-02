import numpy
import cupy
import numba
import numba.cuda as cuda
import torch
import torch.utils.dlpack as torch_dlpack


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
        self.w_sigma = numpy.full(self.N_sigmas, 1 / (2 * self.Nx + 8 / 5), dtype=numpy.float32)
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
        self.weights *= self.measurement_pdf.pdf(glob_es)

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
        self.covariances = self.covariances[sample_index_result]
        self.weights = numpy.full(self.N_gaussians, 1 / self.N_gaussians)


class ParallelGaussianSumUnscentedKalmanFilter(GaussianSumUnscentedKalmanFilter):
    """Implements a parallel Gaussian sum unscented Kalman filter algorithm.
    """

    def __init__(self, f, g, N_gaussians, x0, state_pdf, measurement_pdf):
        super().__init__(f, g, N_gaussians, x0, state_pdf, measurement_pdf)

        self.f_vectorize = self.__f_vec()
        self.g_vectorize = self.__g_vec()

        # Move the data to the GPU
        self.means = cupy.asarray(self.means)
        self.covariances = cupy.asarray(self.covariances)
        self.weights = cupy.asarray(self.weights)
        self.w_sigma = cupy.asarray(self.w_sigma)

        self.threads_per_block = self.tpb = 1024
        self.blocks_per_grid = self.bpg = (self.N_gaussians - 1) // self.threads_per_block + 1

        self._y_dummy = cupy.zeros_like(self.measurement_pdf.draw())

    def __f_vec(self):
        f_jit = cuda.jit(device=True)(self.f)

        @numba.guvectorize(['void(f4[:], i4[:], i4, f4[:])',
                            'void(f4[:], i8[:], i8, f4[:])',
                            'void(f4[:], f4[:], f4, f4[:])',
                            'void(f4[:], f8[:], f8, f4[:])'],
                           '(n), (m), () -> (n)', target='cuda')
        def f_vec(x, u, dt, _x_out):
            ans = f_jit(x, u, dt)
            for i in range(len(ans)):
                _x_out[i] = ans[i]

        return f_vec

    def __g_vec(self):
        g_jit = cuda.jit(device=True)(self.g)

        @numba.guvectorize(['void(f4[:], i4[:], f4[:], f4[:])',
                            'void(f4[:], i8[:], f4[:], f4[:])',
                            'void(f4[:], f4[:], f4[:], f4[:])',
                            'void(f4[:], f8[:], f4[:], f4[:])'],
                           '(n), (m), (p) -> (p)', target='cuda')
        def g_vec(x, u, _y_dummy, _y_out):
            ans = g_jit(x, u)
            for i in range(len(ans)):
                _y_out[i] = ans[i]

        return g_vec

    def __pdf_vec(self):
        pdf_jit = cuda.jit(device=True)(self.measurement_pdf.pdf)

        @numba.guvectorize(['void(f4[:], f4[:])'],
                           '(n) -> (n)', target='cuda')
        def pdf_vec(e, _pdf_out=None):
            _pdf_out = pdf_jit(e)

        return pdf_vec

    @staticmethod
    @cuda.jit
    def __parallel_resample(c, sample_index, r, N):
        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        bw = cuda.blockDim.x
        i = bw * bx + tx

        if i >= N:
            return

        u = (i + r) / N
        k = i
        while c[k] < u:
            k += 1

        cuda.syncthreads()

        while c[k] > u and k >= 0:
            k -= 1

        cuda.syncthreads()

        sample_index[i] = k + 1

    def _get_sigma_points(self):
        t_covariances = torch_dlpack.from_dlpack(cupy.asarray(self.covariances).toDlpack())
        t_stds = torch.cholesky(t_covariances)
        stds = cupy.fromDlpack(torch_dlpack.to_dlpack(t_stds)).swapaxes(1, 2)
        sigmas = cupy.repeat(self.means[:, None, :], self.N_sigmas, axis=1)
        sigmas[:, 1:self.Nx+1, :] += stds
        sigmas[:, self.Nx+1:, :] -= stds

        return sigmas

    def predict(self, u, dt):
        sigmas = self._get_sigma_points()

        # Move the sigma points through the state transition function
        sigmas = self.f_vectorize(sigmas, u, dt)
        sigmas += self.state_pdf.draw((self.N_gaussians, self.N_sigmas))

        self.means = cupy.average(sigmas, axis=1, weights=self.w_sigma)
        sigmas -= self.means[:, None, :]
        self.covariances = sigmas.swapaxes(1, 2) @ (sigmas * self.w_sigma[:, None])

    def update(self, u, z):
        # Local Update
        sigmas = self._get_sigma_points()
        # Move the sigma points through the state observation function
        etas = self.g_vectorize(sigmas, u, self._y_dummy)

        # Compute the Kalman gain
        eta_means = cupy.average(etas, axis=1, weights=self.w_sigma)
        sigmas -= self.means[:, None, :]
        etas -= eta_means[:, None, :]

        P_xys = sigmas.swapaxes(1, 2) @ (etas * self.w_sigma[:, None])
        P_yys = etas.swapaxes(1, 2) @ (etas * self.w_sigma[:, None])
        P_yy_invs = cupy.linalg.inv(P_yys)
        Ks = P_xys @ P_yy_invs

        # Use the gain to update the means and covariances
        z = cupy.asarray(z, dtype=cupy.float32)
        es = z - eta_means
        self.means += (Ks @ es[:, :, None]).squeeze()
        # Dimensions from paper do not work, use corrected version
        self.covariances -= Ks @ P_yys @ Ks.swapaxes(1, 2)

        # Global Update
        # Move the means through the state observation function
        y_means = self.g_vectorize(self.means, u, self._y_dummy)

        glob_es = z - y_means
        self.weights *= self.measurement_pdf.pdf(glob_es)

    def resample(self):
        t_weights = torch_dlpack.from_dlpack(cupy.asarray(self.weights).toDlpack())
        t_cumsum = torch.cumsum(t_weights, 0)
        cumsum = cupy.fromDlpack(torch_dlpack.to_dlpack(t_cumsum))
        cumsum /= cumsum[-1]

        sample_index = cupy.zeros(self.N_gaussians, dtype=cupy.int64)
        random_number = cupy.float64(cupy.random.rand())

        if self.N_gaussians >= 1024:
            threads_per_block = 1024
            blocks_per_grid = (self.N_gaussians - 1) // threads_per_block + 1
        else:
            div_32 = (self.N_gaussians - 1) // 32 + 1
            threads_per_block = 32 * div_32
            blocks_per_grid = 1

        ParallelGaussianSumUnscentedKalmanFilter.__parallel_resample[blocks_per_grid, threads_per_block](
            cumsum, sample_index, random_number, self.N_gaussians
        )

        self.means = cupy.asarray(self.means)[sample_index]
        self.covariances = cupy.asarray(self.covariances)[sample_index]
        self.weights = cupy.full(self.N_gaussians, 1 / self.N_gaussians)

