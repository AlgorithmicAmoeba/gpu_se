import numpy
import cupy
import numba
import numba.cuda as cuda
import torch
import torch.utils.dlpack as torch_dlpack
import gpu_funcs


class GaussianSumUnscentedKalmanFilter:
    """Gaussian Sum Unscented Kalman Filter class implemented to run on the CPU.

    It contains methods that allow the use to perform
    predictions, updates, and resampling.

    Parameters
    ----------
    f : callable
        The state transition function :math:` x_{k+1} += f(x_k, u_k) `

    g : callable
        The state observation function :math:` y_k = g(x_k, u_k) `

    N_particles : int
        The number of particles

    x0 : gpu_funcs.MultivariateGaussianSum
        The initial distribution.
        Represented as a Gaussian sum

    state_pdf, measurement_pdf : gpu_funcs.MultivariateGaussianSum
        Distributions for the state and measurement noise.
        Represented as Gaussian sums

    Attributes
    -----------
    means : numpy.array
        An (N_particles x Nx) array of the particles

    covariances : numpy.array
        An (N_particles x Nx x Nx) array of covariances of the particles

    weights : numpy.array
        A (N_particles) array containing the weights of the particles
    """
    def __init__(self, f, g, N_particles, x0, state_pdf, measurement_pdf):
        self.f = f
        self.g = g
        self.N_particles = int(N_particles)

        self.means = x0.draw(N_particles)

        average_cov = numpy.average(state_pdf.covariances, axis=0)
        self.covariances = numpy.repeat(average_cov[None, :, :], N_particles, axis=0)

        self.weights = numpy.full(N_particles, 1 / N_particles, dtype=numpy.float32)

        self.state_pdf = state_pdf
        self.measurement_pdf = measurement_pdf

        self._Nx = self.means.shape[1]
        self._Ny = measurement_pdf.draw().shape[0]
        self._N_sigmas = 2 * self._Nx + 1

        # weights calculated such that:
        # 1) w_mu + 2*n*w_sigma = 1
        # 2) w_mu / w_sigma = 0.4 / 0.25 ~ Normal_pdf(0) / Normal_pdf(sigma)
        self._w_sigma = numpy.full(self._N_sigmas, 1 / (2 * self._Nx + 8 / 5), dtype=numpy.float32)
        self._w_sigma[0] = 1 / (1 + 5 / 4 * self._Nx)

    def _get_sigma_points(self):
        """Return the sigma points for the current particles
        """
        try:
            stds = numpy.linalg.cholesky(self.covariances).swapaxes(1, 2)
        except numpy.linalg.LinAlgError:
            stds = numpy.linalg.cholesky(self.covariances + 1e-10 * numpy.eye(self._Nx)).swapaxes(1, 2)
        sigmas = numpy.repeat(self.means[:, None, :], self._N_sigmas, axis=1)
        sigmas[:, 1:self._Nx + 1, :] += stds
        sigmas[:, self._Nx + 1:, :] -= stds

        return sigmas

    def predict(self, u, dt):
        """Performs a prediction step on the particles

        Parameters
        ----------
        u : numpy.array
            A (N_inputs) array of the current inputs

        dt : float
            The time step since the previous prediction
        """
        sigmas = self._get_sigma_points()

        # Move the sigma points through the state transition function
        for gaussian in range(self.N_particles):
            for sigma in range(self._N_sigmas):
                sigmas[gaussian, sigma] += self.f(sigmas[gaussian, sigma], u, dt)
        sigmas += self.state_pdf.draw((self.N_particles, self._N_sigmas))

        self.means = numpy.average(sigmas, axis=1, weights=self._w_sigma)
        sigmas -= self.means[:, None, :]
        self.covariances = sigmas.swapaxes(1, 2) @ (sigmas * self._w_sigma[:, None])

    def update(self, u, z):
        """Performs an update step on the particles

        Parameters
        ----------
        u : numpy.array
            A (N_inputs) array of the current inputs

        z : numpy.array
            A (N_outputs) array of the current  measured outputs
        """
        # Local Update
        sigmas = self._get_sigma_points()
        etas = numpy.zeros((self.N_particles, self._N_sigmas, self._Ny))

        # Move the sigma points through the state observation function
        for gaussian in range(self.N_particles):
            for sigma in range(self._N_sigmas):
                etas[gaussian, sigma] = self.g(sigmas[gaussian, sigma], u)

        # Compute the Kalman gain
        eta_means = numpy.average(etas, axis=1, weights=self._w_sigma)
        sigmas -= self.means[:, None, :]
        etas -= eta_means[:, None, :]

        P_xys = sigmas.swapaxes(1, 2) @ (etas * self._w_sigma[:, None])
        P_yys = etas.swapaxes(1, 2) @ (etas * self._w_sigma[:, None])
        P_yy_invs = numpy.linalg.inv(P_yys)
        Ks = P_xys @ P_yy_invs

        # Use the gain to update the means and covariances
        es = z - eta_means
        self.means += (Ks @ es[:, :, None]).squeeze()
        # Dimensions from paper do not work, use corrected version
        self.covariances -= Ks @ P_yys @ Ks.swapaxes(1, 2)

        # Global Update
        y_means = numpy.zeros((self.N_particles, self._Ny))

        # Move the means through the state observation function
        for gaussian in range(self.N_particles):
            y_means[gaussian] = self.g(self.means[gaussian], u)

        glob_es = z - y_means
        self.weights *= self.measurement_pdf.pdf(glob_es)

    def resample(self):
        """Performs a systematic resample of the particles
        based on the weights of the particles
        """
        cumsum = numpy.cumsum(self.weights)
        cumsum /= cumsum[-1]

        sample_index_result = numpy.zeros(self.N_particles, dtype=numpy.int64)
        r = numpy.random.rand()
        k = 0

        for i in range(self.N_particles):
            u = (i + r) / self.N_particles
            while cumsum[k] < u:
                k += 1
            sample_index_result[i] = k

        self.means = self.means[sample_index_result]
        self.covariances = self.covariances[sample_index_result]
        self.weights = numpy.full(self.N_particles, 1 / self.N_particles)


class ParallelGaussianSumUnscentedKalmanFilter(GaussianSumUnscentedKalmanFilter):
    """Gaussian Sum Unscented Kalman Filter class implemented to run on the GPU.

    It contains methods that allow the use to perform
    predictions, updates, and resampling.

    Parameters
    ----------
    f : callable
        The state transition function :math:` x_{k+1} += f(x_k, u_k) `

    g : callable
        The state observation function :math:` y_k = g(x_k, u_k) `

    N_particles : int
        The number of particles

    x0 : gpu_funcs.MultivariateGaussianSum
        The initial distribution.
        Represented as a Gaussian sum

    state_pdf, measurement_pdf : gpu_funcs.MultivariateGaussianSum
        Distributions for the state and measurement noise.
        Represented as Gaussian sums

    Attributes
    -----------
    means : cupy.array
        An (N_particles x Nx) array of the particles

    covariances : cupy.array
        An (N_particles x Nx x Nx) array of covariances of the particles

    weights : cupy.array
        A (N_particles) array containing the weights of the particles
    """

    def __init__(self, f, g, N_particles, x0, state_pdf, measurement_pdf):
        super().__init__(f, g, N_particles, x0, state_pdf, measurement_pdf)

        self.f_vectorize = self.__f_vec()
        self.g_vectorize = self.__g_vec()

        # Move the data to the GPU
        self.means = cupy.asarray(self.means)
        self.covariances = cupy.asarray(self.covariances)
        self.weights = cupy.asarray(self.weights)
        self._w_sigma = cupy.asarray(self._w_sigma)

        self._threads_per_block = self._tpb = 1024
        self._blocks_per_grid = self._bpg = (self.N_particles - 1) // self._threads_per_block + 1

        self._y_dummy = cupy.zeros_like(self.measurement_pdf.draw())

    def __f_vec(self):
        """Vectorizes the state transition function to run on the GPU
        """
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
        """Vectorizes the state observation function to run on the GPU
        """
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
        """Vectorizes the measurement probability density function
        to run on the GPU
        """
        pdf_jit = cuda.jit(device=True)(self.measurement_pdf.pdf)

        @numba.guvectorize(['void(f4[:], f4[:])'],
                           '(n) -> (n)', target='cuda')
        def pdf_vec(e, _pdf_out=None):
            _pdf_out = pdf_jit(e)

        return pdf_vec

    @staticmethod
    @cuda.jit
    def _parallel_resample(cumsum, sample_index, random_number, N_particles):
        """Implements the parallel aspect of the
        systematic resampling algorithm by Nicely

        Parameters
        ----------
        cumsum : cupy.array
            The cumulative sum of the particle weights

        sample_index : cupy.array
            The array where the sample indices will be stored

        random_number : float
            A random float between 0 and 1

        N_particles : int
            The number of particles
        """
        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        bw = cuda.blockDim.x
        i = bw * bx + tx

        if i >= N_particles:
            return

        u = (i + random_number) / N_particles
        k = i
        while cumsum[k] < u:
            k += 1

        cuda.syncthreads()

        while cumsum[k] > u and k >= 0:
            k -= 1

        cuda.syncthreads()

        sample_index[i] = k + 1

    def _get_sigma_points(self):
        """Return the sigma points for the current particles
        """
        t_covariances = torch_dlpack.from_dlpack(cupy.asarray(self.covariances).toDlpack())
        t_stds = torch.cholesky(t_covariances)
        stds = cupy.fromDlpack(torch_dlpack.to_dlpack(t_stds)).swapaxes(1, 2)
        sigmas = cupy.repeat(self.means[:, None, :], self._N_sigmas, axis=1)
        sigmas[:, 1:self._Nx + 1, :] += stds
        sigmas[:, self._Nx + 1:, :] -= stds

        return sigmas

    def predict(self, u, dt):
        """Performs a prediction step on the particles

        Parameters
        ----------
        u : cupy.array
            A (N_inputs) array of the current inputs

        dt : float
            The time step since the previous prediction
        """
        sigmas = self._get_sigma_points()

        # Move the sigma points through the state transition function
        sigmas += self.f_vectorize(sigmas, u, dt)
        sigmas += self.state_pdf.draw((self.N_particles, self._N_sigmas))

        self.means = cupy.average(sigmas, axis=1, weights=self._w_sigma)
        sigmas -= self.means[:, None, :]
        self.covariances = sigmas.swapaxes(1, 2) @ (sigmas * self._w_sigma[:, None])

    def update(self, u, z):
        """Performs an update step on the particles

        Parameters
        ----------
        u : cupy.array
            A (N_inputs) array of the current inputs

        z : cupy.array
            A (N_outputs) array of the current  measured outputs
        """
        # Local Update
        sigmas = self._get_sigma_points()
        # Move the sigma points through the state observation function
        etas = self.g_vectorize(sigmas, u, self._y_dummy)

        # Compute the Kalman gain
        eta_means = cupy.average(etas, axis=1, weights=self._w_sigma)
        sigmas -= self.means[:, None, :]
        etas -= eta_means[:, None, :]

        P_xys = sigmas.swapaxes(1, 2) @ (etas * self._w_sigma[:, None])
        P_yys = etas.swapaxes(1, 2) @ (etas * self._w_sigma[:, None])
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
        """Performs a systematic resample of the particles
        based on the weights of the particles.
        Uses the algorithm by Nicely.
        """
        t_weights = torch_dlpack.from_dlpack(cupy.asarray(self.weights).toDlpack())
        t_cumsum = torch.cumsum(t_weights, 0)
        cumsum = cupy.fromDlpack(torch_dlpack.to_dlpack(t_cumsum))
        cumsum /= cumsum[-1]

        sample_index = cupy.zeros(self.N_particles, dtype=cupy.int64)
        random_number = cupy.float64(cupy.random.rand())

        if self.N_particles >= 1024:
            threads_per_block = 1024
            blocks_per_grid = (self.N_particles - 1) // threads_per_block + 1
        else:
            div_32 = (self.N_particles - 1) // 32 + 1
            threads_per_block = 32 * div_32
            blocks_per_grid = 1

        ParallelGaussianSumUnscentedKalmanFilter._parallel_resample[blocks_per_grid, threads_per_block](
            cumsum, sample_index, random_number, self.N_particles
        )

        self.means = cupy.asarray(self.means)[sample_index]
        self.covariances = cupy.asarray(self.covariances)[sample_index]
        self.weights = cupy.full(self.N_particles, 1 / self.N_particles)
