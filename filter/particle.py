import numpy
import numba
import numba.cuda as cuda
import torch
import torch.utils.dlpack as torch_dlpack
import cupy


class ParticleFilter:
    """Particle filter class implemented to run on the CPU.

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
    particles : numpy.array
        An (N_particles x Nx) array of the particles

    weights : numpy.array
        A (N_particles) array containing the weights of the particles
    """

    def __init__(self, f, g, N_particles, x0, state_pdf, measurement_pdf):

        self.f = f
        self.g = g
        self.N_particles = int(N_particles)

        self.particles = x0.draw(N_particles)
        self.weights = numpy.full(N_particles, 1 / N_particles, dtype=numpy.float32)
        self.state_pdf = state_pdf
        self.measurement_pdf = measurement_pdf

    def predict(self, u, dt):
        """Performs a prediction step on the particles

        Parameters
        ----------
        u : numpy.array
            A (N_inputs) array of the current inputs

        dt : float
            The time step since the previous prediction
        """
        for i, particle in enumerate(self.particles):
            self.particles[i] += self.f(particle, u, dt)
        self.particles += self.state_pdf.draw(self.N_particles)

    def update(self, u, z):
        """Performs an update step on the particles

        Parameters
        ----------
        u : numpy.array
            A (N_inputs) array of the current inputs

        z : numpy.array
            A (N_outputs) array of the current  measured outputs
        """
        for i, particle in enumerate(self.particles):
            y = self.g(particle, u)
            e = z - y
            self.weights[i] *= self.measurement_pdf.pdf(e)

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

        self.particles = self.particles[sample_index_result]
        self.weights = numpy.full(self.N_particles, 1 / self.N_particles)

    def point_estimate(self):
        """Returns the point estimate of the filter"""
        return self.weights @ self.particles

    def point_covariance(self):
        """Returns the maximum singular value of the filter's covariance"""
        dist = self.particles - (self.weights @ self.particles)
        cov = dist.T @ (dist * self.weights[:, None])
        s = numpy.linalg.svd(cov, compute_uv=False)
        return s[0]


class ParallelParticleFilter(ParticleFilter):
    """Particle filter class implemented to run on the GPU.

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
    particles : cupy.array
        An (N_particles x Nx) array of the particles

    weights : cupy.array
        A (N_particles) array containing the weights of the particles
    """

    def __init__(self, f, g, N_particles, x0, state_pdf, measurement_pdf):
        super().__init__(f, g, N_particles, x0, state_pdf, measurement_pdf)

        self.f_vectorize = self.__f_vec()
        self.g_vectorize = self.__g_vec()

        self.particles = cupy.asarray(self.particles)
        self.weights = cupy.asarray(self.weights)

        if self.N_particles >= 1024:
            threads_per_block = 1024
            blocks_per_grid = (self.N_particles - 1) // threads_per_block + 1
        else:
            div_32 = (self.N_particles - 1) // 32 + 1
            threads_per_block = 32 * div_32
            blocks_per_grid = 1

        self._tpb = threads_per_block
        self._bpg = blocks_per_grid

        self._y_dummy = cupy.zeros(
            self.measurement_pdf.draw().shape[1],
            dtype=cupy.float32
        )

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

    def predict(self, u, dt):
        """Performs a prediction step on the particles

        Parameters
        ----------
        u : numpy.array
            A (N_inputs) array of the current inputs

        dt : float
            The time step since the previous prediction
        """
        self.particles += self.f_vectorize(self.particles, u, dt)
        self.particles += self.state_pdf.draw(self.N_particles)

    def update(self, u, z):
        """Performs an update step on the particles

        Parameters
        ----------
        u : numpy.array
            A (N_inputs) array of the current inputs

        z : numpy.array
            A (N_outputs) array of the current  measured outputs
        """
        z = cupy.asarray(z, dtype=cupy.float32)
        ys = cupy.asarray(self.g_vectorize(self.particles, u, self._y_dummy))
        es = z - ys
        ws = cupy.asarray(self.measurement_pdf.pdf(es))
        self.weights *= ws

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

        ParallelParticleFilter._parallel_resample[self._bpg, self._tpb](
            cumsum, sample_index,
            random_number,
            self.N_particles
        )

        self.particles = cupy.asarray(self.particles)[sample_index]
        self.weights = cupy.full(self.N_particles, 1 / self.N_particles)

    def point_estimate(self):
        """Returns the point estimate of the filter"""
        return (self.weights @ self.particles).get()

    def point_covariance(self):
        """Returns the maximum singular value of the filter's covariance"""
        dist = self.particles - (self.weights @ self.particles)
        cov = dist.T @ (dist * self.weights[:, None])
        s = cupy.linalg.svd(cov, compute_uv=False)
        return (s[0]).get()
