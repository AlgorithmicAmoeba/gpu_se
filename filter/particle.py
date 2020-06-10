import numpy
import numba
import numba.cuda as cuda
import torch
import torch.utils.dlpack as torch_dlpack
import cupy


class ParticleFilter:
    """Implements a particle filter algorithm"""

    def __init__(self, f, g, N_particles, x0, state_pdf, measurement_pdf):
        self.f = f
        self.g = g
        self.N_particles = int(N_particles)

        self.particles = x0.draw(N_particles)
        self.weights = numpy.full(N_particles, 1 / N_particles, dtype=numpy.float32)
        self.state_pdf = state_pdf
        self.measurement_pdf = measurement_pdf

    def predict(self, u, dt):
        for i, particle in enumerate(self.particles):
            self.particles[i] += self.f(particle, u, dt) + self.state_pdf.draw()

    def update(self, u, z):
        for i, particle in enumerate(self.particles):
            y = self.g(particle, u)
            e = z - y
            self.weights[i] *= self.measurement_pdf.pdf(e)

    def resample(self):
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


class ParallelParticleFilter(ParticleFilter):
    """Implements a parallel particle filter algorithm.
    """

    def __init__(self, f, g, N_particles, x0, state_pdf, measurement_pdf):
        super().__init__(f, g, N_particles, x0, state_pdf, measurement_pdf)

        self.f_vectorize = self.__f_vec()
        self.g_vectorize = self.__g_vec()

        self.particles_device = cupy.asarray(self.particles)
        self.weights_device = cupy.asarray(self.weights)

        # This object should no longer have anything to do with these variables
        del self.particles
        del self.weights

        self.threads_per_block = self.tpb = 1024
        self.blocks_per_grid = self.bpg = (self.N_particles - 1) // self.threads_per_block + 1

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

    def predict(self, u, dt):
        self.particles_device += self.f_vectorize(self.particles_device, u, dt)
        self.particles_device += self.state_pdf.draw(self.N_particles)

    def update(self, u, z):
        z = cupy.asarray(z, dtype=numpy.float32)
        ys = cupy.asarray(self.g_vectorize(self.particles_device, u, self._y_dummy))
        es = z - ys
        ws = cupy.asarray(self.measurement_pdf.pdf(es))
        self.weights_device *= ws

    def resample(self):
        t_weights = torch_dlpack.from_dlpack(cupy.asarray(self.weights_device).toDlpack())
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

        ParallelParticleFilter.__parallel_resample[blocks_per_grid, threads_per_block](cumsum, sample_index,
                                                                                       random_number, self.N_particles)

        self.particles_device = cupy.asarray(self.particles_device)[sample_index]
        self.weights_device = cupy.full(self.N_particles, 1 / self.N_particles)
