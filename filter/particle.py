import numpy
import numba
import numba.cuda as cuda
import cupy


class ParticleFilter:
    """Implements a particle filter algorithm"""

    def __init__(self, f, g, N_particles, x0, measurement_pdf):
        self.f = f
        self.g = g
        self.N_particles = N_particles

        self.particles = x0.draw(N_particles)
        self.weights = numpy.full(N_particles, 1 / N_particles)
        self.measurement_pdf = measurement_pdf

    def predict(self, u, dt):
        for i, particle in enumerate(self.particles):
            self.particles[i] = self.f(particle, u, dt)

    def update(self, u, z):
        for i, particle in enumerate(self.particles):
            y = self.g(particle, u)
            e = z - y
            self.weights[i] *= self.measurement_pdf.pdf(e)


class ParallelParticleFilter(ParticleFilter):
    """Implements a parallel particle filter algorithm.
    """

    def __init__(self, f, g, N_particles, x0, measurement_pdf):
        super().__init__(f, g, N_particles, x0, measurement_pdf)

        self.f_vectorize = self.__f_vec()
        self.g_vectorize = self.__g_vec()
        # self.pdf_vectorize = self.__pdf_vec()

        self.particles_device = cuda.to_device(self.particles)
        self.weights_device = cuda.to_device(self.weights)

        # This object should no longer have anything to do with these variables
        del self.particles
        del self.weights

        self.threads_per_block = self.tpb = 1024
        self.blocks_per_grid = self.bpg = (self.N_particles - 1) // self.threads_per_block + 1

    def __f_vec(self):
        f_jit = cuda.jit(device=True)(self.f)

        @numba.guvectorize(['void(f8[:], i4, i4, f8[:])',
                            'void(f8[:], i8, i8, f8[:])',
                            'void(f8[:], f4, f4, f8[:])',
                            'void(f8[:], f8, f8, f8[:])'],
                           '(n), (), () -> (n)', target='cuda')
        def f_vec(x, u, dt, _x_out=None):
            _x_out = f_jit(x, u, dt)

        return f_vec

    def __g_vec(self):
        g_jit = cuda.jit(device=True)(self.g)

        @numba.guvectorize(['void(f8[:], i4, f8[:])',
                            'void(f8[:], i8, f8[:])',
                            'void(f8[:], f4, f8[:])',
                            'void(f8[:], f8, f8[:])'],
                           '(n), () -> (n)', target='cuda')
        def g_vec(x, u, _y_out=None):
            _y_out = g_jit(x, u)

        return g_vec

    def __pdf_vec(self):
        pdf_jit = cuda.jit(device=True)(self.measurement_pdf.pdf)

        @numba.guvectorize(['void(f8[:], f8[:])',
                            'void(f8[:], f8[:])',
                            'void(f8[:], f8[:])',
                            'void(f8[:], f8[:])'],
                           '(n) -> (n)', target='cuda')
        def pdf_vec(e, _pdf_out=None):
            _pdf_out = pdf_jit(e)

        return pdf_vec

    def predict(self, u, dt):
        self.particles_device = self.f_vectorize(self.particles_device, u, dt)

    def update(self, u, z):
        z = cupy.asarray(z)
        ys = cupy.asarray(self.g_vectorize(self.particles_device, u))
        es = z - ys
        ws = cupy.asarray(self.measurement_pdf.pdf(es))
        self.weights_device *= ws

        # ParallelParticleFilter.update_kernel[self.tpb, self.bpg](u, z, self.g, self.particles_device,
        #                                                          self.weights_device, self.measurement_pdf)

    # @staticmethod
    # @cuda.jit
    # def update_kernel(u, z, g, particles, weights, measurement_pdf):
    #     tx = cuda.threadIdx.x
    #     bx = cuda.blockIdx.x
    #     bw = cuda.blockDim.x
    #     indx = bw * bx + tx
    #
    #     if indx >= particles.size:
    #         return
    #
    #     y_i = g(particles[indx], u)
    #     e_i = z - y_i
    #     weights[indx] *= measurement_pdf(e_i)
