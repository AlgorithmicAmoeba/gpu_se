import numpy
import numba
import numba.cuda as cuda


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


class ParallelParticleFilter(ParticleFilter):
    """Implements a parallel particle filter algorithm.
    """

    def __init__(self, f, g, N_particles, x0, measurement_pdf):
        super().__init__(f, g, N_particles, x0, measurement_pdf)

        f_jit = cuda.jit(device=True)(f)

        @numba.guvectorize(['void(f8[:], i4, i4, f8[:])',
                            'void(f8[:], i8, i8, f8[:])',
                            'void(f8[:], f4, f4, f8[:])',
                            'void(f8[:], f8, f8, f8[:])'],
                           '(n), (), () -> (n)', target='cuda')
        def f_vec(x, u, dt, _x_out):
            _x_out = f_jit(x, u, dt)

        self.f_vectorize = f_vec

        self.particles_device = cuda.to_device(self.particles)
        self.weights_device = cuda.to_device(self.weights)

        # This object should no longer have anything to do with these variables
        del self.particles
        del self.weights

        self.threads_per_block = self.tpb = 1024
        self.blocks_per_grid = self.bpg = (self.N_particles - 1) // self.threads_per_block + 1

    def predict(self, u, dt):
        self.particles_device = self.f_vectorize(self.particles_device, u, dt)

    def update(self, u, z):
        ParallelParticleFilter.update_kernel[self.tpb, self.bpg](u, z, self.g, self.particles_device,
                                                                 self.weights_device, self.measurement_pdf)

    @staticmethod
    @cuda.jit
    def update_kernel(u, z, g, particles, weights, measurement_pdf):
        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        bw = cuda.blockDim.x
        indx = bw * bx + tx

        if indx >= particles.size:
            return

        y_i = g(particles[indx], u)
        e_i = z - y_i
        weights[indx] *= measurement_pdf(e_i)
