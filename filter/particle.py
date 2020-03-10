import numpy
from numba import cuda


class ParticleFilter:
    """Implements a parallel particle filter algorithm.
    """

    def __init__(self, f, g, N_particles, x0):
        self.f = f
        self.g = g
        self.N_particles = N_particles

        particles_host = x0.draw(N_particles)
        self.particles = cuda.to_device(particles_host)
        self.weights = numpy.full(N_particles, 1/N_particles)

    def predict(self, dt, u):
        threads_per_block = 1024
        blocks_per_grid = (self.N_particles - 1) // threads_per_block + 1

        ParticleFilter.predict_kernel[blocks_per_grid, threads_per_block](dt, u, self.f, self.particles)

    @staticmethod
    @cuda.jit
    def predict_kernel(dt, u, f, particles):
        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        bw = cuda.blockDim.x
        indx = bw * bx + tx

        if indx >= particles.size:
            return

        particles[indx] = f(particles[indx], u, dt)
