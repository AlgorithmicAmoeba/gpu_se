import numpy
import numba


class ParticleFilter:
    """Implements a parallel particle filter algorithm.
    """

    def __init__(self, f, g, N_particles, x0):
        self.f = f
        self.g = g
        self.N_particles = N_particles

        self.particles = x0.draw(N_particles)
        self.weights = numpy.full(N_particles, 1/N_particles)

    @numba.cuda.jit
    def predict(self, dt, u):

