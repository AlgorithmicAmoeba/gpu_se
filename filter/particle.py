import numpy
from numba import cuda


class ParticleFilter:
    """Implements a parallel particle filter algorithm.
    """

    def __init__(self, f, g, N_particles, x0, measurment_pdf):
        self.f = f
        self.g = g
        self.N_particles = N_particles

        particles_host = x0.draw(N_particles)
        self.particles = cuda.to_device(particles_host)
        self.weights = numpy.full(N_particles, 1/N_particles)
        self.measurment_pdf = measurment_pdf

        self.threads_per_block = self.tpb = 1024
        self.blocks_per_grid = self.bpg = (self.N_particles - 1) // self.threads_per_block + 1

    def predict(self, u, dt):
        ParticleFilter.predict_kernel[self.tpb, self.bpg](dt, u, self.f, self.particles)

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

    def update(self, u, z):
        ParticleFilter.update_kernel[self.tpb, self.bpg](u, z,
                                                         self.g, self.particles, self.weights, self.measurment_pdf)

    @staticmethod
    @cuda.jit
    def update_kernel(u, z, g, particles, weights, measurment_pdf):
        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        bw = cuda.blockDim.x
        indx = bw * bx + tx

        if indx >= particles.size:
            return

        y_i = g(particles[indx], u)
        e_i = z - y_i
        weights[indx] *= measurment_pdf(e_i)
