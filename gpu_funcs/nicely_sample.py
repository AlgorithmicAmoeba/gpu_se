import numpy
import numba.cuda as cuda
import cupy

print("In nicely_sampling.py: rewriting cupy.cumsum to be numpy.cumsum")
cupy.cumsum = numpy.cumsum


def nicely_systematic_sample(N, weights):
    c_outer = cupy.cumsum(weights)

    sample_indx_outer = numpy.zeros(N)
    r = numpy.random.rand()
    threads_per_block = 1024
    blocks_per_grid = (N - 1) // threads_per_block + 1
    N_weights = weights.size

    @cuda.jit
    def do_in_parallel(c, sample_indx):
        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        bw = cuda.blockDim.x
        i = bw * bx + tx

        if i >= N:
            return

        u = (i + r) / N
        k = int(i/N*N_weights)
        while c[k] < u:
            k += 1

        cuda.syncthreads()

        while c[k] > u and k >= 0:
            k -= 1

        cuda.syncthreads()

        sample_indx[i] = k + 1

    do_in_parallel[threads_per_block, blocks_per_grid](c_outer, sample_indx_outer)

    return sample_indx_outer
