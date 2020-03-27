import numpy
import numba.cuda as cuda
import cupy


def systematic_sample(N, weights):
    cumsum = numpy.cumsum(weights)
    cumsum /= cumsum[-1]

    sample_index_result = numpy.zeros(N, dtype=numpy.int64)
    r = numpy.random.rand()
    k = 0

    for i in range(N):
        u = (i + r) / N
        while cumsum[k] < u:
            k += 1
        sample_index_result[i] = k

    return sample_index_result


def nicely_systematic_sample(N, weights):
    cumsum = cupy.cumsum(weights)
    cumsum /= cumsum[-1]

    sample_index_result = cupy.zeros(N, dtype=cupy.int64)
    random_number = cupy.float64(cupy.random.rand())
    N_weights = weights.size

    threads_per_block = 1024
    blocks_per_grid = (N - 1) // threads_per_block + 1

    @cuda.jit
    def do_in_parallel(c, sample_index, r):
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

        sample_index[i] = k + 1

    do_in_parallel[threads_per_block, blocks_per_grid](cumsum, sample_index_result, random_number)

    return sample_index_result
