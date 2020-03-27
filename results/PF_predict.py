from filter.particle import ParticleFilter, ParallelParticleFilter
from gpu_funcs.MultivariateGaussianSum import MultivariateGaussianSum
import numpy
import time
import pandas
import matplotlib.pyplot as plt
import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def f(x, u, dt):
    x1, x2 = x

    dx1 = x1 - x2 * u
    dx2 = x1 / u + 2 * x2

    return x1 + dx1 * dt, x2 + dx2 * dt


def g(x, u):
    x1, x2 = x

    return x1 * x2, x2 + u


x0_cpu = MultivariateGaussianSum(means=numpy.array([[10, 0],
                                                    [-10, -10]]),
                                 covariances=numpy.array([[[1, 0],
                                                           [0, 1]],

                                                          [[2, 0.5],
                                                           [0.5, 0.5]]]),
                                 weights=numpy.array([0.3, 0.7]),
                                 library=numpy)

measurement_noise_cpu = MultivariateGaussianSum(means=numpy.array([[1, 0],
                                                                   [0, -1]]),
                                                covariances=numpy.array([[[0.6, 0],
                                                                          [0, 0.6]],

                                                                         [[0.5, 0.1],
                                                                          [0.1, 0.5]]]),
                                                weights=numpy.array([0.85, 0.15]),
                                                library=numpy)

x0_gpu = MultivariateGaussianSum(means=numpy.array([[10, 0],
                                                    [-10, -10]]),
                                 covariances=numpy.array([[[1, 0],
                                                           [0, 1]],

                                                          [[2, 0.5],
                                                           [0.5, 0.5]]]),
                                 weights=numpy.array([0.3, 0.7]))

measurement_noise_gpu = MultivariateGaussianSum(means=numpy.array([[1, 0],
                                                                   [0, -1]]),
                                                covariances=numpy.array([[[0.6, 0],
                                                                          [0, 0.6]],

                                                                         [[0.5, 0.1],
                                                                          [0.1, 0.5]]]),
                                                weights=numpy.array([0.85, 0.15]))


def generate_results():
    N = 15
    count = 5
    times = numpy.zeros((N, 2))
    for i in tqdm.tqdm(range(N)):
        p = ParticleFilter(f, g, 2**(i+1), x0_cpu, measurement_noise_cpu)
        pp = ParallelParticleFilter(f, g, 2**(i+1), x0_gpu, measurement_noise_gpu)

        t_cpu = time.time()
        for j in range(count):
            p.predict(1, 1)
            p.predict(-1, 1)
        times[i, 0] = time.time() - t_cpu

        t_gpu = time.time()
        for j in range(count):
            pp.predict(1, 1)
            pp.predict(-1, 1)
        times[i, 1] = time.time() - t_gpu

    df = pandas.DataFrame(times, columns=['CPU', 'GPU'], index=range(1, N+1))
    df.to_csv('PF_predict.csv')


def plot_results():
    df = pandas.read_csv('PF_predict.csv')
    df['speedup'] = df['CPU']/df['GPU']
    plt.plot(df.index, df['speedup'])


if __name__ == '__main__':
    generate_results()
    plot_results()
