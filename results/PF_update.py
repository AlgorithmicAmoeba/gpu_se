from filter.particle import ParticleFilter, ParallelParticleFilter
import time
import pandas
import matplotlib.pyplot as plt
import tqdm
from results.PF_base import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def generate_results():
    N = 15
    count = 10
    times = numpy.zeros((N, 2))

    z = numpy.array([2.3, 1.2])

    for i in tqdm.tqdm(range(N)):
        p = ParticleFilter(f, g, 2**(i+1), x0_cpu, measurement_noise_cpu)
        pp = ParallelParticleFilter(f, g, 2**(i+1), x0_gpu, measurement_noise_gpu)

        t_cpu = time.time()
        for j in range(count):
            p.update(1.0, z)
        times[i, 0] = time.time() - t_cpu

        t_gpu = time.time()
        for j in range(count):
            pp.update(1.0, z)
        times[i, 1] = time.time() - t_gpu

    df = pandas.DataFrame(times, columns=['CPU', 'GPU'], index=range(1, N+1))
    df.to_csv('PF_update.csv')


def plot_results():
    df = pandas.read_csv('PF_update.csv')
    df['speedup'] = df['CPU']/df['GPU']
    plt.semilogy(df.index, df['speedup'], '.')

    plt.title('Speed-up of particle filter update')
    plt.ylabel('Speed-up')
    plt.xlabel('$ \log_2(N) $ particles')


if __name__ == '__main__':
    generate_results()
    plot_results()