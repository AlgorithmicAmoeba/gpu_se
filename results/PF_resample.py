from filter.particle import ParticleFilter, ParallelParticleFilter
import time
import pandas
import matplotlib.pyplot as plt
import tqdm
from results.PF_base import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def generate_results(redo=False):
    try:
        if redo:
            raise FileNotFoundError

        df = pandas.read_csv('PF_resample.csv')
    except FileNotFoundError:
        df = pandas.DataFrame(columns=['CPU', 'GPU'])

    N_done = df.shape[0]

    N = 30
    count = 50
    times = numpy.zeros((N, 2))
    for i in tqdm.tqdm(range(N)):
        if i < N_done:
            continue

        p = ParticleFilter(f, g, 2**(i+1), x0_cpu, measurement_noise_cpu)
        pp = ParallelParticleFilter(f, g, 2**(i+1), x0_gpu, measurement_noise_gpu)

        p.resample()
        t_cpu = time.time()
        for j in range(count):
            p.resample()
        times[i, 0] = time.time() - t_cpu

        pp.resample()
        t_gpu = time.time()
        for j in range(count):
            pp.resample()
        times[i, 1] = time.time() - t_gpu

    df_new = pandas.DataFrame(times, columns=['CPU', 'GPU'], index=range(1, N+1))
    df.append(df_new)
    df.to_csv('PF_resample.csv')


def plot_results():
    df = pandas.read_csv('PF_resample.csv')
    df['speedup'] = df['CPU']/df['GPU']
    plt.semilogy(df.index, df['speedup'], '.')

    plt.title('Speed-up of particle filter resampling')
    plt.ylabel('Speed-up')
    plt.xlabel('$ \log_2(N) $ particles')


if __name__ == '__main__':
    generate_results()
    plot_results()
