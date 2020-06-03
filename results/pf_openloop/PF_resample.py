import numpy
import time
import pandas
import matplotlib.pyplot as plt
import tqdm
import sim_base


def generate_results(redo=False, cpu=True):
    try:
        if redo:
            raise FileNotFoundError

        df = pandas.read_csv('PF_resample.csv', index_col=0)
    except FileNotFoundError:
        df = pandas.DataFrame(columns=['CPU', 'GPU'])

    N_done = df.shape[0]
    N = 24

    if N_done >= N:
        return

    count = 20
    times = numpy.full((N - N_done, 2), numpy.inf)
    for i in tqdm.tqdm(range(N - N_done)):

        if cpu:
            _, _, _, p = sim_base.get_parts(
                N_particles=2 ** (N_done + i + 1),
                gpu=False
            )
            for j in range(count):
                p.weights = numpy.random.random(size=p.N_particles)
                p.weights /= numpy.sum(p.weights)
                t_cpu = time.time()
                p.resample()
                times[i, 0] = min(time.time() - t_cpu, times[i, 0])

        _, _, _, pp = sim_base.get_parts(
            N_particles=2 ** (N_done + i + 1),
            gpu=True
        )
        for j in range(count):
            pp.weights = numpy.random.random(size=pp.N_particles)
            pp.weights /= numpy.sum(pp.weights)
            t_gpu = time.time()
            pp.resample()
            times[i, 1] = min(time.time() - t_gpu, times[i, 1])

    df_new = pandas.DataFrame(times, columns=['CPU', 'GPU'], index=range(N_done+1, N+1))
    df = df.append(df_new)
    df.to_csv('PF_resample.csv')


def plot_results():
    df = pandas.read_csv('PF_resample.csv', index_col=0)
    df['speedup'] = df['CPU']/df['GPU']
    plt.semilogy(df.index, df['speedup'], '.')

    plt.title('Speed-up of particle filter resampling')
    plt.ylabel('Speed-up')
    plt.xlabel('$ \log_2(N) $ particles')

    plt.savefig('PF_resample.pdf')
    plt.show()


if __name__ == '__main__':
    generate_results(redo=True, cpu=False)
    plot_results()
