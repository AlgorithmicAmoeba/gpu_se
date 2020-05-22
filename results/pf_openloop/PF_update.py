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

        df = pandas.read_csv('PF_update.csv', index_col=0)
    except FileNotFoundError:
        df = pandas.DataFrame(columns=['CPU', 'GPU'])

    N_done = df.shape[0]
    N = 20

    if N_done >= N:
        return

    count = 10
    times = numpy.full((N - N_done, 2),  numpy.inf)

    for i in tqdm.tqdm(range(N - N_done)):
        if cpu:
            _, _, _, p = sim_base.get_parts(
                N_particles=2 ** (N_done + i + 1),
                gpu=False
            )
            for j in range(count):
                u, y = sim_base.get_random_io()
                t_cpu = time.time()
                p.update(u, y)
                times[i, 0] = min(time.time() - t_cpu, times[i, 0])

        _, _, _, pp = sim_base.get_parts(
            N_particles=2 ** (N_done + i + 1),
            gpu=True
        )
        for j in range(count):
            u, y = sim_base.get_random_io()
            t_gpu = time.time()
            pp.update(u, y)
            times[i, 1] = min(time.time() - t_gpu, times[i, 1])

    df_new = pandas.DataFrame(times, columns=['CPU', 'GPU'], index=range(N_done + 1, N + 1))
    df = df.append(df_new)
    df.to_csv('PF_update.csv')


def plot_results():
    df = pandas.read_csv('PF_update.csv', index_col=0)
    df['speedup'] = df['CPU']/df['GPU']
    plt.semilogy(df.index, df['speedup'], '.')

    plt.title('Speed-up of particle filter update')
    plt.ylabel('Speed-up')
    plt.xlabel('$ \log_2(N) $ particles')

    plt.savefig('PF_update.pdf')
    plt.show()


if __name__ == '__main__':
    generate_results(redo=True, cpu=True)
    plot_results()
