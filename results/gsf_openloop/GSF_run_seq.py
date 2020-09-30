import numpy
import time
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import sim_base
import cupy
import statsmodels.tsa.stattools as stats_tools
import torch
import torch.utils.dlpack as torch_dlpack
import filter.particle
import filter.gs_ukf
from decorators import RunSequences, PickleJar


@RunSequences.vectorize
@PickleJar.pickle(path='gsf/raw')
def predict_run_seq(N_particle, N_runs, gpu):
    """Performs a run sequence on the prediction function with the given number
    of particle and number of runs on the CPU or GPU

    Parameters
    ----------
    N_particle : int
        Number of particles

    N_runs : int
        Number of runs in the sequence

    gpu : bool
        If `True` then the GPU implementation is used.
        Otherwise, the CPU implementation is used

    Returns
    -------
    times : numpy.array
        The times of the run sequence
    """
    times = []

    _, _, _, gsf = sim_base.get_parts(
        N_particles=N_particle,
        gpu=gpu,
        pf=False
    )

    for _ in range(N_runs):
        u, _ = sim_base.get_random_io()
        t = time.time()
        gsf.predict(u, 1.)
        times.append(time.time() - t)

    return numpy.array(times)


@RunSequences.vectorize
@PickleJar.pickle(path='gsf/raw')
def update_run_seq(N_particle, N_runs, gpu):
    """Performs a run sequence on the update function with the given number
    of particle and number of runs on the CPU or GPU

    Parameters
    ----------
    N_particle : int
        Number of particles

    N_runs : int
        Number of runs in the sequence

    gpu : bool
        If `True` then the GPU implementation is used.
        Otherwise, the CPU implementation is used

    Returns
    -------
    times : numpy.array
        The times of the run sequence
    """
    times = []

    _, _, _, gsf = sim_base.get_parts(
        N_particles=N_particle,
        gpu=gpu,
        pf=False
    )

    for j in range(N_runs):
        u, y = sim_base.get_random_io()
        t = time.time()
        gsf.update(u, y)
        times.append(time.time() - t)

    return numpy.array(times)


@RunSequences.vectorize
@PickleJar.pickle(path='gsf/raw')
def resample_run_seq(N_particle, N_runs, gpu):
    """Performs a run sequence on the resample function with the given number
    of particle and number of runs on the CPU or GPU

    Parameters
    ----------
    N_particle : int
        Number of particles

    N_runs : int
        Number of runs in the sequence

    gpu : bool
        If `True` then the GPU implementation is used.
        Otherwise, the CPU implementation is used

    Returns
    -------
    times : numpy.array
        The times of the run sequence
    """
    times = []

    _, _, _, gsf = sim_base.get_parts(
        N_particles=N_particle,
        gpu=gpu,
        pf=False
    )

    for j in range(N_runs):
        gsf.weights = numpy.random.random(size=gsf.N_particles)
        gsf.weights /= numpy.sum(gsf.weights)
        t = time.time()
        gsf.resample()
        times.append(time.time() - t)

    return numpy.array(times)


@RunSequences.vectorize
@PickleJar.pickle(path='gsf/raw')
def sigma_points_run_seq(N_particle, N_runs):
    """Performs a run sequence on the sigma point function with the given number
    of particle and number of runs

    Parameters
    ----------
    N_particle : int
        Number of particles

    N_runs : int
        Number of runs in the sequence

    Returns
    -------
    times : numpy.array
        The times of the run sequence
    """
    times = []

    _, _, _, gsf = sim_base.get_parts(
        N_particles=N_particle,
        gpu=True,
        pf=False
    )

    for _ in tqdm.tqdm(range(N_runs)):
        u, _ = sim_base.get_random_io()
        gsf.predict(u, 1.)

        t = time.time()
        # noinspection PyProtectedMember
        gsf._get_sigma_points()
        times.append(time.time() - t)

    return numpy.array(times)


# noinspection PyProtectedMember
@RunSequences.vectorize
@PickleJar.pickle(path='gsf/raw')
def predict_subs_run_seq(N_particle, N_runs):
    """Performs a run sequence on the prediction function's subroutines
     with the given number of particles and number of runs

    Parameters
    ----------
    N_particle : int
        Number of particles

    N_runs : int
        Number of runs in the sequence

    Returns
    -------
    times : numpy.array
        The times of the run sequence
    """
    timess = []

    _, _, _, gsf = sim_base.get_parts(
        N_particles=N_particle,
        gpu=True,
        pf=False
    )

    dt = 1.
    for _ in tqdm.tqdm(range(N_runs)):
        u, _ = sim_base.get_random_io()
        gsf.predict(u, 1.)

        times = []
        t = time.time()
        sigmas = gsf._get_sigma_points()
        times.append(time.time() - t)

        # Move the sigma points through the state transition function
        t = time.time()
        u = cupy.asarray(u)
        times.append(time.time() - t)

        t = time.time()
        sigmas += gsf.f_vectorize(sigmas, u, dt)
        times.append(time.time() - t)

        t = time.time()
        sigmas += gsf.state_pdf.draw((gsf.N_particles, gsf._N_sigmas))
        times.append(time.time() - t)

        t = time.time()
        gsf.means = cupy.average(sigmas, axis=1, weights=gsf._w_sigma)
        times.append(time.time() - t)

        t = time.time()
        sigmas -= gsf.means[:, None, :]
        gsf.covariances = sigmas.swapaxes(1, 2) @ (sigmas * gsf._w_sigma[:, None])
        times.append(time.time() - t)

        timess.append(times)

    return numpy.array(timess)


# noinspection PyProtectedMember
@RunSequences.vectorize
@PickleJar.pickle(path='gsf/raw')
def update_subs_run_seq(N_particle, N_runs):
    """Performs a run sequence on the update function's subroutines
     with the given number of particles and number of runs

    Parameters
    ----------
    N_particle : int
        Number of particles

    N_runs : int
        Number of runs in the sequence

    Returns
    -------
    times : numpy.array
        The times of the run sequence
    """
    timess = []

    _, _, _, gsf = sim_base.get_parts(
        N_particles=N_particle,
        gpu=True,
        pf=False
    )

    for _ in tqdm.tqdm(range(N_runs)):
        u, z = sim_base.get_random_io()
        gsf.predict(u, 1.)

        times = []
        t = time.time()
        # Local Update
        sigmas = gsf._get_sigma_points()
        times.append(time.time() - t)

        # Move the sigma points through the state observation function
        t = time.time()
        u = cupy.asarray(u)
        times.append(time.time() - t)

        t = time.time()
        etas = gsf.g_vectorize(sigmas, u, gsf._y_dummy)
        times.append(time.time() - t)

        # Compute the Kalman gain
        t = time.time()
        eta_means = cupy.average(etas, axis=1, weights=gsf._w_sigma)
        sigmas -= gsf.means[:, None, :]
        etas -= eta_means[:, None, :]

        P_xys = sigmas.swapaxes(1, 2) @ (etas * gsf._w_sigma[:, None])
        P_yys = etas.swapaxes(1, 2) @ (etas * gsf._w_sigma[:, None])
        P_yy_invs = cupy.linalg.inv(P_yys)
        Ks = P_xys @ P_yy_invs
        times.append(time.time() - t)

        # Use the gain to update the means and covariances
        t = time.time()
        z = cupy.asarray(z, dtype=cupy.float32)
        times.append(time.time() - t)

        t = time.time()
        es = z - eta_means
        gsf.means += (Ks @ es[:, :, None]).squeeze()
        # Dimensions from paper do not work, use corrected version
        gsf.covariances -= Ks @ P_yys @ Ks.swapaxes(1, 2)
        times.append(time.time() - t)

        # Global Update
        # Move the means through the state observation function
        t = time.time()
        y_means = gsf.g_vectorize(gsf.means, u, gsf._y_dummy)
        times.append(time.time() - t)

        t = time.time()
        glob_es = z - y_means
        gsf.weights *= gsf.measurement_pdf.pdf(glob_es)
        times.append(time.time() - t)

        timess.append(times)

    return numpy.array(timess)


# noinspection PyProtectedMember
@RunSequences.vectorize
@PickleJar.pickle(path='gsf/raw')
def resample_subs_run_seq(N_particle, N_runs):
    """Performs a run sequence on the resample function's subroutines
     with the given number of particles and number of runs

    Parameters
    ----------
    N_particle : int
        Number of particles

    N_runs : int
        Number of runs in the sequence

    Returns
    -------
    times : numpy.array
        The times of the run sequence
    """
    timess = []

    _, _, _, gsf = sim_base.get_parts(
        N_particles=N_particle,
        gpu=True,
        pf=False
    )

    for _ in tqdm.tqdm(range(N_runs)):
        u, z = sim_base.get_random_io()
        gsf.predict(u, 1.)

        times = []
        t = time.time()
        t_weights = torch_dlpack.from_dlpack(cupy.asarray(gsf.weights).toDlpack())
        t_cumsum = torch.cumsum(t_weights, 0)
        cumsum = cupy.fromDlpack(torch_dlpack.to_dlpack(t_cumsum))
        cumsum /= cumsum[-1]
        times.append(time.time() - t)

        t = time.time()
        sample_index = cupy.zeros(gsf.N_particles, dtype=cupy.int64)
        random_number = cupy.float64(cupy.random.rand())

        if gsf.N_particles >= 1024:
            threads_per_block = 1024
            blocks_per_grid = (gsf.N_particles - 1) // threads_per_block + 1
        else:
            div_32 = (gsf.N_particles - 1) // 32 + 1
            threads_per_block = 32 * div_32
            blocks_per_grid = 1

        filter.gs_ukf.ParallelGaussianSumUnscentedKalmanFilter._parallel_resample[blocks_per_grid, threads_per_block](
            cumsum, sample_index, random_number, gsf.N_particles
        )
        times.append(time.time() - t)

        t = time.time()
        gsf.means = cupy.asarray(gsf.means)[sample_index]
        gsf.covariances = cupy.asarray(gsf.covariances)[sample_index]
        gsf.weights = cupy.full(gsf.N_particles, 1 / gsf.N_particles)
        times.append(time.time() - t)

        timess.append(times)

    return numpy.array(timess)


@RunSequences.vectorize
@PickleJar.pickle(path='gsf/raw')
def no_op_run_seq(N_time, N_runs):
    """Performs a run sequence on a no-op routine with the given sleep time
     and number of runs

    Parameters
    ----------
    N_time : float
        Sleep time

    N_runs : int
        Number of runs in the sequence

    Returns
    -------
    times : numpy.array
        The times of the run sequence
    """
    times = []

    for _ in tqdm.tqdm(range(N_runs)):
        time.sleep(N_time)
        t = time.time()
        times.append(time.time() - t)

    return numpy.array(times)


@RunSequences.vectorize
@PickleJar.pickle(path='gsf/raw')
def time_time_run_seq(N_time, N_runs):
    """Performs a run sequence on the time.time() function with the given sleep time
     and number of runs

    Parameters
    ----------
    N_time : float
        Sleep time

    N_runs : int
        Number of runs in the sequence

    Returns
    -------
    times : numpy.array
        The times of the run sequence
    """
    times = []

    for _ in tqdm.tqdm(range(N_runs)):
        time.sleep(N_time)
        t = time.time()
        time.time()
        times.append(time.time() - t)

    return numpy.array(times)


# noinspection PyTypeChecker
@PickleJar.pickle(path='gsf/processed')
def example_run_seqs():
    """Returns the run sequences for the no_op and time.time() methods

    Returns
    -------
    run_seqss : List
        [no_op; time_time] x [N_particles; run_seq]
    """
    N_times = numpy.array([0.])
    run_seqss = [
        no_op_run_seq(N_times, 100),
        time_time_run_seq(N_times, 100)
    ]
    return run_seqss


# noinspection PyTypeChecker
@PickleJar.pickle(path='gsf/processed')
def cpu_gpu_run_seqs():
    """Returns the run sequences for the predict, update and resample method

    Returns
    -------
    run_seqss : List
        [CPU; GPU] x [predict; update; resample] x [N_particles; run_seq]
    """
    N_particles_cpu = numpy.array([int(i) for i in 2**numpy.arange(0, 19, 0.5)])
    N_particles_gpu = numpy.array([int(i) for i in 2**numpy.arange(0, 19, 0.5)])
    run_seqss = [
        [
            predict_run_seq(N_particles_cpu, 20, False),
            update_run_seq(N_particles_cpu, 100, False),
            resample_run_seq(N_particles_cpu, 100, False)
        ],
        [
            predict_run_seq(N_particles_gpu, 100, True),
            update_run_seq(N_particles_gpu, 100, True),
            resample_run_seq(N_particles_gpu, 100, True)
        ]
    ]
    return run_seqss


# noinspection PyTypeChecker
@PickleJar.pickle(path='gsf/processed')
def gsf_sub_routine_run_seqs():
    """Returns the run sequences for the predict, update and resample subroutines

    Returns
    -------
    run_seqss : List
        [predict; update; resample] x [N_particles; run_seq]
    """
    N_particles_gpu = numpy.array([int(i) for i in 2**numpy.arange(1, 19, 0.5)])
    run_seqss = [
        predict_subs_run_seq(N_particles_gpu, 100),
        update_subs_run_seq(N_particles_gpu, 100),
        resample_subs_run_seq(N_particles_gpu, 100)
    ]
    return run_seqss


def plot_example_benchmark():
    """Plot the no_op run sequence, lag chart and autocorrelation graphs
    """

    run_seqss = example_run_seqs()

    matplotlib.rcParams.update({'font.size': 20})
    fig, axes = plt.subplots(2, 3, sharex='col', figsize=(6.25*3, 5*2))
    for i, (N_time, run_seqs) in enumerate(run_seqss):
        run_seq = run_seqs[0, :]
        ax = axes[i][0]
        ax.plot(run_seq, 'kx')
        if i == 0:
            ax.set_title('Run sequence', pad=12)
        else:
            ax.set_xlabel('Iterations')
        ax.set_ylabel('Time (s)')

        ax = axes[i][1]
        ax.plot(run_seq[:-1], run_seq[1:], 'kx')
        if i == 0:
            ax.set_title('Lag chart', pad=12)
        else:
            ax.set_xlabel(r'$X_{i-1}$')
        ax.set_ylabel(r'$X_{i}$')

        ax = axes[i][2]
        abs_cors = numpy.abs(stats_tools.pacf(run_seq, nlags=10)[1:])
        ax.plot(abs_cors, 'kx')
        if i == 0:
            ax.set_title('Autocorrelation graph', pad=12)
        else:
            ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')

    # plt.suptitle(r'Benchmarking for no-op and time.time()')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('benchmark.pdf')
    plt.show()


def plot_max_auto():
    """Plot the maximum autocorrelation for the predict, update and resample run sequences
    """
    run_seqss = cpu_gpu_run_seqs()

    matplotlib.rcParams.update({'font.size': 20})
    fig, axes = plt.subplots(2, 3, sharey='row', figsize=(6.25 * 3, 5 * 2))
    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            N_parts, run_seqs = run_seqss[row][col]
            N_logs = numpy.log2(N_parts)

            for N_log, run_seq in zip(N_logs, run_seqs):
                abs_cors = numpy.abs(stats_tools.pacf(run_seq, nlags=10)[1:])
                ax.plot(N_log, numpy.max(abs_cors), 'kx')
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 20)
            ax.axhline(0.2, color='r')
            ax.set_xlabel(r'$N_p$')
            ax.set_xticklabels('$2^{' + numpy.char.array(ax.get_xticks(), unicode=True) + '}$')

            if row == 0:
                ax.set_title(['Predict', 'Update', 'Resample'][col])

            if row == 0 and col == 0:
                ax.set_ylabel('CPU', rotation=0, labelpad=25)

            if col == 0 and row == 1:
                ax.set_ylabel('GPU', rotation=0, labelpad=25)
    # fig.suptitle('Maximum autocorrelation values')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('max_autocorrelation.pdf')
    plt.show()


# noinspection PyUnresolvedReferences
def plot_run_seqs():
    """Plot the run sequences for predict, update and resample functions
    """
    run_seqss = cpu_gpu_run_seqs()

    cmap = matplotlib.cm.get_cmap('Spectral')
    norm = matplotlib.colors.Normalize(vmin=1, vmax=24)

    for row in range(2):
        for col in range(3):
            plt.subplot(2, 3, row + 2*col + 1)
            N_parts, run_seqs = run_seqss[row][col]
            N_logs = numpy.log2(N_parts)

            for N_log, run_seq in zip(N_logs, run_seqs):
                normailised_value = (N_log - N_logs[0]) / N_logs[-1]
                plt.semilogy(run_seq, '.', color=cmap(normailised_value))

    plt.colorbar(
        matplotlib.cm.ScalarMappable(
            norm=norm,
            cmap=cmap
        ),
        ax=plt.gcf().get_axes()
    )

    plt.savefig('run_seqs.pdf')
    plt.show()


def plot_speed_up():
    """Plot the speed-up between CPU and GPU implementations
     for predict, update and resample functions
    """
    run_seqss = cpu_gpu_run_seqs()

    matplotlib.rcParams.update({'font.size': 9})
    plt.figure(figsize=(6.25, 5))
    for method in range(3):
        plt.yscale('log')
        logN_part = numpy.log2(run_seqss[0][method][0])
        cpu_times = run_seqss[0][method][1]
        gpu_times = run_seqss[1][method][1][:cpu_times.shape[0]]

        speed_up = numpy.median(cpu_times, axis=1) / numpy.median(gpu_times, axis=1)
        speed_up_err = numpy.abs(
            (
                    numpy.quantile(cpu_times, [0, 1], axis=1) /
                    numpy.quantile(gpu_times, [0, 1], axis=1)
            ) - speed_up
        )
        plt.errorbar(
            logN_part,
            speed_up,
            yerr=speed_up_err,
            fmt=['k.', 'kx', 'k^'][method],
            capsize=3,
            elinewidth=2,
            markeredgewidth=1,
            ecolor=(0, 0, 1, 0.3),
        )

        speed_up_err = numpy.abs(
            (
                    numpy.quantile(cpu_times, [0.1, 0.9], axis=1) /
                    numpy.quantile(gpu_times, [0.1, 0.9], axis=1)
            ) - speed_up
        )
        plt.errorbar(
            logN_part,
            speed_up,
            yerr=speed_up_err,
            fmt=['k.', 'kx', 'k^'][method],
            capsize=5,
            elinewidth=2,
            markeredgewidth=1,
            ecolor=(1, 0, 0, 1),
            label=['Predict', 'Update', 'Resample'][method]
        )

    plt.legend()
    ticks, _ = plt.xticks()
    plt.xticks(
        ticks,
        '$2^{' + numpy.char.array(ticks, unicode=True) + '}$'
    )

    # plt.title('Speed-up of Gaussian sum filter')
    plt.ylabel('Speed-up')
    plt.xlabel(r'$ N_p$')
    plt.xlim(xmin=1, xmax=19.5)
    plt.axhline(1, color='black', alpha=0.4)
    plt.tight_layout()
    plt.savefig('GSF_speedup.pdf')
    plt.show()


def plot_times():
    """Plot the run times of CPU and GPU implementations
     for predict, update and resample functions
    """
    run_seqss = cpu_gpu_run_seqs()

    matplotlib.rcParams.update({'font.size': 9})
    fig, axes = plt.subplots(3, 1, sharey='all', figsize=(6.25, 11))
    for device in range(2):
        for method in range(3):
            ax = axes[method]
            ax.set_yscale('log')
            timess = run_seqss[device][method][1]
            logN_part = numpy.log2(run_seqss[device][method][0])

            times = numpy.median(timess, axis=1)

            times_err = numpy.abs(numpy.quantile(timess, [0, 1], axis=1) - times)
            ax.errorbar(
                logN_part,
                times,
                yerr=times_err,
                fmt=['k.', 'kx'][device],
                capsize=0,
                elinewidth=2,
                markeredgewidth=1,
                ecolor=(0, 0, 1, 0.3),
            )

            times_err = numpy.abs(numpy.quantile(timess, [0.1, 0.9], axis=1) - times)
            ax.errorbar(
                logN_part,
                times,
                yerr=times_err,
                fmt=['k.', 'kx'][device],
                capsize=[5, 3][device],
                elinewidth=2,
                markeredgewidth=1,
                ecolor=(1, 0, 0, 1),
                label=['CPU', 'GPU'][device]
            )

            ax.legend()
            ax.set_title(['Predict', 'Update', 'Resample'][method])
            if method == 0:
                ax.set_ylabel('Time (s)')
            ax.set_xlabel(r'$ N_p$')
            ax.set_xlim(xmin=1, xmax=19.5)
            if device:
                ax.set_xticklabels('$2^{' + numpy.char.array(ax.get_xticks(), unicode=True) + '}$')
    # fig.suptitle('Run times Gaussian sum filter methods')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('GSF_times.pdf')
    plt.show()


def plot_sub_routine_fractions():
    """Plot the run time fractions of GPU implementation subroutines used in
     predict, update and resample functions
    """
    names = [
        [
            'sigma points', 'f (memory copy)', 'f', 'State noise - draw',
            'means', 'covariances'
        ],
        [
            'sigma points', 'g sigmas (memory copy)', 'g sigmas',
            'Kalman gain', 'g means (memory copy)', 'Kalman update', 'g means ',
            'Measurement noise - pdf'
        ],
        [
            'cumsum', 'Nicely algorithm', 'Index copying'
        ]
    ]

    func_seqss = gsf_sub_routine_run_seqs()

    matplotlib.rcParams.update({'font.size': 9})
    fig, axes = plt.subplots(3, 1, sharey='all', figsize=(6.25, 11))
    for i in range(3):
        ax = axes[i]
        N_parts, func_seqs = func_seqss[i]
        logN_part = numpy.log2(N_parts)

        total_times = numpy.sum(numpy.nanmin(func_seqs, axis=1), axis=1)
        frac_times = numpy.nanmin(func_seqs, axis=1).T / total_times
        ax.stackplot(logN_part, frac_times, labels=names[i])

        ax.legend()
        ax.set_title(['Predict', 'Update', 'Resample'][i])
        if i == 0:
            ax.set_ylabel('Fraction of runtime')
        ax.set_xticklabels('$2^{' + numpy.char.array(ax.get_xticks(), unicode=True) + '}$')
        ax.set_xlabel(r'$ N_p $')

    plt.tight_layout()
    plt.savefig('gsf_frac_breakdown.pdf')
    plt.show()


if __name__ == '__main__':
    plot_sub_routine_fractions()
    plot_example_benchmark()
    plot_max_auto()
    plot_times()
    plot_speed_up()
