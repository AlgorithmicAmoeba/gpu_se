import numpy
import time
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import sim_base
import joblib
import cupy
import statsmodels.tsa.stattools as stats_tools
import torch
import torch.utils.dlpack as torch_dlpack
import filter.particle


class RunSequences:
    """A class to manage run sequences for functions.
    Specifically designed to allow vectorization of the process

    Parameters
    ----------
    function : callable
        The function to be vectorized/managed

    path : string
        Location where joblib cache should be recalled and saved to
    """
    def __init__(self, function, path='cache/'):
        self._memory = joblib.Memory(path + function.__name__)
        self.function = self._memory.cache(function)

    def __call__(self, N_particles, N_runs, *args, **kwargs):
        run_seqs = numpy.array(
            [self.function(N_particle, N_runs, *args, **kwargs) for N_particle in N_particles]
        )

        return N_particles, run_seqs

    def clear(self, *args):
        """Clears the stored result of the function with the arguments given

        Parameters
        ----------
        args : tuple
            Arguments of the function
        """
        self.function.call_and_shelve(*args).clear()

    @staticmethod
    def vectorize(function):
        """Decorator function that creates a callable RunSequences class

        Parameters
        ----------
        function : function to be vectorized/managed

        Returns
        -------
        rs : RunSequences
            The RunSequences object that handles vectorized calls
        """
        return RunSequences(function)


@RunSequences.vectorize
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


@RunSequences.vectorize
def f_vectorize_run_seq(N_particle, N_runs, mem_gpu):
    """Performs a run sequence on the f_vectorize function with the given number
    of particle and number of runs

    Parameters
    ----------
    N_particle : int
        Number of particles

    N_runs : int
        Number of runs in the sequence

    mem_gpu : bool
        If `True` then the GPU memory is used.
        Otherwise, the CPU memory is used

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
        if mem_gpu:
            u = cupy.asarray(u)
        gsf.predict(u, 1.)

        t = time.time()
        gsf.f_vectorize(gsf.means, u, 1.)
        times.append(time.time() - t)

    return numpy.array(times)


@RunSequences.vectorize
def state_pdf_draw_run_seq(N_particle, N_runs):
    """Performs a run sequence on the state_pdf.draw() function with the given number
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
        t = time.time()
        gsf.state_pdf.draw(gsf.N_particles)
        times.append(time.time() - t)

    return numpy.array(times)


# noinspection PyProtectedMember
@RunSequences.vectorize
def means_run_seq(N_particle, N_runs):
    """Performs a run sequence on the means subprocess in prediction
    with the given number of particle and number of runs

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
        sigmas = gsf._get_sigma_points()

        t = time.time()
        gsf.means = cupy.average(sigmas, axis=1, weights=gsf._w_sigma)
        times.append(time.time() - t)

    return numpy.array(times)


# noinspection PyProtectedMember
@RunSequences.vectorize
def covariances_run_seq(N_particle, N_runs):
    """Performs a run sequence on the covariances subprocess in prediction
    with the given number of particle and number of runs

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
        sigmas = gsf._get_sigma_points()
        gsf.means = cupy.average(sigmas, axis=1, weights=gsf._w_sigma)

        t = time.time()
        sigmas -= gsf.means[:, None, :]
        gsf.covariances = sigmas.swapaxes(1, 2) @ (sigmas * gsf._w_sigma[:, None])
        times.append(time.time() - t)

    return numpy.array(times)


# noinspection PyProtectedMember
@RunSequences.vectorize
def g_vectorize_sigmas_run_seq(N_particle, N_runs, mem_gpu):
    """Performs a run sequence on the g_vectorize function with the given number
    of particle and number of runs

    Parameters
    ----------
    N_particle : int
        Number of particles

    N_runs : int
        Number of runs in the sequence

    mem_gpu : bool
        If `True` then the GPU memory is used.
        Otherwise, the CPU memory is used

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
        if mem_gpu:
            u = cupy.asarray(u)
        sigmas = gsf._get_sigma_points()

        t = time.time()
        gsf.g_vectorize(sigmas, u, gsf._y_dummy)
        times.append(time.time() - t)

    return numpy.array(times)


# noinspection PyProtectedMember
@RunSequences.vectorize
def kalman_gain_run_seq(N_particle, N_runs):
    """Performs a run sequence on the kalman gain subprocess with the given number
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
        sigmas = gsf._get_sigma_points()
        etas = gsf.g_vectorize(sigmas, u, gsf._y_dummy)

        t = time.time()
        eta_means = cupy.average(etas, axis=1, weights=gsf._w_sigma)
        sigmas -= gsf.means[:, None, :]
        etas -= eta_means[:, None, :]
        P_xys = sigmas.swapaxes(1, 2) @ (etas * gsf._w_sigma[:, None])
        P_yys = etas.swapaxes(1, 2) @ (etas * gsf._w_sigma[:, None])
        P_yy_invs = cupy.linalg.inv(P_yys)
        Ks = P_xys @ P_yy_invs
        times.append(time.time() - t)

        _ = Ks

    return numpy.array(times)


# noinspection PyProtectedMember
@RunSequences.vectorize
def kalman_update_run_seq(N_particle, N_runs):
    """Performs a run sequence on the kalman gain subprocess with the given number
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
        u, z = sim_base.get_random_io()
        gsf.predict(u, 1.)
        sigmas = gsf._get_sigma_points()
        etas = gsf.g_vectorize(sigmas, u, gsf._y_dummy)
        eta_means = cupy.average(etas, axis=1, weights=gsf._w_sigma)
        sigmas -= gsf.means[:, None, :]
        etas -= eta_means[:, None, :]
        P_xys = sigmas.swapaxes(1, 2) @ (etas * gsf._w_sigma[:, None])
        P_yys = etas.swapaxes(1, 2) @ (etas * gsf._w_sigma[:, None])
        P_yy_invs = cupy.linalg.inv(P_yys)
        Ks = P_xys @ P_yy_invs

        t = time.time()
        z = cupy.asarray(z, dtype=cupy.float32)
        es = z - eta_means
        gsf.means += (Ks @ es[:, :, None]).squeeze()
        gsf.covariances -= Ks @ P_yys @ Ks.swapaxes(1, 2)
        times.append(time.time() - t)
        _ = Ks

    return numpy.array(times)


# noinspection PyProtectedMember
@RunSequences.vectorize
def g_vectorize_means_run_seq(N_particle, N_runs, mem_gpu):
    """Performs a run sequence on the g_vectorize function with the given number
    of particle and number of runs

    Parameters
    ----------
    N_particle : int
        Number of particles

    N_runs : int
        Number of runs in the sequence

    mem_gpu : bool
        If `True` then the GPU memory is used.
        Otherwise, the CPU memory is used

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
        if mem_gpu:
            u = cupy.asarray(u)

        t = time.time()
        gsf.g_vectorize(gsf.means, u, gsf._y_dummy)
        times.append(time.time() - t)

    return numpy.array(times)


@RunSequences.vectorize
def measurement_pdf_run_seq(N_particle, N_runs):
    """Performs a run sequence on the measurement_pdf.draw() function with the given number
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
        es = gsf.measurement_pdf.draw(gsf.N_particles)

        t = time.time()
        gsf.measurement_pdf.pdf(es)
        times.append(time.time() - t)

    return numpy.array(times)


@RunSequences.vectorize
def cumsum_run_seq(N_particle, N_runs):
    """Performs a run sequence on the cumsum subpart with the given number
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
        gsf.weights = cupy.random.uniform(size=gsf.N_particles)

        t = time.time()
        t_weights = torch_dlpack.from_dlpack(cupy.asarray(gsf.weights).toDlpack())
        t_cumsum = torch.cumsum(t_weights, 0)
        cumsum = cupy.fromDlpack(torch_dlpack.to_dlpack(t_cumsum))
        cumsum /= cumsum[-1]
        times.append(time.time() - t)

    return numpy.array(times)


# noinspection PyProtectedMember
@RunSequences.vectorize
def parallel_resample_run_seq(N_particle, N_runs):
    """Performs a run sequence on the Nicely resample subpart with the given number
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
        gsf.weights = cupy.random.uniform(size=gsf.N_particles)
        t_weights = torch_dlpack.from_dlpack(cupy.asarray(gsf.weights).toDlpack())
        t_cumsum = torch.cumsum(t_weights, 0)
        cumsum = cupy.fromDlpack(torch_dlpack.to_dlpack(t_cumsum))
        cumsum /= cumsum[-1]

        t = time.time()
        sample_index = cupy.zeros(gsf.N_particles, dtype=cupy.int64)
        random_number = cupy.float64(cupy.random.rand())

        filter.particle.ParallelParticleFilter._parallel_resample[gsf._bpg, gsf._tpb](
            cumsum, sample_index,
            random_number,
            gsf.N_particles
        )
        times.append(time.time() - t)

    return numpy.array(times)


# noinspection PyProtectedMember
@RunSequences.vectorize
def index_copying_run_seq(N_particle, N_runs):
    """Performs a run sequence on the index copying subpart with the given number
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
        gsf.weights = cupy.random.uniform(size=gsf.N_particles)
        t_weights = torch_dlpack.from_dlpack(cupy.asarray(gsf.weights).toDlpack())
        t_cumsum = torch.cumsum(t_weights, 0)
        cumsum = cupy.fromDlpack(torch_dlpack.to_dlpack(t_cumsum))
        cumsum /= cumsum[-1]

        sample_index = cupy.zeros(gsf.N_particles, dtype=cupy.int64)
        random_number = cupy.float64(cupy.random.rand())

        filter.particle.ParallelParticleFilter._parallel_resample[gsf._bpg, gsf._tpb](
            cumsum, sample_index,
            random_number,
            gsf.N_particles
        )

        t = time.time()
        gsf.means = cupy.asarray(gsf.means)[sample_index]
        gsf.covariances = cupy.asarray(gsf.covariances)[sample_index]
        gsf.weights = cupy.full(gsf.N_particles, 1 / gsf.N_particles)
        times.append(time.time() - t)

    return numpy.array(times)


@RunSequences.vectorize
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
        t = time.time()
        time.sleep(N_time)
        times.append(time.time() - t)

    return numpy.array(times)


# noinspection PyTypeChecker
def cpu_gpu_run_seqs():
    """Returns the run sequences for the predict, update and resample method

    Returns
    -------
    run_seqss : List
        [CPU; GPU] x [predict; update; resample] x [N_particles; run_seq]
    """
    N_particles_cpu = numpy.array([int(i) for i in 2**numpy.arange(1, 20, 0.5)])
    N_particles_gpu = numpy.array([int(i) for i in 2**numpy.arange(1, 20, 0.5)])
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


def pf_sub_routine_run_seqs():
    """Returns the run sequences for the subroutines

    Returns
    -------
    run_seqss : List
        [subroutines; N_particles; run_seq]
    """
    N_particles = [int(i) for i in 2**numpy.arange(1, 19, 0.5)]
    f_vec_gpu_mem = lambda x, y: f_vectorize_run_seq(x, y, True)
    f_vec_cpu_mem = lambda x, y: f_vectorize_run_seq(x, y, False)
    g_sigma_gpu_mem = lambda x, y: g_vectorize_sigmas_run_seq(x, y, True)
    g_sigma_cpu_mem = lambda x, y: g_vectorize_sigmas_run_seq(x, y, False)
    g_mean_gpu_mem = lambda x, y: g_vectorize_means_run_seq(x, y, True)
    g_mean_cpu_mem = lambda x, y: g_vectorize_means_run_seq(x, y, False)

    funcs = [
        sigma_points_run_seq, f_vec_gpu_mem, f_vec_cpu_mem, state_pdf_draw_run_seq,
        means_run_seq, covariances_run_seq,
        g_sigma_gpu_mem, g_sigma_cpu_mem, kalman_gain_run_seq,
        kalman_update_run_seq, g_mean_gpu_mem, g_mean_cpu_mem, measurement_pdf_run_seq,
        cumsum_run_seq, parallel_resample_run_seq, index_copying_run_seq
             ]

    run_seqs = [func(N_particles, 50) for func in funcs]

    return run_seqs


def plot_example_benchmark():
    """Plot the no_op run sequence, lag chart and autocorrelation graphs
    """
    N_times = numpy.array([10.])
    # noinspection PyTypeChecker
    N_times, run_seqs = no_op_run_seq(N_times, 100)
    run_seq = run_seqs[-1]

    print(N_times[-1])

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.semilogy(run_seq, 'kx')
    plt.title('Run sequence')
    plt.xlabel('Iterations')
    plt.ylabel('Time (s)')

    plt.subplot(1, 3, 2)
    plt.plot(run_seq[:-1], run_seq[1:], 'kx')
    plt.title('Lag chart')
    plt.xlabel(r'$X_{i-1}$')
    plt.ylabel(r'$X_{i}$')

    plt.subplot(1, 3, 3)
    abs_cors = numpy.abs(stats_tools.pacf(run_seq, nlags=10)[1:])
    plt.plot(abs_cors, 'kx')
    plt.title('Autocorrelation graph')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')

    plt.suptitle(r'Benchmarking for CPU update with $N_p = 2^{19.5}$')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('benchmark.pdf')
    plt.show()


def plot_max_auto():
    """Plot the maximum autocorrelation for the predict, update and resample run sequences
    """
    run_seqss = cpu_gpu_run_seqs()

    fig, axes = plt.subplots(2, 3, sharey='row')
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
            ax.set_xlabel(r'$\log_2(N_p)$')

            if row == 0:
                ax.set_title(['Predict', 'Update', 'Resample'][col])

            if row == 0 and col == 0:
                ax.set_ylabel('CPU', rotation=0)

            if col == 0 and row == 1:
                ax.set_ylabel('GPU', rotation=0)
    fig.suptitle('Maximum autocorrelation values')
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

    for method in range(3):
        cpu_time = numpy.min(run_seqss[0][method][1], axis=1)
        gpu_time = numpy.min(run_seqss[1][method][1], axis=1)

        speed_up = cpu_time / gpu_time[:cpu_time.shape[0]]
        logN_part = numpy.log2(run_seqss[0][method][0])
        plt.semilogy(logN_part, speed_up, ['k.', 'kx', 'k^'][method])

    plt.legend(['Predict', 'Update', 'Resample'])
    plt.title('Speed-up of particle filter')
    plt.ylabel('Speed-up')
    plt.xlabel('$ \log_2(N) $ particles')
    plt.xlim(xmin=1, xmax=19.5)
    plt.axhline(1, color='black', alpha=0.4)
    plt.tight_layout()
    plt.savefig('PF_speedup.pdf')
    plt.show()


def plot_times():
    """Plot the run times of CPU and GPU implementations
     for predict, update and resample functions
    """
    run_seqss = cpu_gpu_run_seqs()

    fig, axes = plt.subplots(1, 3, sharey='all', figsize=(15, 5))
    for device in range(2):
        for method in range(3):
            ax = axes[method]
            times = numpy.min(run_seqss[device][method][1], axis=1)
            logN_part = numpy.log2(run_seqss[device][method][0])
            ax.semilogy(logN_part, times, ['k.', 'kx'][device])
            ax.set_title(['Predict', 'Update', 'Resample'][method])

            ax.legend(['CPU', 'GPU'])
            if method == 0:
                ax.set_ylabel('Time (s)')
            ax.set_xlabel('$ \log_2(N) $ particles')
            ax.set_xlim(xmin=1, xmax=19.5)
    fig.suptitle('Run times particle filter methods')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('GSF_times.pdf')
    plt.show()


def plot_sub_routine_max_auto():
    """Plot the autocorrelation of GPU implementation subroutines used in
     predict, update and resample functions
    """
    run_seqss = pf_sub_routine_run_seqs()
    names = [
        'sigma points', 'f (GPU memory)', 'f (CPU memory)', 'State noise - draw',
        'means', 'covariances', 'g sigmas (GPU memory)', 'g sigmas (CPU memory)',
        'Kalman gain', 'Kalman update', 'g means (GPU memory)', 'g means (CPU memory)',
        'Measurement noise - pdf', 'cumsum', 'Nicely algorithm', 'Index copying'
     ]

    fig, axes = plt.subplots(3, 3, sharey='row', sharex='col')
    for i, (N_parts, run_seqs) in enumerate(run_seqss):
        ax = axes.flatten()[i]
        N_logs = numpy.log2(N_parts)

        for N_log, run_seq in zip(N_logs, run_seqs):
            abs_cors = numpy.abs(stats_tools.pacf(run_seq, nlags=10)[1:])
            ax.plot(N_log, numpy.max(abs_cors), 'kx')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 20)
        ax.axhline(0.2, color='r')
        if i > 5:
            ax.set_xlabel(r'$\log_2(N_p)$')
        ax.set_title(names[i])
    fig.suptitle('Maximum autocorrelation values')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('max_autocorrelation_subroutine.pdf')
    plt.show()


def plot_sub_routine_times():
    """Plot the run times of GPU implementation subroutines used in
     predict, update and resample functions
    """
    names = [
        'sigma points', 'f', 'f (memory copy)', 'State noise - draw',
        'means', 'covariances', 'g sigmas', 'g sigmas (memory copy)',
        'Kalman gain', 'Kalman update', 'g means ', 'g means (memory copy)',
        'Measurement noise - pdf', 'cumsum', 'Nicely algorithm', 'Index copying'
     ]

    run_seqss = pf_sub_routine_run_seqs()

    run_seqss[2] = (run_seqss[2][0], run_seqss[2][1] - run_seqss[1][1])
    run_seqss[7] = (run_seqss[7][0], run_seqss[7][1] - run_seqss[6][1])
    run_seqss[11] = (run_seqss[11][0], run_seqss[11][1] - run_seqss[10][1])

    for i, (N_parts, run_seqs) in enumerate(run_seqss):
        plt.subplot(3, 3, i + 1)
        times = numpy.min(run_seqs, axis=1)
        logN_part = numpy.log2(N_parts)
        plt.semilogy(logN_part, times, '.')
        plt.title(names[i])

    plt.show()


def plot_sub_routine_fractions():
    """Plot the run time fractions of GPU implementation subroutines used in
     predict, update and resample functions
    """
    names = [
        'sigma points', 'f', 'f (memory copy)', 'State noise - draw',
        'means', 'covariances', 'g sigmas', 'g sigmas (memory copy)',
        'Kalman gain', 'Kalman update', 'g means ', 'g means (memory copy)',
        'Measurement noise - pdf', 'cumsum', 'Nicely algorithm', 'Index copying'
    ]

    func_seqss = pf_sub_routine_run_seqs()

    func_seqss[2] = (func_seqss[2][0], func_seqss[2][1] - func_seqss[1][1])
    func_seqss[7] = (func_seqss[7][0], func_seqss[7][1] - func_seqss[6][1])
    func_seqss[11] = (func_seqss[11][0], func_seqss[11][1] - func_seqss[10][1])

    plt.figure(figsize=(15, 5))
    for i, func_indxs in enumerate([
        [0, 1, 2, 3, 4, 5],
        [0, 6, 7, 8, 9, 10, 11, 12],
        [13, 14, 15]
    ]):
        plt.subplot(1, 3, i+1)
        N_parts = func_seqss[0][0]
        logN_part = numpy.log2(N_parts)

        total_times = numpy.zeros_like(N_parts, dtype=numpy.float)
        for func_indx in func_indxs:
            total_times += numpy.min(abs(func_seqss[func_indx][1]), axis=1)

        bottom = None
        for j, func_indx in enumerate(func_indxs):
            times = numpy.min(abs(func_seqss[func_indx][1]), axis=1)
            frac_times = times / total_times
            plt.bar(
                logN_part,
                frac_times,
                width=0.55,
                bottom=bottom,
                label=names[func_indx],
                # color=['#292929', '#c2c2c2', '#808080'][j]
            )
            if bottom is None:
                bottom = frac_times
            else:
                bottom += frac_times
        plt.legend()
        plt.title(['Predict', 'Update', 'Resample'][i])
        if i == 0:
            plt.ylabel('Fraction of runtime')
        plt.xlabel(r'$\log_2(N_p)$')

    plt.tight_layout()
    plt.savefig('gsf_frac_breakdown.pdf')
    plt.show()


if __name__ == '__main__':
    plot_sub_routine_fractions()
    # plot_example_benchmark()
    # plot_sub_routine_max_auto()
    # plot_max_auto()
    # plot_times()
    # plot_speed_up()