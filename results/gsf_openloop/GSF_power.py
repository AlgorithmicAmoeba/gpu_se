import numpy
import matplotlib.pyplot as plt
import sim_base
import time
from decorators import PowerMeasurement


@PowerMeasurement.vectorize
def predict_power_seq(N_particle, t_run, gpu):
    """Performs a power sequence on the prediction function with the given number
    of particle and number of runs on the CPU or GPU

    Parameters
    ----------
    N_particle : int
        Number of particles

    t_run : float
        Minimum run time of the function. Repeats if the time is too short

    gpu : bool
        If `True` then the GPU implementation is used.
        Otherwise, the CPU implementation is used

    Returns
    -------
    runs : int
        The number of times the function was run
    """
    _, _, _, gsf = sim_base.get_parts(
        N_particles=N_particle,
        gpu=gpu,
        pf=False
    )

    t = time.time()
    runs = 0
    while time.time() - t < t_run:
        runs += 1
        u, _ = sim_base.get_random_io()
        gsf.predict(u, 1.)

    return runs


@PowerMeasurement.vectorize
def update_power_seq(N_particle, t_run, gpu):
    """Performs a power sequence on the update function with the given number
    of particle and number of runs on the CPU or GPU

    Parameters
    ----------
    N_particle : int
        Number of particles

    t_run : float
        Minimum run time of the function. Repeats if the time is too short

    gpu : bool
        If `True` then the GPU implementation is used.
        Otherwise, the CPU implementation is used

    Returns
    -------
    runs : int
        The number of times the function was run
    """
    _, _, _, gsf = sim_base.get_parts(
        N_particles=N_particle,
        gpu=gpu,
        pf=False
    )

    t = time.time()
    runs = 0
    while time.time() - t < t_run:
        runs += 1
        u, y = sim_base.get_random_io()
        gsf.update(u, y)

    return runs


@PowerMeasurement.vectorize
def resample_power_seq(N_particle, t_run, gpu):
    """Performs a power sequence on the resample function with the given number
    of particle and number of runs on the CPU or GPU

    Parameters
    ----------
    N_particle : int
        Number of particles

    t_run : float
        Minimum run time of the function. Repeats if the time is too short

    gpu : bool
        If `True` then the GPU implementation is used.
        Otherwise, the CPU implementation is used

    Returns
    -------
    runs : int
        The number of times the function was run
    """
    _, _, _, gsf = sim_base.get_parts(
        N_particles=N_particle,
        gpu=gpu,
        pf=False
    )

    t = time.time()
    runs = 0
    while time.time() - t < t_run:
        runs += 1
        gsf.weights = numpy.random.random(size=gsf.N_particles)
        gsf.weights /= numpy.sum(gsf.weights)
        gsf.resample()

    return runs


@PowerMeasurement.vectorize
def nothing_power_seq(N_particle, t_run):
    """Performs a power sequence on the no-op function with the given number
    of particle and number of runs on the CPU or GPU.
    Used to check default power usage

    Parameters
    ----------
    N_particle : int
        Number of particles

    t_run : float
        Minimum run time of the function. Repeats if the time is too short

    Returns
    -------
    runs : int
        The number of times the function was run
    """
    _ = N_particle
    t = time.time()
    runs = 0
    while time.time() - t < t_run:
        runs += 1
        time.sleep(246)

    return runs


# noinspection PyTypeChecker
def cpu_gpu_power_seqs():
    """Returns the power sequences for all the runs

    Returns
    -------
    power_seqss : List
        [CPU; GPU] x [predict; update; resample] x [N_particles; power_seq]
    """
    N_particles_cpu = numpy.array([int(i) for i in 2**numpy.arange(1, 19, 0.5)])
    N_particles_gpu = numpy.array([int(i) for i in 2**numpy.arange(1, 19, 0.5)])
    power_seqss = [
        [
            predict_power_seq(N_particles_cpu, 5, False),
            update_power_seq(N_particles_cpu, 5, False),
            resample_power_seq(N_particles_cpu, 5, False)
        ],
        [
            predict_power_seq(N_particles_gpu, 5, True),
            update_power_seq(N_particles_gpu, 5, True),
            resample_power_seq(N_particles_gpu, 5, True)
        ]
    ]
    return power_seqss


def plot_energy_per_run():
    """Plot the energy per run of CPU and GPU implementations of the
     predict, update and resample functions
    """
    powerss = cpu_gpu_power_seqs()
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey='all')
    plt.rcParams.update({'font.size': 12})

    for cpu_gpu in range(2):
        for method in range(3):
            ax = axes[method]

            N_parts, powers = powerss[cpu_gpu][method]
            N_logs = numpy.log2(N_parts)
            total_power = powers[:, 0]
            if cpu_gpu:
                total_power += powers[:, 1]

            ax.semilogy(
                N_logs,
                total_power,
                '.',
                label=['CPU', 'GPU'][cpu_gpu]
            )
            ax.legend()
            ax.set_xlabel(r'$\log_2(N_p)$', fontsize=12)
            if method == 0:
                ax.set_ylabel(r'$\frac{\mathrm{J}}{\mathrm{run}}$', fontsize=12)
            ax.set_title(['Predict', 'Update', 'Resample'][method])

    fig.suptitle('Energy per run')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('energy_per_run.pdf')
    plt.show()


if __name__ == '__main__':
    plot_energy_per_run()
