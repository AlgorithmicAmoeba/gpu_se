import numpy
import matplotlib.pyplot as plt
import sim_base
import joblib
import subprocess
import psutil
import time
import multiprocessing
import scipy.integrate


class PowerMeasurement:
    """A class to measure power drawn for functions.
        Specifically designed to allow vectorization of the process
        of power measurement

        Parameters
        ----------
        function : callable
            The function to be vectorized/managed

        path : string, optional
            Location where joblib cache should be recalled and saved to

        CPU_max_power : float, optional
            The power the CPU draws at 100% use
        """
    def __init__(self, function, path='cache/', CPU_max_power=30):
        self._memory = joblib.Memory(path + function.__name__)
        self.function = function
        self.CPU_max_power = CPU_max_power
        self._particle_call = self._particle_call_gen()

    def __call__(self, N_particles, t_run, *args, **kwargs):
        powers = numpy.array(
            [
                self._particle_call(N_particle, t_run, *args, **kwargs)
                for N_particle in N_particles
            ]
        )
        powers[:, 0] *= self.CPU_max_power
        return N_particles, powers

    def _particle_call_gen(self):
        """Generates the function that spawns the power measurement process
        and runs the function for the required amount of time"""

        @self._memory.cache
        def particle_call(N_particle, t_run, *args, **kwargs):
            queue = multiprocessing.Queue()
            power_process = multiprocessing.Process(
                target=PowerMeasurement._power_seq,
                args=(queue,)
            )
            power_process.start()
            N_runs = self.function(N_particle, t_run, *args, **kwargs)
            queue.put('Done')

            while queue.qsize() < 2:
                time.sleep(0.3)

            queue.get()
            power_seq = queue.get()

            power = scipy.integrate.trapz(power_seq[1:, :], power_seq[0], axis=1) / N_runs

            queue.close()
            queue.join_thread()
            power_process.join()

            return power

        return particle_call

    def clear(self, *args):
        """Clears the stored result of the function with the arguments given

        Parameters
        ----------
        args : tuple
            Arguments of the function
        """
        self._particle_call.call_and_shelve(*args).clear()

    @staticmethod
    def vectorize(function):
        """Decorator function that creates a callable PowerMeasurement class

        Parameters
        ----------
        function : function to be vectorized/managed

        Returns
        -------
        pm : RunSequences
            The PowerMeasurement object that handles vectorized calls
        """
        return PowerMeasurement(function)

    @staticmethod
    def get_GPU_power():
        """Uses the nvidia-smi interface to query the current power drawn by the GPU

        Returns
        -------
        power : float
            The current power drawn by the GPU
        """
        return float(
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"
                 ]
            )
        )

    @staticmethod
    def get_CPU_frac():
        """Uses the psutil library to query the current CPU usage
        fraction

        Returns
        -------
        frac : float
            The current current CPU usage fraction
        """
        return psutil.cpu_percent()/100

    @staticmethod
    def _power_seq(q):
        """A function meant to be run in parallel with another function.
        This function takes readings of the CPU usage percentage and GPU power usage.
        Parameters
        ----------
        q : multiprocessing.Queue
            A thread safe method of message passing between the host process and this one.
            Allows this process to return a numpy array of measurements
        """
        times, cpu_frac, gpu_power = [], [], []

        while q.empty():
            times.append(time.time())
            cpu_frac.append(PowerMeasurement.get_CPU_frac())
            gpu_power.append(PowerMeasurement.get_GPU_power())
            time.sleep(0.2)

        q.put(numpy.array([times, cpu_frac, gpu_power]))


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
    _, _, _, p = sim_base.get_parts(
        N_particles=N_particle,
        gpu=gpu
    )

    t = time.time()
    runs = 0
    while time.time() - t < t_run:
        runs += 1
        u, _ = sim_base.get_random_io()
        p.predict(u, 1.)

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
    _, _, _, p = sim_base.get_parts(
        N_particles=N_particle,
        gpu=gpu
    )

    t = time.time()
    runs = 0
    while time.time() - t < t_run:
        runs += 1
        u, y = sim_base.get_random_io()
        p.update(u, y)

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
    _, _, _, p = sim_base.get_parts(
        N_particles=N_particle,
        gpu=gpu
    )

    t = time.time()
    runs = 0
    while time.time() - t < t_run:
        runs += 1
        p.weights = numpy.random.random(size=p.N_particles)
        p.weights /= numpy.sum(p.weights)
        p.resample()

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
    N_particles_cpu = numpy.array([int(i) for i in 2**numpy.arange(1, 24, 0.5)])
    N_particles_gpu = numpy.array([int(i) for i in 2**numpy.arange(1, 24, 0.5)])
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
