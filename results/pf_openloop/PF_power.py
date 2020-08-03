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
    def __init__(self, function, path='cache/', CPU_max_power=30):
        self.memory = joblib.Memory(path + function.__name__)
        self.function = function
        self.CPU_max_power = CPU_max_power
        self.particle_call = self.particle_call_gen()

    def __call__(self, N_particles, t_run, *args, **kwargs):
        powers = numpy.array(
            [
                self.particle_call(N_particle, t_run, *args, **kwargs)
                for N_particle in N_particles
            ]
        )
        powers[:, 0] *= self.CPU_max_power
        return N_particles, powers

    def particle_call_gen(self):

        @self.memory.cache
        def particle_call(N_particle, t_run, *args, **kwargs):
            queue = multiprocessing.Queue()
            power_process = multiprocessing.Process(
                target=PowerMeasurement.power_seq,
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
        self.particle_call.call_and_shelve(*args).clear()

    @staticmethod
    def vectorize(function):
        return PowerMeasurement(function)

    @staticmethod
    def get_GPU_power():
        return float(
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"
                 ]
            )
        )

    @staticmethod
    def get_CPU_frac():
        return psutil.cpu_percent()/100

    @staticmethod
    def power_seq(q):
        times, cpu_frac, gpu_power = [], [], []

        while q.empty():
            times.append(time.time())
            cpu_frac.append(PowerMeasurement.get_CPU_frac())
            gpu_power.append(PowerMeasurement.get_GPU_power())
            time.sleep(0.2)

        q.put(numpy.array([times, cpu_frac, gpu_power]))


@PowerMeasurement.vectorize
def predict_power_seq(N_particle, t_run, gpu):
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
    _ = N_particle
    t = time.time()
    runs = 0
    while time.time() - t < t_run:
        runs += 1
        time.sleep(246)

    return runs


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
    powerss = cpu_gpu_power_seqs()
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
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
            ax.set_ylabel(r'$\frac{\mathrm{J}}{\mathrm{run}}$', fontsize=12)
            ax.set_title(['Predict', 'Update', 'Resample'][method])

    fig.suptitle('Energy per run')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('energy_per_run.pdf')
    plt.show()


if __name__ == '__main__':
    plot_energy_per_run()
