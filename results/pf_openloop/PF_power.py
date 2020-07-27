import numpy
import matplotlib
import matplotlib.pyplot as plt
import sim_base
import joblib
import subprocess
import psutil


class PowerSequences:
    def __init__(self, function, path='cache/'):
        self.memory = joblib.Memory(path + function.__name__)
        self.function = self.memory.cache(function)

    def __call__(self, N_particles, N_runs, *args, **kwargs):
        power_seqs = numpy.array(
            [self.function(int(N_particle), N_runs, *args, **kwargs) for N_particle in N_particles]
        )

        return N_particles, power_seqs

    def clear(self, *args):
        self.function.call_and_shelve(*args).clear()

    @staticmethod
    def vectorize(function):
        return PowerSequences(function)

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


@PowerSequences.vectorize
def predict_power_seq(N_particle, N_runs, gpu):
    cpu_frac, gpu_power = [], []

    _, _, _, p = sim_base.get_parts(
        N_particles=N_particle,
        gpu=gpu
    )

    for _ in range(N_runs):
        u, _ = sim_base.get_random_io()
        p.predict(u, 1.)
        cpu_frac.append(PowerSequences.get_CPU_frac())
        gpu_power.append(PowerSequences.get_GPU_power())

    return numpy.array([cpu_frac, gpu_power])


@PowerSequences.vectorize
def update_power_seq(N_particle, N_runs, gpu):
    cpu_frac, gpu_power = [], []

    _, _, _, p = sim_base.get_parts(
        N_particles=N_particle,
        gpu=gpu
    )

    for j in range(N_runs):
        u, y = sim_base.get_random_io()
        p.update(u, y)
        cpu_frac.append(PowerSequences.get_CPU_frac())
        gpu_power.append(PowerSequences.get_GPU_power())

    return numpy.array([cpu_frac, gpu_power])


@PowerSequences.vectorize
def resample_power_seq(N_particle, N_runs, gpu):
    cpu_frac, gpu_power = [], []

    _, _, _, p = sim_base.get_parts(
        N_particles=N_particle,
        gpu=gpu
    )

    for j in range(N_runs):
        p.weights = numpy.random.random(size=p.N_particles)
        p.weights /= numpy.sum(p.weights)
        p.resample()
        cpu_frac.append(PowerSequences.get_CPU_frac())
        gpu_power.append(PowerSequences.get_GPU_power())

    return numpy.array([cpu_frac, gpu_power])


def cpu_gpu_power_seqs():
    """Returns the power sequences for all the runs

    Returns
    -------
    power_seqss : List
        [CPU; GPU] x [predict; update; resample] x [N_particles; power_seq]
    """
    N_particles_cpu = 2**numpy.arange(1, 20, 0.5)
    N_particles_gpu = 2**numpy.arange(1, 24, 0.5)
    power_seqss = [
        [
            # predict_power_seq(N_particles_cpu, 20, False),
            # update_power_seq(N_particles_cpu, 20, False),
            # resample_power_seq(N_particles_cpu, 20, False)
        ],
        [
            predict_power_seq(N_particles_gpu, 20, True),
            update_power_seq(N_particles_gpu, 20, True),
            resample_power_seq(N_particles_gpu, 20, True)
        ]
    ]
    return power_seqss


power_seqss = cpu_gpu_power_seqs()
fig, axes = plt.subplots(2, 2)
for cpu_gpu in range(1, 2):
    for method in range(3):
        N_parts, power_seqs = power_seqss[cpu_gpu][method]
        N_logs = numpy.log2(N_parts)
        power_seqs_max = numpy.max(power_seqs, axis=2)

        for frac_power in range(2):
            ax = axes[cpu_gpu][frac_power]

            ax.plot(
                N_logs,
                power_seqs_max[:, frac_power],
                label=['Predict', 'Update', 'Resample'][method]
            )
            ax.legend()

plt.show()
# predict_seqs = power_seqss[1][0]
# predict20_seq = predict_seqs[1][19]
# predict5_seq = predict_seqs[1][4]
# plt.plot(predict20_seq[0, :])
# plt.plot(predict5_seq[0, :])
# plt.show()
# plt.plot(predict20_seq[1, :])
# plt.plot(predict5_seq[1, :])
# plt.show()
