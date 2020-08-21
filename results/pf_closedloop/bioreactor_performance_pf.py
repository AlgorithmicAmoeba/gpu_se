import numpy
import sim_base
import joblib
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath('../pf_openloop'))
# noinspection PyUnresolvedReferences
import PF_run_seq
import PF_power


memory = joblib.Memory('cache/')


@memory.cache
def get_sim(N_particles, dt_control, dt_predict, end_time=50, pf=True):
    sim = sim_base.Simulation(N_particles, dt_control, dt_predict, end_time, pf)
    sim.simulate()
    return sim


def performance_per_joule():
    run_seqss = PF_run_seq.cpu_gpu_run_seqs()
    powerss = PF_power.cpu_gpu_power_seqs()

    ppjs, mpc_fracss = [], []
    for cpu_gpu in range(2):
        dt_controls = numpy.min(run_seqss[cpu_gpu][0][1], axis=1)
        dt_predicts = dt_controls.copy()
        N_particles = run_seqss[cpu_gpu][0][0]
        for method in range(1, 3):
            _, run_seqs = run_seqss[cpu_gpu][method]
            times = numpy.min(run_seqs, axis=1)
            dt_controls += times

        dt_controls = numpy.maximum(dt_controls, 1)
        dt_predicts = numpy.maximum(dt_predicts, 0.1)

        method_power = []
        for method in range(3):
            _, powers = powerss[cpu_gpu][method]
            power = powers[:, 0]
            if cpu_gpu:
                power += powers[:, 1]

            method_power.append(power)

        ppj, mpc_fracs = [], []
        for i in range(len(N_particles)):
            sim = get_sim(
                int(N_particles[i]),
                dt_controls[i],
                dt_predicts[i],
                pf=True
            )

            predict_power, update_power, resample_power = [method_power[j][i] for j in range(3)]

            total_power = sim.predict_count * predict_power + sim.update_count * (update_power + resample_power)
            ppj.append(sim.performance/total_power)
            mpc_fracs.append(sim.mpc_frac)

        ppjs.append(ppj)
        mpc_fracss.append(mpc_fracs)

    N_particles = [run_seqss[0][0][0], run_seqss[1][0][0]]
    ppjs = numpy.array(ppjs)
    mpc_fracss = numpy.array(mpc_fracss)

    return N_particles, ppjs, mpc_fracss


def plot_ppjs():
    N_particles, ppjs, _ = performance_per_joule()
    plt.semilogy(numpy.log2(N_particles[0]), ppjs[0], 'k.', label='CPU')
    plt.semilogy(numpy.log2(N_particles[1]), ppjs[1], 'kx', label='GPU')
    plt.xlabel('$ \log_2(N) $ particles')
    plt.ylabel(r'$\frac{\mathrm{ITAE}^{-1}}{\mathrm{J}}$')
    plt.title('Performance per energy')
    plt.legend()
    plt.savefig('PF_ppj.pdf')
    plt.show()


def plot_mpc_fracs():
    N_particles, _, mpc_fracss = performance_per_joule()
    plt.plot(numpy.log2(N_particles[0]), mpc_fracss[0], 'k.', label='CPU')
    plt.plot(numpy.log2(N_particles[1]), mpc_fracss[1], 'kx', label='GPU')
    plt.xlabel('$ \log_2(N) $ particles')
    plt.ylabel(r'$\frac{\mathrm{ITAE}^{-1}}{\mathrm{J}}$')
    plt.title('Performance per energy')
    plt.legend()
    plt.savefig('PF_mpc_frac.pdf')
    plt.show()


plot_ppjs()
plot_mpc_fracs()
