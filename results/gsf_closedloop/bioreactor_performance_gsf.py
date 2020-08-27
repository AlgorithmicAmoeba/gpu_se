import numpy
import sim_base
import joblib
import matplotlib
import matplotlib.cm, matplotlib.colors
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath('../gsf_openloop'))
# noinspection PyUnresolvedReferences
import GSF_run_seq
# noinspection PyUnresolvedReferences
import GSF_power
import decorators


memory = joblib.Memory('cache/')


@memory.cache
def get_sim(N_particles, dt_control, dt_predict, end_time=50, pf=True):
    sim = sim_base.Simulation(N_particles, dt_control, dt_predict, end_time, pf)
    sim.simulate()
    ans = sim.performance, sim.mpc_frac, sim.predict_count, sim.update_count
    return ans


memory = joblib.Memory('cache/covariance')


@memory.cache
def get_covariance(N_particles, dt_control, dt_predict, end_time=50, pf=True):
    sim = sim_base.Simulation(N_particles, dt_control, dt_predict, end_time, pf)
    sim.simulate()
    ans = sim.covariance_point_size
    return ans


def performance_per_joule():
    run_seqss = GSF_run_seq.cpu_gpu_run_seqs()
    powerss = GSF_power.cpu_gpu_power_seqs()

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
            performance, mpc_frac, predict_count, update_count = get_sim(
                int(N_particles[i]),
                dt_controls[i],
                dt_predicts[i],
                pf=False
            )

            predict_power, update_power, resample_power = [method_power[j][i] for j in range(3)]

            total_power = predict_count * predict_power + update_count * (update_power + resample_power)
            ppj.append(performance / total_power)
            mpc_fracs.append(mpc_frac)

        ppjs.append(ppj)
        mpc_fracss.append(mpc_fracs)

    N_particles = [run_seqss[0][0][0], run_seqss[1][0][0]]
    ppjs = numpy.array(ppjs)
    mpc_fracss = numpy.array(mpc_fracss)

    return N_particles, ppjs, mpc_fracss


def get_performance():
    run_seqss = GSF_run_seq.cpu_gpu_run_seqs()

    performancess = []
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

        performances = []
        for i in range(len(N_particles)):
            performance, _, _, _ = get_sim(
                int(N_particles[i]),
                dt_controls[i],
                dt_predicts[i],
                pf=False
            )

            performances.append(performance)

        performancess.append(performances)

    N_particles = [run_seqss[0][0][0], run_seqss[1][0][0]]
    performancess = numpy.array(performancess)

    return N_particles, performancess


@decorators.Pickler.pickle_me
def covariance_point_estimates():
    run_seqss = GSF_run_seq.cpu_gpu_run_seqs()

    pcovss = []
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

        pcovs = []
        for i in range(len(N_particles)):
            pcov = get_covariance(
                int(N_particles[i]),
                dt_controls[i],
                dt_predicts[i],
                pf=False
            )

            pcovs.append(pcov)

        pcovss.append(pcovs)

    N_particles = [run_seqss[0][0][0], run_seqss[1][0][0]]
    pcovss = numpy.array(pcovss)

    return N_particles, pcovss


def plot_ppjs():
    N_particles, ppjs, _ = performance_per_joule()
    plt.semilogy(numpy.log2(N_particles[0]), ppjs[0], 'k.', label='CPU')
    plt.semilogy(numpy.log2(N_particles[1]), ppjs[1], 'kx', label='GPU')
    plt.xlabel('$ \log_2(N) $ particles')
    plt.ylabel(r'$\frac{\mathrm{ITAE}^{-1}}{\mathrm{J}}$')
    plt.title('Performance per energy')
    plt.legend()
    plt.savefig('GSF_ppj.pdf')
    plt.show()


def plot_mpc_fracs():
    N_particles, _, mpc_fracss = performance_per_joule()
    plt.plot(numpy.log2(N_particles[0]), mpc_fracss[0], 'k.', label='CPU')
    plt.plot(numpy.log2(N_particles[1]), mpc_fracss[1], 'kx', label='GPU')
    plt.xlabel('$ \log_2(N) $ particles')
    plt.ylabel(r'Fraction MPC convergence')
    plt.title('MPC convergence')
    plt.ylim(ymin=0)
    plt.legend()
    plt.savefig('GSF_mpc_frac.pdf')
    plt.show()


def plot_performances():
    N_particles, performancess = get_performance()
    plt.plot(numpy.log2(N_particles[0]), performancess[0], 'k.', label='CPU')
    plt.plot(numpy.log2(N_particles[1]), performancess[1], 'kx', label='GPU')
    plt.xlabel('$ \log_2(N) $ particles')
    plt.ylabel(r'$\mathrm{ITAE}^{-1}$')
    plt.title('Performance')
    plt.legend()
    plt.savefig('PF_performance.pdf')
    plt.show()


def plot_pcov():
    N_particles, pcovss = covariance_point_estimates()
    cmap = matplotlib.cm.get_cmap('plasma')
    ts = numpy.linspace(0, 50, 500)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey='row')
    for cpu_gpu in range(2):
        ax = axes[cpu_gpu]
        log2_Npart = numpy.log2(N_particles[cpu_gpu])
        norm = matplotlib.colors.Normalize(vmin=log2_Npart[0], vmax=log2_Npart[-1])
        for i in range(len(log2_Npart)):
            ax.plot(ts, pcovss[cpu_gpu, i], color=cmap(norm(log2_Npart[i])))
        ax.set_title(['CPU', 'GPU'][cpu_gpu])
        ax.set_xlabel('Time (min)')

        if cpu_gpu:
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gcf().gca())
        else:
            ax.set_ylabel(r'$\bar{\sigma}(\Sigma)$')

    plt.savefig('GSF_cov_con.pdf')
    plt.show()


# plot_ppjs()
# plot_mpc_fracs()
# plot_performances()
plot_pcov()