import os
import sys
import joblib
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy
import sim_base

sys.path.append(os.path.abspath('../pf_openloop'))
# noinspection PyUnresolvedReferences
import PF_run_seq
import PF_power
import decorators


memory = joblib.Memory('cache/')


@memory.cache
def get_sim(N_particles, dt_control, dt_predict, monte_carlo=0, end_time=50, pf=True):
    _ = monte_carlo
    sim = sim_base.Simulation(N_particles, dt_control, dt_predict, end_time, pf)
    sim.simulate()
    ans = sim.performance, sim.mpc_frac, sim.predict_count, sim.update_count, sim.covariance_point_size
    return ans


@decorators.Pickler.pickle_me
def get_results():
    monte_carlo_sims = 5
    run_seqss = PF_run_seq.cpu_gpu_run_seqs()
    powerss = PF_power.cpu_gpu_power_seqs()

    ppj_cpugpu, mpc_frac_cpugpu, performance_cpugpu, pcov_cpugpu = [], [], [], []
    for cpu_gpu in range(2):
        sums = numpy.min(run_seqss[cpu_gpu][0][1], axis=1)
        t_predicts = sums.copy()
        N_particles = run_seqss[0][0][0]
        for method in range(1, 3):
            _, run_seqs = run_seqss[cpu_gpu][method]
            times = numpy.min(run_seqs, axis=1)
            sums += times

        dt_controls = numpy.maximum(sums, 1)

        method_power = []
        for method in range(3):
            _, powers = powerss[cpu_gpu][method]
            power = powers[:, 0]
            if cpu_gpu:
                power += powers[:, 1]

            method_power.append(power)

        ppjss, mpc_fracss, performancess, pcovss = [], [], [], []
        for i in range(len(N_particles)):
            dt_control = dt_controls[i]
            if dt_control > 1:
                dt_predict = t_predicts[i]
            else:
                n = numpy.floor((1 - sums[i]) / t_predicts[i]) + 1
                dt_predict = max(dt_control / n, 0.1)

            ppjs, mpc_fracs, performances, pcovs = [], [], [], []
            for monte_carlo in range(monte_carlo_sims):
                performance, mpc_frac, predict_count, update_count, pcov = get_sim(
                    int(N_particles[i]),
                    dt_controls[i],
                    dt_predict,
                    monte_carlo,
                    pf=True
                )

                predict_power, update_power, resample_power = [method_power[j][i] for j in range(3)]
                total_power = predict_count * predict_power + update_count * (update_power + resample_power)

                ppjs.append(performance / total_power)
                mpc_fracs.append(mpc_frac)
                performances.append(performance)
                pcovs.append(pcov)

            ppjss.append(ppjs)
            mpc_fracss.append(mpc_fracs)
            performancess.append(performances)
            pcovss.append(pcovs)

        ppj_cpugpu.append(ppjss)
        mpc_frac_cpugpu.append(mpc_fracss)
        performance_cpugpu.append(performancess)
        pcov_cpugpu.append(pcovss)

    N_particles = [run_seqss[0][0][0], run_seqss[0][0][0]]
    ppj_cpugpu = numpy.array(ppj_cpugpu)
    mpc_frac_cpugpu = numpy.array(mpc_frac_cpugpu)
    performance_cpugpu = numpy.array(performance_cpugpu)
    pcov_cpugpu = numpy.array(pcov_cpugpu)

    return N_particles, ppj_cpugpu, mpc_frac_cpugpu, performance_cpugpu, pcov_cpugpu


def plot_ppjs():
    N_particles, ppj_cpugpu, _, _, _ = get_results()
    for cpu_gpu in range(2):
        ppjss = ppj_cpugpu[cpu_gpu]
        logN_part = numpy.log2(N_particles[cpu_gpu])
        ppjs = numpy.average(ppjss, axis=1)
        plt.semilogy(
            logN_part,
            ppjs,
            ['k.', 'kx'][cpu_gpu],
            label=['CPU', 'GPU]'][cpu_gpu]
        )
    plt.xlabel('$ \log_2(N) $ particles')
    plt.ylabel(r'$\frac{\mathrm{ITAE}^{-1}}{\mathrm{J}}$')
    plt.title('Performance per energy')
    plt.legend()
    plt.savefig('PF_ppj.pdf')
    plt.show()


def plot_mpc_fracs():
    N_particles, _, mpc_frac_cpugpu, _, _ = get_results()
    for cpu_gpu in range(2):
        mpc_fracss = mpc_frac_cpugpu[cpu_gpu]
        logN_part = numpy.log2(N_particles[cpu_gpu])
        mpc_fracs = numpy.average(mpc_fracss, axis=1)
        plt.plot(
            logN_part,
            mpc_fracs,
            ['k.', 'kx'][cpu_gpu],
            label=['CPU', 'GPU]'][cpu_gpu]
        )
    plt.xlabel('$ \log_2(N) $ particles')
    plt.ylabel(r'Fraction MPC convergence')
    plt.title('MPC convergence')
    plt.ylim(ymin=0)
    plt.legend()
    plt.savefig('PF_mpc_frac.pdf')
    plt.show()


def plot_performances():
    N_particles, _, _, performance_cpugpu, _ = get_results()
    for cpu_gpu in range(2):
        performancess = performance_cpugpu[cpu_gpu]
        logN_part = numpy.log2(N_particles[cpu_gpu])
        performances = numpy.average(performancess, axis=1)
        plt.semilogy(
            logN_part,
            performances,
            ['k.', 'kx'][cpu_gpu],
            label=['CPU', 'GPU]'][cpu_gpu]
        )
    plt.xlabel('$ \log_2(N) $ particles')
    plt.ylabel(r'$\mathrm{ITAE}^{-1}$')
    plt.title('Performance')
    plt.legend()
    plt.savefig('PF_performance.pdf')
    plt.show()


def plot_pcov():
    N_particles, _, _, _, pcov_cpugpu = get_results()
    cmap = matplotlib.cm.get_cmap('plasma')
    ts = numpy.linspace(0, 500, 5000)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey='row')
    for cpu_gpu in range(2):
        ax = axes[cpu_gpu]
        log2_Npart = numpy.log2(N_particles[cpu_gpu])
        pcovss = pcov_cpugpu[cpu_gpu]
        indx = numpy.argmin(pcovss[-1])
        pcovs = pcovss[:, indx]

        norm = matplotlib.colors.Normalize(vmin=log2_Npart[0], vmax=log2_Npart[-1])
        for i in range(len(log2_Npart)):
            ax.plot(ts, pcovs[i], color=cmap(norm(log2_Npart[i])))
        ax.set_title(['CPU', 'GPU'][cpu_gpu])
        ax.set_xlabel('Time (min)')

        if cpu_gpu:
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gcf().gca())
        else:
            ax.set_ylabel(r'$\bar{\sigma}(\Sigma)$')

    plt.savefig('PF_cov_con.pdf')
    plt.show()


plot_ppjs()
plot_mpc_fracs()
plot_performances()
plot_pcov()
