import os
import sys
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy
import sim_base
from decorators import PickleJar

if __name__ == '__main__':
    sys.path.append(os.path.abspath('../gsf_openloop'))
    # noinspection PyUnresolvedReferences
    import gsf_run_seq
    # noinspection PyUnresolvedReferences
    import gsf_power


@PickleJar.pickle(path='gsf/raw')
def get_sim(N_particles, dt_control, dt_predict, monte_carlo=0, end_time=50, pf=False):
    """Returns simulations results for a given simulation configuration.

    Parameters
    ----------
    N_particles : int
        Number of particles

    dt_control, dt_predict : float
        Control and prediction periods

    monte_carlo : int, optional
        The monte carlo indexing number

    end_time : float, optional
        Simulation end time

    pf : bool, optional
        Should the filter be the particle filter or gaussian sum filter

    Returns
    -------
    performance : float
        The simulation's ISE performance

    mpc_frac : float
        Fraction of MPC convergence

    predict_count, update_count : int
        Number of times the predict and update/resample methods were called

    covariance_point_size : numpy.array
        Maximum singular value of the covariance point estimate for each time instance
    """
    _ = monte_carlo
    sim = sim_base.Simulation(N_particles, dt_control, dt_predict, end_time, pf)
    sim.simulate()
    ans = sim.performance, sim.mpc_frac, sim.predict_count, sim.update_count, sim.covariance_point_size
    return ans


@PickleJar.pickle(path='gsf/processed')
def get_results(end_time=50, monte_carlo_sims=1):
    """Aggregates simulation results and performance post simulation calculations

    Parameters
    ----------
    end_time : float
        Simulation end time

    monte_carlo_sims : int
        The number of monte carlo simulations required

    Returns
    -------
    N_particles, energy_cpugpu, runtime_cpugpu, mpc_frac_cpugpu, performance_cpugpu, pcov_cpugpu : numpy.array
        Number of particles, energy measurements, run times, MPC convergence fractions,
        performance measurements, and covariance results from simulations
    """
    run_seqss = gsf_run_seq.cpu_gpu_run_seqs()
    powerss = gsf_power.cpu_gpu_power_seqs()

    energy_cpugpu, runtime_cpugpu, mpc_frac_cpugpu, performance_cpugpu, pcov_cpugpu = [], [], [], [], []
    for cpu_gpu in range(2):
        sums = numpy.min(run_seqss[cpu_gpu][0][1], axis=1)
        N_particles = run_seqss[cpu_gpu][0][0]

        for method in range(1, 3):
            _, run_seqs = run_seqss[cpu_gpu][method]
            times = numpy.min(run_seqs, axis=1)
            sums += times

        runtime_cpugpu.append(sums)

        method_power = []
        for method in range(3):
            _, powers = powerss[cpu_gpu][method]
            power = powers[:, 0]
            if cpu_gpu:
                power += powers[:, 1]

            method_power.append(power)

        energyss, mpc_fracss, performancess, pcovss = [], [], [], []
        for i in range(len(N_particles)):
            dt_control = dt_predict = 0.1

            energys, mpc_fracs, performances, pcovs = [], [], [], []
            for monte_carlo in range(monte_carlo_sims):
                # noinspection PyTupleAssignmentBalance
                performance, mpc_frac, predict_count, update_count, pcov = get_sim(
                    int(N_particles[i]),
                    dt_control,
                    dt_predict,
                    monte_carlo,
                    end_time=end_time,
                    pf=False
                )

                predict_energy, update_energy, resample_energy = [method_power[j][i] for j in range(3)]
                total_energy = predict_count * predict_energy + update_count * (update_energy + resample_energy)

                energys.append(total_energy)
                mpc_fracs.append(mpc_frac)
                performances.append(performance)
                pcovs.append(pcov)

            energyss.append(energys)
            mpc_fracss.append(mpc_fracs)
            performancess.append(performances)
            pcovss.append(pcovs)

        energy_cpugpu.append(energyss)
        mpc_frac_cpugpu.append(mpc_fracss)
        performance_cpugpu.append(performancess)
        pcov_cpugpu.append(pcovss)

    N_particles = [run_seqss[0][0][0], run_seqss[1][0][0]]
    energy_cpugpu = numpy.array(energy_cpugpu)
    mpc_frac_cpugpu = numpy.array(mpc_frac_cpugpu)
    performance_cpugpu = numpy.array(performance_cpugpu)
    pcov_cpugpu = numpy.array(pcov_cpugpu)

    return N_particles, energy_cpugpu, runtime_cpugpu, mpc_frac_cpugpu, performance_cpugpu, pcov_cpugpu


def plot_performance_vs_utilisation():
    """Plots the ISE performance against utilisation fractions.
    Third dimension shows power consumption"""
    N_particles, energy_cpugpu, runtime_cpugpu, _, performance_cpugpu, _ = get_results()

    cmap = matplotlib.cm.get_cmap('plasma')
    matplotlib.rcParams.update({'font.size': 9})
    plt.figure(figsize=(6.25, 5))
    for cpu_gpu in range(2):
        energys = numpy.log10(numpy.average(energy_cpugpu[cpu_gpu], axis=1))
        performances = numpy.average(performance_cpugpu[cpu_gpu], axis=1)
        utilizations = runtime_cpugpu[cpu_gpu] / (0.1*60)

        norm = matplotlib.colors.Normalize(vmin=energys[0], vmax=energys[-1])
        for i in range(len(energys)):
            if i == 0:
                plt.loglog(
                    utilizations[i],
                    performances[i],
                    ['k.', 'k^'][cpu_gpu],
                    label=['CPU', 'GPU'][cpu_gpu],
                    color=cmap(norm(energys[i]))
                )
            else:
                plt.loglog(
                    utilizations[i],
                    performances[i],
                    ['k.', 'k^'][cpu_gpu],
                    color=cmap(norm(energys[i]))
                )

        if cpu_gpu:
            cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gcf().gca())
            cbar.ax.set_xlabel(r'$ \frac{\mathrm{energy}}{\mathrm{period}} $ ($ W $)')
            rounded_ticks = numpy.round(cbar.ax.get_yticks(), 1)
            cbar.ax.set_yticklabels('$10^{' + numpy.char.array(rounded_ticks, unicode=True) + '}$')

    plt.axvline(1, color='red')
    plt.xlabel(r'Utilization')
    plt.ylabel(r'ISE')
    # plt.title('Closedloop performance versus utilization')
    plt.legend()
    plt.tight_layout()
    plt.savefig('gsf_performance_vs_utilisation.pdf')
    plt.show()


def plot_performance_per_watt():
    """Plots the ISE performance against power consumption.
    Third dimension shows number of particles"""
    N_particles, energy_cpugpu, runtime_cpugpu, _, performance_cpugpu, _ = get_results()

    cmap = matplotlib.cm.get_cmap('plasma')
    matplotlib.rcParams.update({'font.size': 9})
    plt.figure(figsize=(6.25, 5))
    for cpu_gpu in range(2):
        log2_Npart = numpy.log2(N_particles[cpu_gpu])
        energys = numpy.average(energy_cpugpu[cpu_gpu], axis=1)
        performances = numpy.average(performance_cpugpu[cpu_gpu], axis=1)
        utilizations = runtime_cpugpu[cpu_gpu] / (0.1 * 60)

        norm = matplotlib.colors.Normalize(vmin=log2_Npart[0], vmax=log2_Npart[-1])
        for i in range(len(energys)):
            if i == 0:
                plt.loglog(
                    energys[i] / (0.1 * 60),
                    performances[i],
                    ['.', '^'][cpu_gpu] if utilizations[i] < 1 else 'x',
                    label=['CPU', 'GPU]'][cpu_gpu],
                    color=cmap(norm(log2_Npart[i]))
                )
            else:
                plt.loglog(
                    energys[i] / (0.1 * 60),
                    performances[i],
                    ['.', '^'][cpu_gpu] if utilizations[i] < 1 else 'x',
                    color=cmap(norm(log2_Npart[i]))
                )

        if cpu_gpu:
            cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gcf().gca())
            cbar.ax.set_xlabel(r'$ N_p $')
            cbar.ax.set_yticklabels('$ 2^{' + numpy.char.array(cbar.ax.get_yticks(), unicode=True) + '}$')

    plt.xlabel(r'$ \frac{\mathrm{energy}}{\mathrm{period}} $ ($ W $)')
    plt.ylabel(r'ISE')
    # plt.title('Closedloop performance versus power')
    plt.legend()
    plt.tight_layout()
    plt.savefig('gsf_performance_per_watt.pdf')
    plt.show()


def plot_pcov():
    """Plots the covariance convergence against time"""
    N_particles, _, _, _, _, pcov_cpugpu = get_results(
        end_time=1500,
        monte_carlo_sims=1
    )
    cmap = matplotlib.cm.get_cmap('plasma')
    matplotlib.rcParams.update({'font.size': 18})
    plt.figure(figsize=(6.25*2, 5))
    ts = numpy.linspace(0, 1500, 15000)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey='row')
    for cpu_gpu in range(2):
        ax = axes[cpu_gpu]
        log2_Npart = numpy.log2(N_particles[cpu_gpu])
        pcovss = pcov_cpugpu[cpu_gpu]
        pcovs = numpy.average(pcovss, axis=1)

        norm = matplotlib.colors.Normalize(vmin=log2_Npart[0], vmax=log2_Npart[-1])
        for i in range(len(log2_Npart)):
            if i == numpy.where(pcovs == numpy.nanmax(pcovs))[0][0]:
                continue
            ax.plot(ts, pcovs[i], color=cmap(norm(log2_Npart[i])))
        ax.set_title(['CPU', 'GPU'][cpu_gpu])
        ax.set_xlabel('Time (min)')
        ax.set_xlim(xmin=0, xmax=500)

        if cpu_gpu:
            cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gcf().gca())
            cbar.ax.set_xlabel(r'$ N_p $')
            cbar.ax.set_yticklabels('$ 2^{' + numpy.char.array(cbar.ax.get_yticks(), unicode=True) + '}$')
        else:
            ax.set_ylabel(r'$\bar{\sigma}(\Sigma)$')

    plt.savefig('GSF_cov_con.pdf')
    plt.show()


if __name__ == '__main__':
    plot_performance_vs_utilisation()
    plot_performance_per_watt()
    plot_pcov()
