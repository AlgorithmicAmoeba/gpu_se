import numpy
import sim_base
import model
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
def get_simulation_performance(N_particles, dt_control, dt_predict):
    # Simulation set-up
    end_time = 50
    ts = numpy.linspace(0, end_time, end_time*10)
    dt = ts[1]

    bioreactor, lin_model, K, pf = sim_base.get_parts(
        dt_control=dt_control,
        N_particles=N_particles,

    )
    state_pdf, measurement_pdf = sim_base.get_noise()

    # Initial values
    us = [numpy.array([0.06, 0.2])]
    xs = [bioreactor.X.copy()]
    ys = [bioreactor.outputs(us[-1])]
    ys_meas = [bioreactor.outputs(us[-1])]
    xs_pf = [pf.point_estimate()]
    ys_pf = [
        model.Bioreactor.static_outputs(
                pf.point_estimate(),
                us[-1]
            )
    ]

    biass = []

    t_next_control, t_next_predict = 0, 0
    predict_count, update_count = 0, 0
    for t in ts[1:]:
        if t > t_next_predict:
            pf.predict(us[-1], dt)
            predict_count += 1
            t_next_predict += dt_predict

        if t > t_next_control:
            U_temp = us[-1].copy()
            if K.y_predicted is not None:
                biass.append(lin_model.yn2d(ys_meas[-1]) - K.y_predicted)

            pf.update(us[-1], ys_meas[-1][lin_model.outputs])
            pf.resample()
            update_count += 1

            xs_pf.append(pf.point_estimate())
            # noinspection PyBroadException
            try:
                u = K.step(lin_model.xn2d(xs_pf[-1]), lin_model.un2d(us[-1]), lin_model.yn2d(ys_meas[-1]))
            except:
                u = numpy.array([0.06, 0.2])
            U_temp[lin_model.inputs] = lin_model.ud2n(u)
            us.append(U_temp.copy())
            t_next_control += dt_control
        else:
            us.append(us[-1])

        bioreactor.step(dt, us[-1])
        bioreactor.X += state_pdf.draw().get()
        outputs = bioreactor.outputs(us[-1])
        ys.append(outputs.copy())
        outputs[lin_model.outputs] += measurement_pdf.draw().get()
        ys_meas.append(outputs)
        xs.append(bioreactor.X.copy())

        ys_pf.append(
            numpy.array(
                model.Bioreactor.static_outputs(
                    pf.point_estimate(),
                    us[-1]
                )
            )
        )

    ys_pf = numpy.array(ys_pf)
    performance = sim_base.performance(ys_pf, lin_model.yd2n(K.ysp), ts)

    return performance, predict_count, update_count


def performance_per_joule():
    run_seqss = PF_run_seq.cpu_gpu_run_seqs()
    powerss = PF_power.cpu_gpu_power_seqs()

    ppjs = []
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

        ppj = []
        for i in range(len(N_particles)):
            performance, predict_count, update_count = get_simulation_performance(
                int(N_particles[i]),
                dt_controls[i],
                dt_predicts[i]
            )

            predict_power, update_power, resample_power = [method_power[j][i] for j in range(3)]

            total_power = predict_count * predict_power + update_count * (update_power + resample_power)
            ppj.append((1/performance)/total_power)

        ppjs.append(ppj)

    N_particles = [run_seqss[0][0][0], run_seqss[1][0][0]]
    ppjs = numpy.array(ppjs)

    return N_particles, ppjs


def plot_ppjs():
    N_particles, ppjs = performance_per_joule()
    plt.semilogy(numpy.log2(N_particles[0]), ppjs[0], 'k.', label='CPU')
    plt.semilogy(numpy.log2(N_particles[1]), ppjs[1], 'kx', label='GPU')
    plt.xlabel('$ \log_2(N) $ particles')
    plt.ylabel(r'$\frac{\mathrm{ITAE}^{-1}}{\mathrm{J}}$')
    plt.title('Performance per energy')
    plt.legend()
    plt.show()


plot_ppjs()
