import numpy
import sim_base
import model
import sys
import os
sys.path.append(os.path.abspath('../pf_openloop'))
import PF_power
# noinspection PyUnresolvedReferences
import PF_run_seqs


class PowerLookup:
    """Looks up the amount of power a method uses.
    Prevents cpu_gpu_power_seqs from being called multiple times"""
    def __init__(self):
        self.powerss = PF_power.cpu_gpu_power_seqs()

    def __call__(self, cpu_gpu, method, N_part=None):
        N_parts, powers = self.powerss[cpu_gpu][method]
        total_power = powers[:, 0]
        if cpu_gpu:
            total_power += powers[:, 1]

        if N_part is None:
            return N_parts, total_power

        power = total_power[numpy.where(N_parts == N_part)[0][0]]

        return power


class TimeLookup:
    """Looks up the amount of time a method uses.
    Prevents cpu_gpu_run_seqs from being called multiple times"""
    def __init__(self):
        self.run_seqss = PF_run_seqs.cpu_gpu_run_seqs()

    def __call__(self, cpu_gpu, method, N_part=None):
        N_parts, run_seqs = self.run_seqss[cpu_gpu][method]
        times = numpy.min(run_seqs, axis=1)

        if N_part is None:
            return N_parts, times

        time = times[numpy.where(N_parts == N_part)[0][0]]

        return time


def get_simulation_performance(N_particles, dt_control, dt_predict):
    # Simulation set-up
    end_time = 50
    ts = numpy.linspace(0, end_time, end_time*10)
    dt = ts[1]
    assert dt <= dt_control and dt <= dt_predict

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
    ys_pf = [
        model.Bioreactor.static_outputs(
                (pf.weights @ pf.particles).get(),
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

            x_pf = (pf.weights @ pf.particles).get()
            u = K.step(lin_model.xn2d(x_pf), lin_model.un2d(us[-1]), lin_model.yn2d(ys_meas[-1]))
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

