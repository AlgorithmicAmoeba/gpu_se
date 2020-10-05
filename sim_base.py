import numpy
import cupy
import controller
import model.LinearModel
import gpu_funcs.MultivariateGaussianSum
import filter.particle
import scipy.integrate


def get_parts(dt_control=1, N_particles=2*15, gpu=True, pf=True):
    """Returns the parts needed for a closedloop simulation.
    Allows customization of the control period, number of particles
    and whether the simulation should use the GPU implementation or
    CPU implementation.

    Parameters
    ----------
    dt_control : float, optional
        Control period

    N_particles : int, optional
        Number of particles for PF

    gpu : bool, optional
        Should the GPU implementation be used?

    pf : bool
        If `True` then the particle filter is used
        otherwise, the GSF is used

    Returns
    -------
    bioreactor : model.Bioreactor
        The nonlinear system model

    lin_model : model.LinearModel
        Linear system model

    K : controller.MPC
        MPC controller

    pf : {filter.ParticleFilter, filter.ParallelParticleFilter}
        Particle filter
    """
    # Bioreactor
    bioreactor = model.Bioreactor(
        X0=model.Bioreactor.find_SS(
            numpy.array([0.06, 0.2]),
            #            Ng,         Nx,      Nfa, Ne, Nh
            numpy.array([260/180, 640/24.6, 1000/116, 0, 0])
        ),
        high_N=False
    )

    # Linear model
    lin_model = model.LinearModel.create_LinearModel(
        bioreactor,
        x_bar=model.Bioreactor.find_SS(
            numpy.array([0.04, 0.1]),
            #           Ng,         Nx,      Nfa, Ne, Nh
            numpy.array([260/180, 640/24.6, 1000/116, 0, 0])
        ),
        #          Fg_in (L/h), Cg (mol/L), Fm_in (L/h)
        u_bar=numpy.array([0.04, 0.1]),
        T=dt_control
    )
    #  Select states, outputs and inputs for MPC
    lin_model.select_subset(
        states=[0, 2],  # Cg, Cfa
        inputs=[0, 1],  # Fg_in, Fm_in
        outputs=[0, 2],  # Cg, Cfa
    )

    # Controller
    K = controller.MPC(
        P=int(300//dt_control),
        M=max(int(200//dt_control), 1),
        Q=numpy.diag([0.1, 1]),
        R=numpy.diag([1, 1]),
        lin_model=lin_model,
        ysp=lin_model.yn2d(numpy.array([280, 850]), subselect=False),
        u_bounds=[
            numpy.array([0, numpy.inf]) - lin_model.u_bar[0],
            numpy.array([0, numpy.inf]) - lin_model.u_bar[1]
        ]
    )

    # Filter
    if gpu:
        if pf:
            my_filter = filter.ParallelParticleFilter
        else:
            my_filter = filter.ParallelGaussianSumUnscentedKalmanFilter
        my_library = cupy
    else:
        if pf:
            my_filter = filter.ParticleFilter
        else:
            my_filter = filter.GaussianSumUnscentedKalmanFilter
        my_library = numpy

    state_pdf, measurement_pdf = get_noise(my_library)
    x0, _ = get_noise(my_library)
    x0.means += my_library.array(bioreactor.X[numpy.newaxis, :])
    pf = my_filter(
        f=bioreactor.homeostatic_DEs,
        g=bioreactor.static_outputs,
        N_particles=N_particles,
        x0=x0,
        state_pdf=state_pdf,
        measurement_pdf=measurement_pdf
    )

    return bioreactor, lin_model, K, pf


def get_noise(lib=cupy, deterministic=False):
    """Returns measurement and state noise.
    Allows customization of whether the simulation should use the GPU
    implementation or CPU implementation, and whether a
    deterministic version of the noise should be returned.
    The deterministic version is useful for testing.

    Parameters
    ----------
    lib : {cupy, numpy}, optional
        The math library for computations

    deterministic : bool, optional
     Should a deterministic version be used?

    Returns
    -------
    state_pdf, measurement_pdf : {gpu_funcs.MultivariateGaussianSum, gpu_funcs.DeterministicGaussianSum}
        State and measurement noise objects
    """
    if deterministic:
        distribution = gpu_funcs.DeterministicGaussianSum
    else:
        distribution = gpu_funcs.MultivariateGaussianSum
    state_pdf = distribution(
        means=numpy.zeros(shape=(2, 5)),
        covariances=numpy.array([
            numpy.diag([1e-4, 1e-7, 1e-3, 1e-3, 1e-7]),
            numpy.diag([1e-3, 1e-6, 1e-2, 1e-2, 1e-6])
        ]),
        weights=numpy.array([0.75, 0.25]),
        library=lib
    )
    measurement_pdf = distribution(
        means=numpy.array([[1e-1, 0],
                           [0, -1e-1]]),
        covariances=numpy.array([[[6e-2, 0],
                                  [0, 8e-2]],

                                 [[500, 100],
                                  [100, 700]]]),
        weights=numpy.array([0.85, 0.15]),
        library=lib
    )
    return state_pdf, measurement_pdf


def performance(ys, r, ts):
    """Returns the ITAE performance of a run.

    Parameters
    ----------
    ys : numpy.array
        Values of the outputs

    r  : numpy.array
        Set points

    ts : numpy.array
        Times

    Returns
    -------
    itae : float
        Integral of the Time Absolute Error
    """
    se = (ys - r)**2
    ise = sum([scipy.integrate.simps(se_ax * ts, ts) for se_ax in numpy.rollaxis(se, 1)])
    return ise


def get_random_io():
    """Get random system input and output for simulations

    Returns
    -------
    u, y : numpy.array
        Random inputs and outputs, respectively
    """
    u = numpy.array([
        numpy.random.uniform(low=0, high=0.1),
        numpy.random.uniform(low=0, high=0.2)
    ])
    y = numpy.array([
        numpy.random.uniform(low=0.25, high=0.3),
        numpy.random.uniform(low=0.8, high=0.9)
    ])
    return u, y


class Simulation:
    """Holds details of a simulation"""
    def __init__(self, N_particles, dt_control, dt_predict, end_time=50, pf=True):
        self.ts = numpy.linspace(0, end_time, end_time*10)
        self.dt = self.ts[1]
        self.dt_control = dt_control
        self.dt_predict = dt_predict

        self.bioreactor, self.lin_model, self.K, self.f = get_parts(
            dt_control=dt_control,
            N_particles=N_particles,
            pf=pf
        )

        self.state_pdf, self.measurement_pdf = get_noise()

        self.us = [numpy.array([0.06, 0.2])]
        self.xs = [self.bioreactor.X.copy()]
        self.ys = [self.bioreactor.outputs(self.us[-1])]
        self.ys_meas = [self.bioreactor.outputs(self.us[-1])]
        self.xs_f = [self.f.point_estimate()]
        self.ys_f = [
            numpy.array(
                model.Bioreactor.static_outputs(
                    self.f.point_estimate(),
                    self.us[-1]
                )
            )
        ]
        self.covariance_point_size = [self.f.point_covariance()]

        self.biass = []
        self.performance = None
        self.mpc_frac = None
        self.predict_count, self.update_count = 0, 0

    def simulate(self):
        """Performs a simulation using the simulation parameters"""
        t_next_control, t_next_predict = 0, 0
        mpc_converged, mpc_no_converged = 0, 0
        for t in self.ts[1:]:
            if t > t_next_predict:
                self.f.predict(self.us[-1], self.dt)
                self.predict_count += 1
                t_next_predict += self.dt_predict

            if t > t_next_control:
                U_temp = self.us[-1].copy()
                if self.K.y_predicted is not None:
                    self.biass.append(self.lin_model.yn2d(self.ys_meas[-1]) - self.K.y_predicted)

                self.f.update(self.us[-1], self.ys_meas[-1][self.lin_model.outputs])
                self.f.resample()
                self.update_count += 1

                self.xs_f.append(self.f.point_estimate())
                # noinspection PyBroadException
                try:
                    u = self.K.step(
                        self.lin_model.xn2d(self.xs_f[-1]),
                        self.lin_model.un2d(self.us[-1]),
                        self.lin_model.yn2d(self.ys_meas[-1])
                    )
                    mpc_converged += 1
                except:
                    u = numpy.array([0.06, 0.2])
                    mpc_no_converged += 1
                U_temp[self.lin_model.inputs] = self.lin_model.ud2n(u)
                self.us.append(U_temp.copy())
                t_next_control += self.dt_control
            else:
                self.us.append(self.us[-1])

            self.bioreactor.step(self.dt, self.us[-1])
            self.bioreactor.X += self.state_pdf.draw().get().squeeze()
            outputs = self.bioreactor.outputs(self.us[-1])
            self.ys.append(outputs.copy())
            outputs[self.lin_model.outputs] += self.measurement_pdf.draw().get().squeeze()
            self.ys_meas.append(outputs)
            self.xs.append(self.bioreactor.X.copy())
            self.ys_f.append(
                numpy.array(
                    model.Bioreactor.static_outputs(
                        self.f.point_estimate(),
                        self.us[-1]
                    )
                )
            )
            self.covariance_point_size.append(self.f.point_covariance())

        self.us = numpy.array(self.us)
        self.xs = numpy.array(self.xs)
        self.ys = numpy.array(self.ys)
        self.ys_meas = numpy.array(self.ys_meas)
        self.xs_f = numpy.array(self.xs_f)
        self.ys_f = numpy.array(self.ys_f)
        self.covariance_point_size = numpy.array(self.covariance_point_size)
        self.performance = performance(
            self.ys[:, self.lin_model.outputs],
            self.ys_f,
            self.ts
        )
        self.mpc_frac = mpc_converged / (mpc_converged + mpc_no_converged)
