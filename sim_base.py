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
    x0.means = bioreactor.X[numpy.newaxis, :]
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
        means=numpy.zeros(shape=(1, 5)),
        covariances=numpy.diag([1e-4, 1e-7, 1e-3, 1e-3, 1e-7])[numpy.newaxis, :, :],
        weights=numpy.array([1.]),
        library=lib
    )
    measurement_pdf = distribution(
        means=numpy.array([[1e-1, 0],
                           [0, -1e-1]]),
        covariances=numpy.array([[[6e-2, 0],
                                  [0, 8e-2]],

                                 [[5e-2, 1e-2],
                                  [1e-2, 7e-2]]]),
        weights=numpy.array([0.85, 0.15]),
        library=lib
    )
    return state_pdf, measurement_pdf


def performance(ys, r, ts):
    """Returns the IAE performance of a run.

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
    iae : float
        Integral of the Absolute Error
    """
    ae = numpy.abs((ys - r)/r)
    iae = sum([scipy.integrate.simps(ae_ax * ts, ts) for ae_ax in numpy.rollaxis(ae, 1)])
    return iae


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
