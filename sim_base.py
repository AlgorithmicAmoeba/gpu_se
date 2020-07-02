import numpy
import cupy
import controller
import model.LinearModel
import gpu_funcs.MultivariateGaussianSum
import filter.particle
import scipy.integrate


def get_parts(dt_control=1, N_particles=2*15, gpu=True):
    # Bioreactor
    bioreactor = model.Bioreactor(
        X0=model.Bioreactor.find_SS(
            numpy.array([0.06, 0.2]),
            #            Ng,         Nx,      Nfa, Ne, Nh
            numpy.array([0.26/180, 0.64/24.6, 1/116, 0, 0])
        ),
        high_N=False
    )

    # Linear model
    lin_model = model.LinearModel.create_LinearModel(
        bioreactor,
        x_bar=model.Bioreactor.find_SS(
            numpy.array([0.04, 0.1]),
            #           Ng,         Nx,      Nfa, Ne, Nh
            numpy.array([0.26/180, 0.64/24.6, 1/116, 0, 0])
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
        M=int(200//dt_control),
        Q=numpy.diag([1e2, 1e3]),
        R=numpy.diag([1, 1]),
        lin_model=lin_model,
        ysp=lin_model.yn2d(numpy.array([0.28, 1.15]), subselect=False),
        u_bounds=[
            numpy.array([0, numpy.inf]) - lin_model.u_bar[0],
            numpy.array([0, numpy.inf]) - lin_model.u_bar[1]
        ]
    )

    # PF
    if gpu:
        my_filter = filter.ParallelParticleFilter
        my_library = cupy
    else:
        my_filter = filter.ParticleFilter
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
    if deterministic:
        distribution = gpu_funcs.DeterministicGaussianSum
    else:
        distribution = gpu_funcs.MultivariateGaussianSum
    state_pdf = distribution(
        means=numpy.zeros(shape=(1, 5)),
        covariances=numpy.diag([1e-10, 1e-13, 1e-9, 1e-9, 1e-13])[numpy.newaxis, :, :],
        weights=numpy.array([1.]),
        library=lib
    )
    measurement_pdf = distribution(
        means=numpy.array([[1e-4, 0],
                           [0, -1e-4]]),
        covariances=numpy.array([[[6e-5, 0],
                                  [0, 8e-5]],

                                 [[5e-5, 1e-5],
                                  [1e-5, 7e-5]]]),
        weights=numpy.array([0.85, 0.15]),
        library=lib
    )
    return state_pdf, measurement_pdf


def performance(ys, r, ts):
    ae = numpy.abs((ys - r)/r)
    iae = sum([scipy.integrate.simps(ae_ax * ts, ts) for ae_ax in numpy.rollaxis(ae, 1)])
    return iae


def get_random_io():
    u = numpy.array([
        numpy.random.uniform(low=0, high=0.1),
        numpy.random.uniform(low=0, high=0.2)
    ])
    y = numpy.array([
        numpy.random.uniform(low=0.25, high=0.3),
        numpy.random.uniform(low=0.8, high=0.9)
    ])
    return u, y
