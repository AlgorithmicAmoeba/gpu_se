import numpy
import controller
import model.LinearModel
import gpu_funcs.MultivariateGaussianSum
import filter.particle


def get_parts(dt_control=1, N_particles=2*15):
    # Bioreactor
    bioreactor = model.Bioreactor(
        X0=model.Bioreactor.find_SS(
            numpy.array([0.06, 5/180, 0.2]),
            #            Ng,         Nx,      Nfa, Ne, Nh
            numpy.array([0.26/180, 0.64/24.6, 1/116, 0, 0])
        ),
        high_N=False
    )

    # Linear model
    lin_model = model.LinearModel.create_LinearModel(
        bioreactor,
        x_bar=model.Bioreactor.find_SS(
            numpy.array([0.04, 5/180, 0.1]),
            #           Ng,         Nx,      Nfa, Ne, Nh
            numpy.array([0.26/180, 0.64/24.6, 1/116, 0, 0])
        ),
        #          Fg_in (L/h), Cg (mol/L), Fm_in (L/h)
        u_bar=numpy.array([0.04, 5/180, 0.1]),
        T=dt_control
    )
    #  Select states, outputs and inputs for MPC
    lin_model.select_subset(
        states=[0, 2],  # Cg, Cfa
        inputs=[0, 2],  # Fg_in, Fm_in
        outputs=[0, 2],  # Cg, Cfa
    )

    # Controller
    K = controller.MPC(
        P=200,
        M=160,
        Q=numpy.diag([1e2, 1e3]),
        R=numpy.diag([1, 1]),
        lin_model=lin_model,
        ysp=lin_model.yn2d(numpy.array([0.28, 0.85]), subselect=False),
        u_bounds=[
            numpy.array([0, numpy.inf]) - lin_model.u_bar[0],
            numpy.array([0, numpy.inf]) - lin_model.u_bar[1]
        ]
    )

    # PF
    pf = filter.ParallelParticleFilter(
        f=bioreactor.homeostatic_DEs,
        g=bioreactor.static_outputs,
        N_particles=N_particles,
        x0=gpu_funcs.MultivariateGaussianSum(
            means=bioreactor.X[numpy.newaxis, :],
            covariances=numpy.diag([1e-10, 1e-8, 1e-9, 1e-9, 1e-9])[numpy.newaxis, :, :],
            weights=numpy.array([1.])
        ),
        state_pdf=gpu_funcs.MultivariateGaussianSum(
            means=numpy.zeros(shape=(1, 5)),
            covariances=numpy.diag([1e-10, 1e-8, 1e-9, 1e-9, 1e-9])[numpy.newaxis, :, :],
            weights=numpy.array([1.])
        ),
        measurement_pdf=gpu_funcs.MultivariateGaussianSum(
            means=numpy.array([[1e-4, 0],
                               [0, -1e-4]]),
            covariances=numpy.array([[[6e-5, 0],
                                      [0, 8e-5]],

                                     [[5e-5, 1e-5],
                                      [1e-5, 7e-5]]]),
            weights=numpy.array([0.85, 0.15])
        )
    )

    return bioreactor, lin_model, K, pf


def get_noise():
    state_pdf = gpu_funcs.MultivariateGaussianSum(
        means=numpy.zeros(shape=(1, 5)),
        covariances=numpy.diag([1e-10, 1e-8, 1e-9, 1e-9, 1e-9])[numpy.newaxis, :, :],
        weights=numpy.array([1.])
    )
    measurement_pdf = gpu_funcs.MultivariateGaussianSum(
        means=numpy.array([[1e-4, 0],
                           [0, -1e-4]]),
        covariances=numpy.array([[[6e-5, 0],
                                  [0, 8e-5]],

                                 [[5e-5, 1e-5],
                                  [1e-5, 7e-5]]]),
        weights=numpy.array([0.85, 0.15])
    )
    return state_pdf, measurement_pdf
