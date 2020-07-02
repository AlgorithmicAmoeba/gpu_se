import numpy
import cupy
import sim_base
import filter


def test_gsufk():
    state_pdf, measurement_pdf = sim_base.get_noise(lib=numpy)
    bioreactor, _, _, _ = sim_base.get_parts(gpu=False)
    x0, _ = sim_base.get_noise(lib=numpy)
    x0.means = bioreactor.X[numpy.newaxis, :]
    gf = filter.GaussianSumUnscentedKalmanFilter(
        f=bioreactor.homeostatic_DEs,
        g=bioreactor.static_outputs,
        N_gaussians=7,
        x0=x0,
        state_pdf=state_pdf,
        measurement_pdf=measurement_pdf
    )

    u, z = sim_base.get_random_io()
    gf.predict(u, 0.1)
    gf.update(u, z)
    gf.resample()


def test_pgfukf():
    state_pdf, measurement_pdf = sim_base.get_noise(lib=cupy)
    bioreactor, _, _, _ = sim_base.get_parts(gpu=True)
    x0, _ = sim_base.get_noise(lib=cupy)
    x0.means = bioreactor.X[numpy.newaxis, :]
    gf = filter.ParallelGaussianSumUnscentedKalmanFilter(
        f=bioreactor.homeostatic_DEs,
        g=bioreactor.static_outputs,
        N_gaussians=7,
        x0=x0,
        state_pdf=state_pdf,
        measurement_pdf=measurement_pdf
    )

    u, z = sim_base.get_random_io()
    gf.predict(u, 0.1)
    gf.update(u, z)
    gf.resample()
