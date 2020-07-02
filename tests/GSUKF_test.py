import numpy
import cupy
import sim_base
import filter


def test_gsufk():
    state_pdfi, measurement_pdfi = sim_base.get_noise(lib=numpy)
    bioreactor, _, _, _ = sim_base.get_parts(gpu=False)
    x0i, _ = sim_base.get_noise(lib=numpy)
    x0i.means = bioreactor.X[numpy.newaxis, :]
    gf = filter.GaussianSumUnscentedKalmanFilter(
        f=bioreactor.homeostatic_DEs,
        g=bioreactor.static_outputs,
        N_gaussians=7,
        x0=x0i,
        state_pdf=state_pdfi,
        measurement_pdf=measurement_pdfi
    )

    ui, zi = sim_base.get_random_io()
    gf.predict(ui, 0.1)
    gf.update(ui, zi)
    gf.resample()


def test_pgfukf():
    state_pdfi, measurement_pdfi = sim_base.get_noise(lib=cupy)
    bioreactor, _, _, _ = sim_base.get_parts(gpu=True)
    x0i, _ = sim_base.get_noise(lib=cupy)
    x0i.means = bioreactor.X[numpy.newaxis, :]
    gf = filter.ParallelGaussianSumUnscentedKalmanFilter(
        f=bioreactor.homeostatic_DEs,
        g=bioreactor.static_outputs,
        N_gaussians=7,
        x0=x0i,
        state_pdf=state_pdfi,
        measurement_pdf=measurement_pdfi
    )

    ui, zi = sim_base.get_random_io()
    gf.predict(ui, 0.1)
    gf.update(ui, zi)
    gf.resample()
