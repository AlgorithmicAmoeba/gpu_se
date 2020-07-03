import numpy
import cupy
import sim_base
import filter
import pytest


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


def test_same():
    """Test if the CPU and GPU implenations gie the same results"""
    state_pdf, measurement_pdf = sim_base.get_noise(deterministic=True)
    bioreactor, _, _, _ = sim_base.get_parts()
    x0, _ = sim_base.get_noise(deterministic=True)
    x0.means = bioreactor.X[numpy.newaxis, :]
    pgf = filter.ParallelGaussianSumUnscentedKalmanFilter(
        f=bioreactor.homeostatic_DEs,
        g=bioreactor.static_outputs,
        N_gaussians=7,
        x0=x0,
        state_pdf=state_pdf,
        measurement_pdf=measurement_pdf
    )

    state_pdf, measurement_pdf = sim_base.get_noise(lib=numpy, deterministic=True)
    bioreactor, _, _, _ = sim_base.get_parts(gpu=False)
    x0, _ = sim_base.get_noise(lib=numpy, deterministic=True)
    x0.means = bioreactor.X[numpy.newaxis, :]
    gf = filter.GaussianSumUnscentedKalmanFilter(
        f=bioreactor.homeostatic_DEs,
        g=bioreactor.static_outputs,
        N_gaussians=7,
        x0=x0,
        state_pdf=state_pdf,
        measurement_pdf=measurement_pdf
    )

    def same():
        assert numpy.average(pgf.means.get() - gf.means) == pytest.approx(0, abs=1e-7)
        assert numpy.average(pgf.covariances.get() - gf.covariances) == pytest.approx(0, abs=1e-10)
        normalised_pgf_weights = pgf.weights.get()
        normalised_pgf_weights /= sum(normalised_pgf_weights)

        normalised_gf_weights = gf.weights
        normalised_gf_weights /= sum(normalised_gf_weights)
        assert numpy.average(normalised_pgf_weights - normalised_gf_weights) == pytest.approx(0, abs=1e-7)

    u, z = sim_base.get_random_io()
    # same()
    #
    # pgf.predict(u, 0.1)
    # gf.predict(u, 0.1)
    # same()

    gf.update(u, z)
    pgf.update(u, z)
    same()

    pgf.resample()
    gf.resample()
    same()


test_same()
