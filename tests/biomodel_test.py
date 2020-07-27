import numpy
import tqdm
import model
import pytest


def test_biomodel():
    # Simulation set-up
    end_time = 500
    ts = numpy.linspace(0, end_time, end_time*10)
    dt = ts[1]

    bioreactor = model.Bioreactor(
        #                Ng,         Nx,      Nfa, Ne, Nh
        X0=numpy.array([3000 / 180, 1 / 24.6, 0 / 116, 0., 0.]),
        high_N=True
    )

    # Initial values
    us = [numpy.array([0., 0.])]
    xs = [bioreactor.X.copy()]
    ys = [bioreactor.outputs(us[-1])]

    not_cleared = True
    for t in tqdm.tqdm(ts[1:]):
        if t < 25:
            us.append(numpy.array([0., 0.]))
        elif t < 200:
            if not_cleared:
                bioreactor.X[[0, 2, 3, 4]] = 0
                not_cleared = False
                bioreactor.high_N = False

            us.append(numpy.array([0.06, 0.2]))
        elif t < 500:
            us.append(numpy.array([0.04, 0.1]))
        else:
            us.append(us[-1])

        bioreactor.step(dt, us[-1])
        outputs = bioreactor.outputs(us[-1])
        ys.append(outputs.copy())
        xs.append(bioreactor.X.copy())

    ys = numpy.array(ys)

    assert numpy.all(
        yi == pytest.approx(vi) for yi, vi in zip(
            ys[-1], [280,  632, 1121, 0, 50.5]
        )
    )
