import numpy
import model
import pytest


def test_linearise():
    bioreactor = model.Bioreactor(
            X0=model.Bioreactor.find_SS(
                numpy.array([0.06, 0.2]),
                #            Ng,         Nx,      Nfa, Ne, Nh
                numpy.array([260/180, 640/24.6, 1000/116, 0, 0])
            ),
            high_N=False
        )

    lin_model = model.LinearModel.create_LinearModel(
            bioreactor,
            x_bar=model.Bioreactor.find_SS(
                numpy.array([0.04, 0.1]),
                #           Ng,         Nx,      Nfa, Ne, Nh
                numpy.array([260/180, 640/24.6, 1000/116, 0, 0])
            ),
            #          Fg_in (L/h), Cg (mol/L), Fm_in (L/h)
            u_bar=numpy.array([0.04, 0.1]),
            T=1
        )

    assert lin_model.A[0, 0] == pytest.approx(0.72648)
