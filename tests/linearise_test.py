import numpy
import scipy.signal
import model.LinearModel

X0 = numpy.array([1., 320.])
cstr = model.CSTRModel(X0)

X_op = X0
input_op = numpy.array([0.])

dt = 0.1

lin_model = model.LinearModel.create_LinearModel(cstr, X_op, input_op, dt)

# Calculated linear values from model

Ca, T = X_op
Q, = input_op

V, Ca0, dH, E, rho, R, Ta0, k0, Cp, F = 5, 1, -4.78e4, 8.314e4, 1e3, 8.314, 310, 72e7, 0.239, 0.1

A = numpy.array([[-F/V - k0*numpy.exp(-E/R/T), -k0*numpy.exp(-E/R/T)*Ca*E/R/T**2],
                 [-dH/rho/Cp*k0*numpy.exp(-E/R/T), -F/V - k0*numpy.exp(-E/R/T)*Ca*dH/rho/Cp*E/R/T**2]])

B = numpy.array([[0], [1/rho/Cp/V]])

C = numpy.array([[1, 0]])

D = numpy.array([[0]])

Nx = A.shape[0]
# We now have a continuous system, let's use the scipy.signal to get the discrete one
Ad, Bd, Cd, Dd, _ = scipy.signal.cont2discrete((A, B, C, D), dt)

for numeric, calucluated in zip([lin_model.A, lin_model.B, lin_model.C, lin_model.D], [Ad, Bd, Cd, Dd]):
    assert numpy.max(numpy.abs(numeric - calucluated)) < 1e-8
