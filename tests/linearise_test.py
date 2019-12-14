import numpy
import linearise
import model

X0 = numpy.array([1., 320.])
cstr = model.CSTRModel(X0)

X_op = X0
inpup_op = numpy.array([0., 0.1])

lin_model = linearise.create_LinearModel(cstr, X_op, inpup_op)

# Calculated linear values from model

Ca, T = X_op
Q, F = inpup_op

V, Ca0, dH, E, rho, R, Ta0, k0, Cp = 5, 1, -4.78e4, 8.314e4, 1e3, 8.314, 310, 72e7, 0.239

A = numpy.array([[-F/V - k0*numpy.exp(-E/R/T), -k0*numpy.exp(-E/R/T)*Ca*E/R/T**2],
    [-dH/rho/Cp*k0*numpy.exp(-E/R/T), -F/V - k0*numpy.exp(-E/R/T)*Ca*dH/rho/Cp*E/R/T**2]])

B = numpy.array([[0, (Ca0 - Ca)/V], [1/rho/Cp/V, (Ta0 - T)/V]])

C = numpy.array([[1, 0], [0, 1]])

D = numpy.array([[0], [0]])

for numeric, calucluated in zip([lin_model.A, lin_model.B, lin_model.C, lin_model.D], [A, B, C, D]):
    assert numpy.max(numpy.abs(numeric - calucluated)) < 1e-8
