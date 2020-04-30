import numpy
from model.BioreactorModel import Bioreactor
from model import LinearModel

#                    Ng,        Nx,         Nfa, Ne, Na, Nb, Nh, V, T
X_op = numpy.array([0.28/180, 0.639773/24.6, 2/116, 0, 1e-5, 0, 4.857e-3, 1.077, 35])

model = Bioreactor(X_op, pH_calculations=True)

# Inputs
Cn_in = 0.625 * 10 / 60  # (g/L) / (g/mol) = mol/L
CgFg = 0.23

Cg_in = 314.19206 / 180  # (g/L) / (g/mol) = mol/L
Ca_in = 10  # mol/L
Cb_in = 10  # mol/L
Fm_in = 0

Fg_in = CgFg / 180 / Cg_in  # (g/h) / (g/mol) / (mol/L) = L/h
Fn_in = 0.625 / 1000 / Cn_in / 60  # (mg/h) / (mg/g) / (mol/L) / (g/mol) = L/h
Fa_in = 6e-9
Fb_in = 6e-9  # L/h
F_out = Fg_in + Fn_in + Fa_in + Fb_in + Fm_in

T_amb = 25
Q = 5 / 9

input_op = numpy.array([Fg_in, Cg_in, Fa_in, Ca_in, Fb_in, Cb_in, Fm_in, F_out, T_amb, Q])

dt = 0.1

lin_model = LinearModel.create_LinearModel(model, X_op, input_op, dt)

states = [2, 4, 5, 7]
inputs = [2, 4, 6]
outputs = [2, 9]

A = lin_model.A[states][:, states]
B = lin_model.B[states][:, inputs]
C = lin_model.C[outputs][:, states]
D = lin_model.D[outputs][:, inputs]
