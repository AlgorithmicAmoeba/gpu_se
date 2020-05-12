import numpy
import tqdm
import matplotlib.pyplot as plt
import controller
import model.LinearModel

# Simulation set-up
end_time = 80
ts = numpy.linspace(0, end_time, end_time*10)
dt = ts[1]
dt_control = 1
assert dt <= dt_control

# Tank model
X0 = numpy.array([0.5])
tank = model.TankModel(X0)

X_op = numpy.array([1.])
U_op = numpy.array([0.1])
tank1 = model.TankModel(X_op)
Y_op = tank1.outputs(U_op)

lin_model = model.LinearModel.create_LinearModel(tank, X_op, U_op, dt_control)

# set point
r = numpy.array([0.6]) - Y_op

# Controller parameters
P = 20
M = 8
Q = numpy.diag([10])
R = numpy.diag([0.1])

# Bounds
u_bounds = [numpy.array([0, 0.2]) - U_op[0]]
#
u_step_bounds = [numpy.array([-50, 10])]

K = controller.LQR(P, M, Q, R, lin_model, r)

# Controller initial params
us = [numpy.asarray(U_op)]
ys = [tank.outputs(us[-1])]
xs = [tank.X]

t_next = 0
for t in tqdm.tqdm(ts[1:]):
    if t > t_next:
        du = K.step(xs[-1] - X_op, us[-1] - U_op, ys[-1] - Y_op)
        u = us[-1] + du
        us.append(u)
        t_next += dt_control
    else:
        us.append(us[-1])
    tank.step(dt, us[-1])
    ys.append(tank.outputs((us[-1])))
    xs.append(tank.X)

ys = numpy.array(ys)
us = numpy.array(us)

plt.subplot(1, 2, 1)
plt.plot(ts, ys)

plt.subplot(1, 2, 2)
plt.plot(ts, us[:, 0])
plt.show()
