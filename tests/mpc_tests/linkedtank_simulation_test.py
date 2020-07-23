import numpy
import tqdm
import matplotlib.pyplot as plt
import controller
import model.LinearModel
import pytest
import tests.mpc_tests.LinkedTanks as LinkedTanks

# Simulation set-up
end_time = 80
ts = numpy.linspace(0, end_time, end_time*100)
dt = ts[1]
dt_control = 1
assert dt <= dt_control

# Plant
X0 = numpy.array([50., 60.])
diag_tank = LinkedTanks.LinkedTanks(X0)

# Linearise plant for MPC model
lin_model = model.LinearModel.create_LinearModel(
    diag_tank,
    x_bar=numpy.array([50., 60.]),
    u_bar=numpy.array([10., 12.]),
    T=dt_control
)
X_op = lin_model.x_bar
U_op = lin_model.u_bar
Y_op = lin_model.y_bar

# MPC
r = numpy.array([100., 30])

K = controller.MPC(
    P=20,
    M=8,
    Q=numpy.diag([10, 10]),
    R=numpy.diag([0.1, 0.1])*0,
    lin_model=lin_model,
    ysp=lin_model.yn2d(r)
)

# Controller initial params
us = [lin_model.u_bar]
ys = [lin_model.y_bar]
xs = [diag_tank.X]

biass = []
ctrl_moves0 = []

t_next = 0
for t in tqdm.tqdm(ts[1:]):
    diag_tank.step(dt, us[-1])
    ys.append(diag_tank.outputs((us[-1])))
    xs.append(diag_tank.X)

    if t > t_next:
        if K.y_predicted is not None:
            biass.append(ys[-1] - Y_op - K.y_predicted)
        u = K.step(xs[-1] - X_op, us[-1] - U_op, ys[-1] - Y_op)
        us.append(u + U_op)
        if t_next == 0:
            res = K.prob.solve()
            ctrl_moves0 = res.x[(K.P + 1) * 1 + K.P * 1:]

        t_next += dt_control
    else:
        us.append(us[-1])

ys = numpy.array(ys)
us = numpy.array(us)

biass = numpy.array(biass)


def test_linked_tank_bias():
    a = biass[100:]
    assert numpy.array(a) - numpy.average(a) == pytest.approx(0)


def test_linked_tank_SS():
    a = ys[1000:] - r
    assert a == pytest.approx(0, abs=1e-3)


if __name__ == '__main__':
    plt.subplot(2, 2, 1)
    plt.plot(ts, ys)
    plt.title('h')

    plt.subplot(2, 2, 2)
    plt.plot(ts, us)
    plt.title('Fin')

    plt.subplot(2, 2, 3)
    plt.plot(biass)
    plt.title('bias')

    # plt.subplot(2, 2, 4)
    # plt.stem(
    #     numpy.arange(0, (K.M + 2) * dt_control, dt_control),
    #     numpy.cumsum(ctrl_moves0) + U_op,
    #     use_line_collection=True
    # )
    # plt.title('ctrl_moves0')
    plt.show()
