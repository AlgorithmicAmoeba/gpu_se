import numpy
import tqdm
import matplotlib.pyplot as plt
import controller
import model.LinearModel
import pytest

# Simulation set-up
end_time = 80
ts = numpy.linspace(0, end_time, end_time*100)
dt = ts[1]
dt_control = 1
assert dt <= dt_control

# Plant
X0 = numpy.array([50.])
tank_linear = model.TankModel(X0, linear=True)
tank_nonlinear = model.TankModel(X0, linear=False)

# Linearise plant for MPC model
X_op = numpy.array([50.])
U_op = numpy.array([10.])
tank1 = model.TankModel(X_op)
Y_op = tank1.outputs(U_op)

lin_model_linear = model.LinearModel.create_LinearModel(
    tank_linear,
    x_bar=numpy.array([50.]),
    u_bar=numpy.array([10.]),
    T=dt_control
)

lin_model_nonlinear = model.LinearModel.create_LinearModel(
    tank_nonlinear,
    x_bar=numpy.array([50.]),
    u_bar=numpy.array([10.]),
    T=dt_control)

# MPC
r = numpy.array([100.])

K_linear = controller.MPC(
    P=20,
    M=8,
    Q=numpy.diag([10]),
    R=numpy.diag([0.1])*0,
    lin_model=lin_model_linear,
    ysp=lin_model_linear.yn2d(r)
)

K_nonlinear = controller.MPC(
    P=20,
    M=8,
    Q=numpy.diag([10]),
    R=numpy.diag([0.1])*0,
    lin_model=lin_model_nonlinear,
    ysp=lin_model_nonlinear.yn2d(r)
)

# Controller initial params
us_linear = [lin_model_linear.u_bar]
ys_linear = [lin_model_linear.y_bar]
xs_linear = [tank_linear.X]

biass_linear = []
ctrl_moves0_linear = []

us_nonlinear = [lin_model_nonlinear.u_bar]
ys_nonlinear = [lin_model_nonlinear.y_bar]
xs_nonlinear = [tank_nonlinear.X]

biass_nonlinear = []
ctrl_moves0_nonlinear = []

t_next = 0
for t in tqdm.tqdm(ts[1:]):
    tank_linear.step(dt, us_linear[-1])
    ys_linear.append(tank_linear.outputs((us_linear[-1])))
    xs_linear.append(tank_linear.X)

    tank_nonlinear.step(dt, us_nonlinear[-1])
    ys_nonlinear.append(tank_nonlinear.outputs((us_nonlinear[-1])))
    xs_nonlinear.append(tank_nonlinear.X)
    if t > t_next:
        if K_linear.y_predicted is not None:
            biass_linear.append(ys_linear[-1] - Y_op - K_linear.y_predicted)
        u = K_linear.step(xs_linear[-1] - X_op, us_linear[-1] - U_op, ys_linear[-1] - Y_op)
        us_linear.append(u + U_op)
        if t_next == 0:
            res = K_linear.prob.solve()
            ctrl_moves0_linear = res.x[(K_linear.P + 1) * 1 + K_linear.P * 1:]

        if K_nonlinear.y_predicted is not None:
            biass_nonlinear.append(ys_nonlinear[-1] - Y_op - K_nonlinear.y_predicted)
        u = K_nonlinear.step(xs_nonlinear[-1] - X_op, us_nonlinear[-1] - U_op, ys_nonlinear[-1] - Y_op)
        us_nonlinear.append(u + U_op)
        if t_next == 0:
            res = K_nonlinear.prob.solve()
            ctrl_moves0_nonlinear = res.x[(K_nonlinear.P + 1) * 1 + K_nonlinear.P * 1:]
        t_next += dt_control
    else:
        us_linear.append(us_linear[-1])
        us_nonlinear.append(us_nonlinear[-1])

ys_linear = numpy.array(ys_linear)
us_linear = numpy.array(us_linear)

ys_nonlinear = numpy.array(ys_nonlinear)
us_nonlinear = numpy.array(us_nonlinear)

biass_linear = numpy.array(biass_linear)
biass_nonlinear = numpy.array(biass_nonlinear)


def test_linear_tank_bias():
    assert numpy.max(biass_linear) < 0.1


def test_nonlinear_tank_bias():
    a = biass_nonlinear[100:]
    assert numpy.array(a) - numpy.average(a) == pytest.approx(0)


def test_linear_tank_SS():
    a = ys_linear[500:] - r
    assert a == pytest.approx(0, abs=1e-3)


def test_nonlinear_tank_SS():
    a = ys_nonlinear[500:] - r
    assert a == pytest.approx(0, abs=1e-3)


if __name__ == '__main__':
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.plot(ts, ys_linear)
    plt.title('h')

    plt.subplot(2, 2, 2)
    plt.plot(ts, us_linear[:, 0])
    plt.title('Fin')

    plt.subplot(2, 2, 3)
    plt.plot(biass_linear)
    plt.title('bias')

    plt.subplot(2, 2, 4)
    plt.stem(
        numpy.arange(0, (K_linear.M + 2) * dt_control, dt_control),
        numpy.cumsum(ctrl_moves0_linear) + U_op,
        use_line_collection=True
    )
    plt.title('ctrl_moves0')

    plt.figure(2)
    plt.subplot(2, 2, 1)
    plt.plot(ts, ys_nonlinear)
    plt.title('h')

    plt.subplot(2, 2, 2)
    plt.plot(ts, us_nonlinear[:, 0])
    plt.title('Fin')

    plt.subplot(2, 2, 3)
    plt.plot(biass_nonlinear)
    plt.title('bias')

    plt.subplot(2, 2, 4)
    plt.stem(
        numpy.arange(0, (K_nonlinear.M + 2) * dt_control, dt_control),
        numpy.cumsum(ctrl_moves0_nonlinear) + U_op,
        use_line_collection=True
    )
    plt.title('ctrl_moves0')
    plt.show()
