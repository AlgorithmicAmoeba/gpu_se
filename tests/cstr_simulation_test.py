import numpy
import tqdm
import matplotlib.pyplot as plt
import controller
import model.LinearModel
import scipy.optimize
import scipy.signal

# Simulation set-up
end_time = 80
ts = numpy.linspace(0, end_time, end_time*100)
dt = ts[1]
dt_control = 0.1
assert dt <= dt_control

cstr_SS = model.CSTRModel(X0=numpy.array([0.1, 500]))


def find_SS(x_ss):
    temp = cstr_SS.X
    cstr_SS.X = x_ss
    ans = cstr_SS.DEs(numpy.array([0.]))
    cstr_SS.X = temp
    return ans


# CSTR model
cstr = model.CSTRModel(
    X0=numpy.asarray(
        scipy.optimize.fsolve(
            find_SS,
            numpy.array([0.48, 412.0])
        ) + numpy.array([0.02, 10])
    )
)

# Linear CSTR model
lin_model = model.LinearModel.create_LinearModel(
    cstr,
    x_bar=numpy.asarray(
        scipy.optimize.fsolve(
            find_SS,
            numpy.array([0.48, 412.0])
        )
    ),
    u_bar=numpy.array([0.]),
    T=dt_control)

# set point
r = numpy.array([0.])

# Controller parameters
K = controller.MPC(
    P=300,
    M=300,
    Q=numpy.diag([1e3]),
    R=numpy.diag([1e-4]),
    lin_model=lin_model,
    ysp=r)

# Controller initial params
us = [lin_model.u_bar]
xs = [cstr.X.copy()]
ys = [cstr.outputs(us[-1])]

biass = []
ctrl_moves0 = []

t_next = 0
for t in tqdm.tqdm(ts[1:]):
    cstr.step(dt, us[-1])
    ys.append(cstr.outputs((us[-1])))
    xs.append(cstr.X.copy())

    if t > t_next:
        if K.y_predicted is not None:
            biass.append(lin_model.yn2d(ys[-1]) - K.y_predicted)

        u = K.step(lin_model.xn2d(xs[-1]), lin_model.un2d(us[-1]), lin_model.yn2d(ys[-1]))
        us.append(lin_model.ud2n(u))

        if t_next == 0:
            res = K.prob.solve()
            ctrl_moves0 = res.x[(K.P + 1) * 2 + K.P * 1:]

        t_next += dt_control
    else:
        us.append(us[-1])

ys = numpy.array(ys)
us = numpy.array(us)
xs = numpy.array(xs)

biass = numpy.array(biass)

plt.subplot(3, 2, 1)
plt.plot(ts, ys)
plt.axhline(lin_model.y_bar + r, color='r')
plt.title('C')

plt.subplot(3, 2, 2)
plt.plot(ts, xs[:, 1])
plt.title('T')

plt.subplot(3, 2, 3)
plt.plot(ts, us[:, 0]/60)
plt.title('Q')

plt.subplot(3, 2, 5)
plt.plot(
    numpy.arange(0, end_time, dt_control),
    biass
)
plt.title('bias')

plt.subplot(3, 2, 6)
plt.stem(
    numpy.arange(0, (K.M + 2) * dt_control, dt_control),
    numpy.cumsum(ctrl_moves0)/60 + lin_model.u_bar,
    use_line_collection=True
)
plt.title('ctrl_moves0')
plt.show()
