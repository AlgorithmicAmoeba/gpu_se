import numpy
from decorators import PickleJar


@PickleJar.pickle(path='tests')
def f(x):
    return x**2


@PickleJar.pickle(path='tests')
def g(x):
    return x + 1


xs = numpy.arange(10)

f.clear_single(xs)

ys = g(xs)
print(ys)
print(f(xs))
