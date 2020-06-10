import numpy
import sim_base

_, _, _, pp = sim_base.get_parts(
    N_particles=2**24,
    gpu=True
)


def test_ParallelParticleFilter_predict():
    pp.predict([1.], 1.)


def test_ParallelParticleFilter_update():
    z = numpy.array([2.3, 1.2])
    pp.update([1.], z)


def test_ParallelParticleFilter_resample():
    pp.resample()
