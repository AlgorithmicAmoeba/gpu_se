import multiprocessing
import subprocess
import time
import joblib
import numpy
import psutil
import scipy.integrate
import pickle
import os
import cupy


global_cache_settings = {
    'force_rerun': False,
    'force_same_code': True
}


# noinspection PyProtectedMember
class PickleJar(joblib.memory.MemorizedFunc):
    """Implements joblib.Memory but that allows the pickling
    to work across multiple computers
    """
    def __init__(self, func, location='', cache_settings=None):
        if cache_settings is None:
            cache_settings = global_cache_settings
        self.cache_settings = cache_settings

        joblib.memory._build_func_identifier = lambda f: f.__name__

        dirname = os.path.dirname(__file__)
        location = os.path.join(dirname, 'picklejar', location)

        super().__init__(func, location)

        if self.cache_settings['force_same_code']:
            func_code, source_file, first_line = joblib.memory.get_func_code(self.func)
            self._write_func_code(func_code, first_line)

    @staticmethod
    def pickle(path):
        return lambda fun: PickleJar(fun, path)

    def clear_single(self, *args, **kwargs):
        """Clears the single stored result of the function with the arguments given

        Parameters
        ----------
        args : tuple
            Arguments of the function
        """
        self.call_and_shelve(*args, **kwargs).clear()

    def __call__(self, *args, **kwargs):
        if self.cache_settings['force_rerun']:
            self.clear_single(*args, **kwargs)
        return super().__call__(*args, **kwargs)


class RunSequences:
    """A class to manage run sequences for functions.
    Specifically designed to allow vectorization of the process

    Parameters
    ----------
    func : callable
        The function to be vectorized/managed
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, N_particles, *args, **kwargs):
        run_seqs = numpy.array(
            [self.func(int(N_particle), *args, **kwargs) for N_particle in N_particles]
        )

        return N_particles, run_seqs

    @staticmethod
    def vectorize(function):
        """Decorator function that creates a callable RunSequences class

        Parameters
        ----------
        function : function to be vectorized/managed

        Returns
        -------
        rs : RunSequences
            The RunSequences object that handles vectorized calls
        """
        return RunSequences(function)


class PowerMeasurement:
    """A class to measure power drawn for functions.
        Specifically designed to allow vectorization of the process
        of power measurement

        Parameters
        ----------
        function : callable
            The function to be vectorized/managed

        CPU_max_power : float, optional
            The power the CPU draws at 100% use
        """
    def __init__(self, function, CPU_max_power=30):
        self.function = function
        self.CPU_max_power = CPU_max_power
        self.__name__ = self.function.__name__
        # noinspection PyUnresolvedReferences
        self.__code__ = self.function.__code__

    def __call__(self, N_particle, t_run, *args, **kwargs):
        return self._particle_call(N_particle, t_run, *args, **kwargs)

    def _particle_call(self, N_particle, t_run, *args, **kwargs):
        """Spawns the power measurement process
        and runs the function for the required amount of time"""
        queue = multiprocessing.Queue()
        power_process = multiprocessing.Process(
            target=PowerMeasurement._power_seq,
            args=(queue,)
        )
        power_process.start()
        res = self.function(N_particle, t_run, *args, **kwargs)
        queue.put('Done')

        while queue.qsize() < 2:
            time.sleep(0.3)

        queue.get()
        power_seq = queue.get()

        power = scipy.integrate.trapz(power_seq[1:, :], power_seq[0], axis=1)
        # noinspection PyUnresolvedReferences
        power[0] *= self.CPU_max_power

        queue.close()
        queue.join_thread()
        power_process.join()

        return res, power

    @staticmethod
    def measure(function, *agrs, **kwargs):
        """Decorator function that creates a callable PowerMeasurement class

        Parameters
        ----------
        function : function to be vectorized/managed

        Returns
        -------
        pm : RunSequences
            The PowerMeasurement object that handles vectorized calls
        """
        return PowerMeasurement(function, *agrs, **kwargs)

    @staticmethod
    def get_GPU_power():
        """Uses the nvidia-smi interface to query the current power drawn by the GPU

        Returns
        -------
        power : float
            The current power drawn by the GPU
        """
        return float(
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"
                 ]
            )
        )

    @staticmethod
    def get_CPU_frac():
        """Uses the psutil library to query the current CPU usage
        fraction

        Returns
        -------
        frac : float
            The current current CPU usage fraction
        """
        return psutil.cpu_percent()/100

    @staticmethod
    def _power_seq(q):
        """A function meant to be run in parallel with another function.
        This function takes readings of the CPU usage percentage and GPU power usage.
        Parameters
        ----------
        q : multiprocessing.Queue
            A thread safe method of message passing between the host process and this one.
            Allows this process to return a numpy array of measurements
        """
        times, cpu_frac, gpu_power = [], [], []

        while q.empty():
            times.append(time.time())
            cpu_frac.append(PowerMeasurement.get_CPU_frac())
            gpu_power.append(PowerMeasurement.get_GPU_power())
            time.sleep(0.2)

        q.put(numpy.array([times, cpu_frac, gpu_power]))
