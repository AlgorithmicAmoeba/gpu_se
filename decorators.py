import multiprocessing
import subprocess
import time
import joblib
import numpy
import psutil
import scipy.integrate
import pickle
import os


class RunSequences:
    """A class to manage run sequences for functions.
    Specifically designed to allow vectorization of the process

    Parameters
    ----------
    function : callable
        The function to be vectorized/managed

    path : string
        Location where joblib cache should be recalled and saved to
    """
    def __init__(self, function, path='cache/'):
        self._memory = joblib.Memory(path + function.__name__)
        self.function = self._memory.cache(function)

    def __call__(self, N_particles, N_runs, *args, **kwargs):
        run_seqs = numpy.array(
            [self.function(int(N_particle), N_runs, *args, **kwargs) for N_particle in N_particles]
        )

        return N_particles, run_seqs

    def clear(self, *args):
        """Clears the stored result of the function with the arguments given

        Parameters
        ----------
        args : tuple
            Arguments of the function
        """
        self.function.call_and_shelve(*args).clear()

    @staticmethod
    def vectorize(function, *agrs, **kwargs):
        """Decorator function that creates a callable RunSequences class

        Parameters
        ----------
        function : function to be vectorized/managed

        Returns
        -------
        rs : RunSequences
            The RunSequences object that handles vectorized calls
        """
        return RunSequences(function, *agrs, **kwargs)


class PowerMeasurement:
    """A class to measure power drawn for functions.
        Specifically designed to allow vectorization of the process
        of power measurement

        Parameters
        ----------
        function : callable
            The function to be vectorized/managed

        path : string, optional
            Location where joblib cache should be recalled and saved to

        CPU_max_power : float, optional
            The power the CPU draws at 100% use
        """
    def __init__(self, function, path='cache/', CPU_max_power=30):
        self._memory = joblib.Memory(path + function.__name__)
        self.function = function
        self.CPU_max_power = CPU_max_power
        self._particle_call = self._particle_call_gen()

    def __call__(self, N_particles, t_run, *args, **kwargs):
        powers = numpy.array(
            [
                self._particle_call(N_particle, t_run, *args, **kwargs)
                for N_particle in N_particles
            ]
        )
        powers[:, 0] *= self.CPU_max_power
        return N_particles, powers

    def _particle_call_gen(self):
        """Generates the function that spawns the power measurement process
        and runs the function for the required amount of time"""

        @self._memory.cache
        def particle_call(N_particle, t_run, *args, **kwargs):
            queue = multiprocessing.Queue()
            power_process = multiprocessing.Process(
                target=PowerMeasurement._power_seq,
                args=(queue,)
            )
            power_process.start()
            N_runs = self.function(N_particle, t_run, *args, **kwargs)
            queue.put('Done')

            while queue.qsize() < 2:
                time.sleep(0.3)

            queue.get()
            power_seq = queue.get()

            power = scipy.integrate.trapz(power_seq[1:, :], power_seq[0], axis=1) / N_runs

            queue.close()
            queue.join_thread()
            power_process.join()

            return power

        return particle_call

    def clear(self, *args):
        """Clears the stored result of the function with the arguments given

        Parameters
        ----------
        args : tuple
            Arguments of the function
        """
        self._particle_call.call_and_shelve(*args).clear()

    @staticmethod
    def vectorize(function, *agrs, **kwargs):
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


class Pickler:
    """
    Parameters
    ----------
    function : callable
            The function to be managed

    path : string, optional
        Location where pickle cache should be recalled and saved to
    """
    def __init__(self, function, path='pickled/'):
        self.path = path + function.__name__
        self.function = function
        try:
            os.makedirs(self.path)
        except FileExistsError:
            pass

    def __call__(self, *args, **kwargs):
        # noinspection PyBroadException
        try:
            result = self.function(*args, **kwargs)
            f = open(self.path + '/object.pickle', 'wb')
            pickle.dump(result, f)
            f.close()
            return result
        except:
            f = open(self.path + '/object.pickle', 'rb')
            result = pickle.load(f)
            return result

    @staticmethod
    def pickle_me(function, *args, **kwargs):
        return Pickler(function, *args, **kwargs)
