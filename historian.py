import pandas


class Historian:
    """Records data from simulations"""
    def __init__(self):
        self._data = []
        self._ready = True
        self._df = pandas.DataFrame()

    def log(self, time, data):
        data['ts'] = time
        self._data.append(data)
        self._ready = False

    def df(self):
        if self._ready:
            return self._df

        new_data = pandas.DataFrame(self._data)
        new_data.set_index('ts', inplace=True)
        self._df = self._df.append(new_data)
        self._ready = True
        return self._df