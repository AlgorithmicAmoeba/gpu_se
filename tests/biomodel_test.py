import numpy
import tqdm
import pandas
import matplotlib.pyplot as plt
from model.BioreactorModel import Bioreactor
from model.inputter import Inputs
from historian import Historian

ts = numpy.linspace(0, 230, 1000)

inputs = Inputs()

# Biomass C H_1.8 O_0.5 N_0.2 => 24.6 g/mol
#     Ng, Nx, Nfa, Ne, Nco, No, Nn, Na, Nb, Nez, Nfaz, Nezfa, V, Vg, T
X0 = [3.1/180, 1e-3/24.6, 0, 0, 0, 0, 2/60, 1e-5, 0, 0, 0, 0, 1.077, 0.1, 25]

model = Bioreactor(X0, inputs, pH_calculations=True)
model_reagents = ['Ng', 'Nx', 'Nfa', 'Ne', 'Nco', 'No', 'Nn', 'Na', 'Nb', 'Nez', 'Nfaz', 'Nezfa']
model_states = ['V', 'Vg', 'T', 'pH']
model_names = model_reagents + model_states
molar_mass = numpy.array([180, 24.6, 116, 46, 44, 32, 60, 36.5, 40, 1, 1, 1])

history = Historian()

for ti in tqdm.tqdm(ts[1:]):
    model.step(ts[1])
    history.log(ti, dict(zip(model_names, model.outputs())))

concentration_data = pandas.read_csv('../model/run_9_conc.csv')

plt.plot(concentration_data['Time']+30, concentration_data['Glucose'], '.')
# plt.plot(history.df[], history['Ng']*180/history['V'])
Cg = history.df()['Ng']*180/history.df()['V']
Cg.plot()
plt.show()
