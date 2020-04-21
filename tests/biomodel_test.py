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
#     Ng, Nx, Nfa, Ne, Na, Nb, Nh, V, T
X0 = [3.1/180, 1e-3/24.6, 0, 0, 1e-5, 0, 0, 1.077, 25]

model = Bioreactor(X0, pH_calculations=True)
model_reagents = ['Ng', 'Nx', 'Nfa', 'Ne', 'Na', 'Nb', 'Nh']
model_states = ['V', 'T', 'pH']
model_names = model_reagents + model_states
molar_mass = numpy.array([180, 24.6, 116, 46, 36.5, 40, 1])

history = Historian()

for ti in tqdm.tqdm(ts[1:]):
    ins = inputs(ti)
    model.step(ts[1], ins)
    history.log(ti, dict(zip(model_names, model.outputs(ins))))

    if model.high_N and ti > 30:
        model.high_N = False
        model.X[[0, 2, 3]] = 0

concentration_data = pandas.read_csv('../model/run_9_conc.csv')
ts_data = concentration_data['Time']+30
Cs = (history.df()[model_reagents]*molar_mass).div(history.df()['V'], axis=0)

for i, C in enumerate(model_reagents):
    plt.subplot(2, 5, i+1)
    Cs[C].plot()
    plt.title(C)

for i, C in enumerate(model_states):
    plt.subplot(2, 5, i+8)
    history.df()[C].plot()
    plt.title(C)

# for i, C_data in zip([0, 2, 3], ['Glucose', 'Fumaric', 'Ethanol']):
#     plt.subplot(3, 3, i+1)
#     plt.plot(ts_data, concentration_data[C_data], '.')

print(Cs[['Ng', 'Nfa', 'Ne', 'Nx']].tail(1))
print(Cs[['Nfa', 'Ne', 'Nx']].tail(1)/3.1)
plt.show()

# concentration_data = pandas.read_csv('../model/run_9_conc.csv')
# ts_data = concentration_data['Time']
#
# for i, C_data in zip([0, 1, 2], ['Glucose', 'Fumaric', 'Ethanol']):
#     plt.subplot(1, 3, i+1)
#     plt.plot(ts_data, concentration_data[C_data], '.')
#
# concentration_data = pandas.read_csv('../model/run_7_conc.csv')
# ts_data = concentration_data['Time']
#
# for i, C_data in zip([0, 1, 2], ['Glucose', 'Fumaric', 'Ethanol']):
#     plt.subplot(1, 3, i+1)
#     plt.plot(ts_data, concentration_data[C_data], '.')
#     plt.axvline(66)
#     plt.axvline(101)
#     plt.axvline(137)
#
# plt.show()
