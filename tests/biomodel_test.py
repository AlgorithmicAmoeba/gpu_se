import numpy
import tqdm
import pandas
from model.BioreactorModel import Bioreactor
from model.inputter import Inputs

ts = numpy.linspace(0, 230, 1000)

inputs = Inputs()

# Biomass C H_1.8 O_0.5 N_0.2 => 24.6 g/mol
#     Ng, Nx, Nfa, Ne, Nco, No, Nn, Na, Nb, Nez, Nfaz, Nezfa, V, Vg, T
X0 = [3.1/180, 1e-3/24.6, 0, 0, 0, 0, 2/60, 1e-5, 0, 0, 0, 0, 1.077, 0.1, 25]

m = Bioreactor(X0, inputs, pH_calculations=True)
Xs = [X0]

for ti in tqdm.tqdm(ts[1:]):
    m.step(ts[1])

xls = pandas.ExcelWriter('results/biotest.xlsx', engine='xlsxwriter')

model_names = ['Ng', 'Nx', 'Nfa', 'Ne', 'Nco', 'No', 'Nn', 'Na', 'Nb', 'Nez', 'Nfaz', 'Nezfa', 'V', 'Vg', 'T', 'pH']
model_data = pandas.DataFrame(m.get_data(), index=ts, columns=model_names)
model_data.index.name = 'ts'
model_data.to_excel(xls, 'model')

xls.save()
