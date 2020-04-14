import numpy
import tqdm
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
