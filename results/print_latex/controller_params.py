import sim_base
import sympy
import numpy

_, _, K, _ = sim_base.get_parts()

sympy.print_latex(sympy.Matrix(numpy.diag(K.Q).T))

sympy.print_latex(sympy.Matrix(numpy.diag(K.R).T))
