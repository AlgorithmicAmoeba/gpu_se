import sim_base
import sympy
import numpy

state_pdf, measurement_pdf = sim_base.get_noise()

sympy.print_latex(sympy.Matrix(numpy.diag(state_pdf.covariances_device[0])).T)

sympy.print_latex(sympy.Matrix(measurement_pdf.means_device[0]).T)

sympy.print_latex(sympy.Matrix(measurement_pdf.means_device[1]).T)

sympy.print_latex(sympy.Matrix(numpy.diag(measurement_pdf.covariances_device[0])).T)

sympy.print_latex(sympy.Matrix(numpy.diag(measurement_pdf.covariances_device[1])).T)

sympy.print_latex(sympy.Matrix(measurement_pdf.weights).T)
