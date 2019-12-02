"""

This is A COPY OF the main file in which the configuration for the documentation is made.
To configure this for YOUR next project - search for the pattern #CHNAGEME# in this file and follow the instructions.

"""
# Build with `make clean latexpdf`

# -*- coding: utf-8 -*-
#
# Sphinx185 documentation build configuration file, created by
# sphinx-quickstart on Sat May 12 19:34:31 2018.

# DalyaG: This file was heavily modified from its original build.
# Hope you find this useful :)


# -- Define here your working directory -------------------------------------------------------------------------------

import os
import sys
# #CHNAGEME# Change this to be the correct local path
# Also include paths to required libraries
sys.path.append(os.path.abspath('/home/ex/Documents/Masters/gpu_se'))


# -- Some general info  about the project -----------------------------------------------------------------------------

# #CHNAGEME# Change this to fit your project.
project = u'GPU accelerated state estimators'
copyright = u'2020, RoosDC'
author = u'Darren Craig Roos'


# -- A few basic configurations ---------------------------------------------------------------------------------------

# The documentation in this project will be mostly generated from .rst files
# In This project, every auto-documented module/class has its own .rst file, under the main documentation dir,
#   which is rendered into an .html page.
# Get more information here: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
source_suffix = ['.rst']

# This is the name of the main page of the project.
# It means that you need to have an `index.rst` file where you will design the landing page of your project.
# It will be rendered into an .html page that you can find at: `_build/html/index.html`
# (this is a standard name. change it only if you know what you are doing)
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build']

# List here any paths that contain templates, relative to this directory.
# You can find some not-so-intuitive information here: http://www.sphinx-doc.org/en/master/templating.html
# But the best way to learn is by example, right? :)
# So, for example, in this project, I wanted to change the title of the Table Of Contents in the sidebar.
#   So I copied `<Sphinx install dir>/themes/basic/globaltoc.html` into the `_templates` folder,
#      and replaced 'Table of Content' with 'Universe'.
# #CHNAGEME# In documentation_template_for_your_next_project/_templates/globaltoc.html
#            Change "Universe" into the name of your project
# templates_path = ['_templates']


# -- Define and configure non-default extensions ----------------------------------------------------------------------

# You can find a list of available extension here: http://www.sphinx-doc.org/en/master/extensions.html
# Use the extension `sphinx.ext.napoleon` rather than `numpydocs` as the latter gives wierd errors with class methods
extensions = ['sphinx.ext.napoleon', 'sphinx.ext.todo',
              'sphinx.ext.viewcode', 'sphinx.ext.autodoc', 'sphinx.ext.imgmath']

# Above extensions explanation and configurations:
# Remove the underscore chatacter after the todo_ in the following lines
# todo_: When you use the syntax ".. todo_:: some todo_ comment" in your docstring,
#         it will appear in a highlighted box in the documentation.
#       In order for this extension to work, make sure you include the following:
todo_include_todos = True

# viewcode: Next to each function/module in the documentation, you will have an internal link to the source code.
#           The source code will have colors defined by the Pygments (syntax highlighting) style.
#           You can checkout the available pygments here: https://help.farbox.com/pygments.html
pygments_style = 'native'

# autodoc: The best thing about Sphinx IMHO is autodoc.
#          It allows Sphinx to automatically generate documentation for the docstrings in your code.
#          Get more info here: http://www.sphinx-doc.org/en/master/ext/autodoc.html
# Some useful configurations:
autoclass_content = "both"  # Include both the class's and the init's docstrings.
autodoc_member_order = 'bysource'  # In the documentation, keep the same order of members as in the code.
autodoc_default_flags = ['members']  # Default: include the docstrings of all the class/module members.

# imgmath: Sphinx allows use of LaTeX in the html documentation, but not directly. It is first rendered to an image.
# You can add here whatever preamble you are used to adding to your LaTeX document.
imgmath_latex_preamble = r'''
\usepackage{xcolor}
\definecolor{offwhite}{rgb}{238,238,238}
\everymath{\color{offwhite}}
\everydisplay{\color{offwhite}}
'''

# Options for LaTeX
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "figure_align": "htbp",
    "preamble": r"""
        \usepackage{listings}
        \lstset{ 
            language=Python,                 % the language of the code
        }
    """,
}
