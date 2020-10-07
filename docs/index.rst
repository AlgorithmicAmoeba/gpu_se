.. This is A COPY OF the main index.rst file which is rendered into the landing page of your documentation.
   Follow the inline instructions to configure this for YOUR next project.



Welcome to GPU SE documentation !
=========================================================
|

This project creates results required for my Master's thesis in the performance enhancement of
state estimators using graphical processing units.

The source code is available `here <https://github.com/darren-roos/gpu_se>`__.
The cache is available `here <github.com/darren-roos/picklejar>`__.
My thesis is available `here <github.com/darren-roos/thesis>`__.

The results available in the cache are from runs performed on a machine with an AMD Ryzen 5 2400G,
32 GB of RAM and a GeForce GTX 1070 (8 GB of on board memory).
The machine was running Ubuntu 18.04.4 LTS with a Python 3.8.2 environment.
Further details about the Python environment can be found in the environment.yml file in
the source code repository.
|

.. maxdepth = 1 means the Table of Contents will only links to the separate pages of the documentation.
   Increasing this number will result in deeper links to subtitles etc.

.. Below is the main Table Of Content
   You have below a "dummy" file, that holds a template for a class.
   To add pages to your documentation:
        * Make a file_name.rst file that follows one of the templates in this project
        * Add its name here to this TOC


.. toctree::
    :maxdepth: 1
    :name: mastertoc

    model
    controller
    gaussian_sums
    filters
    results

.. Delete this line until the * to generate index for your project: * :ref:`genindex`

.. Finished personalizing all the relevant details? Great! Now make this your main index.rst,
   And run `make clean latexpdf` from your documentation folder :)
