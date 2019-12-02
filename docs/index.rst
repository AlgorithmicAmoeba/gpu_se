.. This is A COPY OF the main index.rst file which is rendered into the landing page of your documentation.
   Follow the inline instructions to configure this for YOUR next project.



Welcome to GPU SE documentation !
=========================================================
|

This project creates results required for my Master's thesis in the performance enhancement of
state estimators using graphical processign units.

The source code is available `here <https://github.com/darren-roos/INSERTLINK>`_.

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

   controller
   model
   noise

.. Delete this line until the * to generate index for your project: * :ref:`genindex`

.. Finished personalizing all the relevant details? Great! Now make this your main index.rst,
   And run `make clean latexpdf` from your documentation folder :)
