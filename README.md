<h1>KLTPicker2</h1>

KLT Picker: particle picking using data-driven optimal templates.

Current version: 2.0

Project's homepage: https://github.com/ShkolniskyLab/kltpicker2

Date: 01/2022

This is the Python version containing the complete source code of the KLT Picker. A MATLAB version is available at https://github.com/ShkolniskyLab/kltpicker2_matlab.

Please cite the following paper when using this package: 
Eldar A, Landa B, Shkolnisky Y. KLT picker: Particle picking using data-driven optimal templates. J Struct Biol. 2020 May;210(2) 107473.

<h2>Recommended Environments:</h2>
The package has been tested on Ubuntu 16.04 and Windows 10. It should probably work on other versions of Windows and Linux, but has not been tested on them yet. Similarly for macOS.

* Python 3.6.0+ is required.

* The package makes use of the pyfftw package, which in turn uses the FFTW library. Before installing KLTPicker make sure you have the FFTW library installed on your system: http://www.fftw.org/fftw3_doc/Installation-and-Customization.html#Installation-and-Customization

* For **optional** GPU support: see cupy instalation requirments https://docs.cupy.dev/en/stable/install.html.

<h2>Install KLTPicker</h2>
<h3>Install KLTPicker via pip:</h3>
We recommend installing KLTPicker via pip:


    $ pip install kltpicker2

In order to enable the GPU support (provided that your system satisfies the above requirements):


    $ pip install kltpicker2[gpu]

<h3>Install KLTPicker from source</h3>
The tarball of the source tree is available via pip download kltpicker. You can install KLTPicker from the tarball:


    $ pip install kltpicker2-x.x.x.tar.gz


You can also install the development version of KLTPicker from a cloned Git repository:


    $ git clone https://github.com/ShkolniskyLab/kltpicker2.git

    $ cd kltpicker2

    $ pip install .

<h2>Uninstall KLTPicker</h2>
Use pip to uninstall KLTPicker:


    $ pip uninstall kltpicker2

<h2>Upgrade KLTPicker</h2>
Just use pip with -U option:


    $ pip install -U kltpicker2

<h2>Getting started:</h2>
Please read the user manual for usage instructions, available at the homepage of the project on Github: https://github.com/ShkolniskyLab/kltpicker
