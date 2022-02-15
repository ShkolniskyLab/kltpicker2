from setuptools import setup, find_packages
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='kltpicker2',
      version='2.0.1',
      description='KLT Picker: Particle picking using data-driven optimal templates',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/ShkolniskyLab/kltpicker2',
      author='Yoel Shkolnisky',
      author_email='yoelsh@tauex.tau.ac.il',
      packages=find_packages(),
      license='GNU General Public License v3.0',
      entry_points={
        "console_scripts": ['kltpicker = kltpicker.main:main']
        },
      install_requires=[
          'numpy>=1.16',
          'mrcfile',
          'scipy>=1.3',
          'pyfftw',
          'progressbar2',
          'numba',
          'scikit-image',
          'opencv-python'
      ],
      extras_require={'gpu': ['cupy']},
      python_requires='~=3.6',
      zip_safe=False)
