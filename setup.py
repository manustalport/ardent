from setuptools import setup

from ardent import __version__

setup(
    name='ardent',
    version=__version__,
    description='A Python package for exoplanet dynamical detection limits from RV data',

    url='https://github.com/manustalport/ardent',
    author='Manu Stalport',
    author_email='manu.stalport@uliege.be',
    packages=['ardent'],
    install_requires=['rebound==4.4.3', 'reboundx==4.3.0'],

    py_modules=['ardent'],

    classifiers=['Intended Audience :: Science/Research',
                 'Programming Language :: Python :: 3'],
)
