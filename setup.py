from setuptools import setup


setup(
    name='ardent',
    version='0.1.0',
    description='A Python package for exoplanet dynamical detection limits from RV data',

    url='https://github.com/manustalport/ardent',
    author='Manu Stalport',
    author_email='manu.stalport@uliege.be',
    license = 'BSD 3-Clause',
    packages=['ardent'],
    install_requires=['numpy',
                      'matplotlib',
                      'pandas',
                      'joblib',
                      'tqdm',
                      'PyAstronomy',
                      'rebound==4.4.3',
                      'reboundx==4.3.0'],

    py_modules=['ardent'],

    classifiers=['Intended Audience :: Science/Research',
                 'Programming Language :: Python :: 3'],
)
