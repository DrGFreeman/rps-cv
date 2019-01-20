from setuptools import setup, find_packages
from rpscv import __version__

setup(name='rpscv',
      version=__version__,
      desription='Library for the Rock-Paper-Scissors game using computer vision and machine learning on Raspberry Pi',
      url='https://github.com/DrGFreeman/rps-cv',
      author='Julien de la Bruere-Terreault',
      author_email='drgfreeman@tuta.io',
      licence='MIT',
      packages=find_packages(),
      python_requires='>3.4.0',
      )

