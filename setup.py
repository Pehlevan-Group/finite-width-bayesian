from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
INSTALL_REQUIRES = []

setup(
    name='finite-width-bayesian',
    license='MIT License',
    author='Pehlevan Group',
    author_email='canatara@gmail.com',
    install_requires=INSTALL_REQUIRES,
    url='https://github.com/Pehlevan-Group/finite-width-bayesian',
    long_description=long_description,
    packages=find_packages(),
    long_description_content_type='text/markdown',
    description='Representatian Learning in Wide Bayesian Neural Networks',
    python_requires='>=3.6')
