from setuptools import setup, find_packages

setup(
    name='QuantKit',
    version='0.1.0',
    description='Custom Financial Data Analysis Toolkit',
    author='Rayansh Singh',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'yfinance',
        'plotly',
        'scipy',
        'datetime',
        'requests'
    ],
)
