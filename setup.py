from setuptools import setup, find_packages

setup(
    name="Stockify",
    version="0.1.0",
    author="Rayansh Singh",
    author_email="rayansh365@gmail.com",
    description="A Python library for financial data analysis and portfolio management",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ray-singh/Stockify",
    packages=find_packages(exclude=["tests*", "tests.*"]),  # Exclude tests folder
    install_requires=[
        "yfinance",
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "datetime",
        "seaborn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
