from setuptools import setup, find_packages

setup(
    name="SL_specdecomp",
    version="0.1.0",
    url="https://github.com/Stephen-Lab-BU/SL_specdecomp",
    license="BSD-3-Clause-Clear",
    author="Patrick F. Bloniasz",
    author_email="patrick.bloniasz@gmail.com",
    description="A Python package for conducting power spectral decomposition.",
    long_description=open("README.rst").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.18.0",
        "matplotlib>=3.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.10",
)