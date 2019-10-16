# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from os.path import join
from re import search

PACKAGE_NAME = "blrlib"

with open(join(PACKAGE_NAME, "__init__.py"), encoding="utf-8") as f:
    version = search(r"__version__\s+=\s+(.*)", f.read()).group(1)

with open("README.rst", encoding="utf-8") as f:
    readme = f.read()

setup(
    name=PACKAGE_NAME,
    version=version,
    license="MIT",
    description="The Package for Block Low Rank Matrix Computations.",
    long_description=readme,
    author="tbs-lab",
    author_email="tsubasa.i.0516@gmail.com",
    url="https://tbs-lab.github.io",
    project_urls={},
    packages=find_packages(PACKAGE_NAME),
    package_dir={"": PACKAGE_NAME},
    install_requires=["numpy", "scipy"],
)
