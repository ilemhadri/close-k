"""Metadata for package to allow installation with pip."""

import setuptools

exec(open("closek/version.py").read())
setuptools.setup(
    name="closek",
    description="Scikit-learn-style implementation of the close-k classifier.",
    author="Bryan He",
    author_email="bryanhe@stanford.edu",
    url="https://arxiv.org/abs/1811.00521",
    version=__version__,
    packages=setuptools.find_packages(),
    install_requires=[
    ],
    tests_require=[
    ],
)

