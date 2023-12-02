# setup.py
from setuptools import setup

with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setup(
    name="logivision",
    version="0.1.0",
    packages=["logivision"],
    install_requires=[install_requires],
)
