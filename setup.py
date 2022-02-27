from setuptools import find_packages, setup


def get_req():
    with open("requirements.txt", mode="r") as f:
        requirements = f.readlines()
    return requirements


setup(
    name="toolkit",
    version="0.1.0",
    description="Python Distribution Utilities for images, bokeh",
    author="Mattia Federici",
    author_email="mattia.federici@orobix.it",
    packages=find_packages(),
    license="MIT",
    python_requires=">3",
    install_requires=get_req(),
)
