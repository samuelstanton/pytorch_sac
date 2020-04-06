from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pytorch-mbpo",
    version="0.0.1",
    author="Samuel Stanton",
    author_email="ss13641@nyu.edu",
    description="MBPO implemented in PyTorch",
    url="https://github.com/samuelstanton/pytorch-mbpo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
    install_requires=["gym"]
)
