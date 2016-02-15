#!/usr/bin/env python
from setuptools import setup, find_packages

setup (
    name = "scape",
    description = "Karoo Array Telescope common astronomical library'",
    author = "Ludwig Schwardt",
    author_email = "ludwig@ska.ac.za",
    packages = find_packages(),
    url='http://ska.ac.za/',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    platforms = [ "OS Independent" ],
    install_requires = ['nose',],
    keywords="kat kat7 ska",
    zip_safe = False,
    setup_requires=['katversion'],
    use_katversion=True,
    # Bitten Test Suite
    test_suite = "nose.collector",
)
