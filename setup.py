from setuptools import find_packages, setup

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [line.strip() for line in open("requirements.txt").readlines()]
requirements_dev = [line.strip() for line in open("requirements-dev.txt").readlines()]

setup(
    name="digitalrivers",
    version="0.1.0",
    description="DEM and raster data processing library",
    author="Mostafa Farrag",
    author_email="moah.farag@gmail.come",
    url="https://github.com/MAfarrag/digital-rivers",
    keywords=["GIS", "DEM", "Hydrology"],
    long_description=readme + "\n\n" + history,
    repository="https://github.com/MAfarrag/digital-rivers",
    documentation="https://digital-rivers.readthedocs.io/",
    long_description_content_type="text/markdown",
    license="GNU General Public License v3",
    zip_safe=False,
    packages=find_packages(include=["digitalrivers", "digitalrivers.*"]),
    test_suite="tests",
    tests_require=requirements_dev,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: GIS",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
)
