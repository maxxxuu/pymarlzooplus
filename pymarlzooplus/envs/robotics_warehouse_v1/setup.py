import pathlib
from setuptools import setup, find_packages
# The directory containing this file
HERE = pathlib.Path(__file__).parent

setup(
    name="rware_v1",
    version="1.0.3",
    description="Multi-Robot Warehouse environment for reinforcement learning",
    long_description_content_type="text/markdown",
    author="Filippos Christianos",
    url="https://github.com/semitable/robotic-warehouse",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=[
        "numpy",
        "gymnasium",
        "pyglet",
        "networkx",
        "tqdm",
    ],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
