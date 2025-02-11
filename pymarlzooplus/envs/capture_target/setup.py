#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="capture_target",
    version="1.0.0",
    description="Multi-agent-single-target (MAST) environment based on Capture Target",
    long_description_content_type="text/markdown",
    url="https://github.com/yuchen-x/MacDeepMARL",
    packages=find_packages("src"),
    keywords=["Capture Target", "AI", "Reinforcement Learning"],
    package_data={
            "capture_target_ai_py": [
                "*.py"
            ]},
    install_requires=[
        "numpy",
        "pyglet",
        "gymnasium"
    ],
    include_package_data=True,
)
