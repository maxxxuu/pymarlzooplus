#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="overcooked_ai_py",
    version="1.1.0",
    description="Cooperative multi-agent environment based on Overcooked",
    long_description_content_type="text/markdown",
    author="Micah Carroll",
    author_email="mdc@berkeley.edu",
    url="https://github.com/HumanCompatibleAI/overcooked_ai",
    packages=find_packages("src"),
    package_dir={"": "src"},
    keywords=["Overcooked", "AI", "Reinforcement Learning"],
    include_package_data=True,
    package_data={
        "overcooked_ai_py": [
            "data/layouts/*.layout",
            "data/planners/*.py",
            "data/graphics/*.png",
            "data/graphics/*.json",
            "data/fonts/*.ttf",
        ],
    },
    install_requires=[
        "dill",
        "numpy",
        "scipy",
        "tqdm",
        "gymnasium",
        "pettingzoo",
        "ipython",
        "pygame",
        "ipywidgets",
    ],
)
