from setuptools import setup, find_packages

setup(
    name="pressureplate",
    version="0.0.1",
    description="Multi-agent environment for reinforcement learning",
    long_description_content_type="text/markdown",
    author="Trevor McInroe",
    url="https://github.com/semitable/boulder-push",
    packages=find_packages(),
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
    ],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
