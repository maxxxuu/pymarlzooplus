from setuptools import find_packages, setup
import os

extras = {
    'LBF': ['lbforaging>1.1.1'],
    'LBF_V2': [
        'lbforaging_v2 @ file:./pymarlzooplus/envs/lb-foraging_v2'
    ],
    'RWARE': ['rware>1.0.3'],
    'RWARE_V1': [
        'rware_v1 @ file:./pymarlzooplus/envs/robotics_warehouse_v1'
    ],
    'MPE': [
        'mpe @ file:./pymarlzooplus/envs/multiagent_particle_envs'
    ],
    'PettingZoo': [
        'transformers',
        'pettingzoo',
        "pettingzoo[atari]",
        'autorom',
        "pettingzoo[butterfly]",
        "pettingzoo[mpe]",
        "pettingzoo[sisl]",
        "pettingzoo[classic]"
    ],
    'Overcooked': [
        'overcooked_ai_py @ file:./pymarlzooplus/envs/overcooked_ai'
    ],
    'PressurePlate': [
        'pressureplate @ file:./pymarlzooplus/envs/pressureplate_ai'
    ],
    'CaptureTarget': [
        'capture_target_ai_py @ file:./pymarlzooplus/envs/capture_target'
    ],
    'BoxPushing': [
        'box_pushing_ai_py @ file:./pymarlzooplus/envs/box_pushing'
    ],
}

extras["all"] = sorted(set(dep for deps in extras.values() for dep in deps))

with open('requirements.txt') as f:
    required = f.read().splitlines()

# Check if the DISPLAY environment variable is set and install the corresponding OpenCV version
if 'DISPLAY' in os.environ:
    opencv_pack = "opencv-python"
else:
    opencv_pack = "opencv-python-headless"
required.append(opencv_pack)

setup(
    name='pymarlzooplus',
    version='0.1.0',
    author='AILabDsUnipi',
    author_email='gp.papadopoulos.george@gmail.com',
    description='An extended benchmarking of multi-agent reinforcement learning algorithms',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AILabDsUnipi/pymarlzooplus',
    packages=find_packages(),
    include_package_data=True,
    license='Apache License 2.0',
    install_requires=required,
    extras_require=extras,
    python_requires='>=3.8',
)


